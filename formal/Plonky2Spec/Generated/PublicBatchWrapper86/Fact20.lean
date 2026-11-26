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

theorem publicBatchWrapper86_f15449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15655)) := by
  have c97047 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97048 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97049 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97050 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97051 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97052 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_14 := arithEq_of_rows h (row := 1585) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_14
  simp only [← c97047, ← c97048, ← c97049] at e_1585_14
  have e_1585_15 := arithEq_of_rows h (row := 1585) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_15
  simp only [← c97050, ← c97051, k2, ← c97052] at e_1585_15
  have hr := e_1585_15
  simp only [e_1585_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15656)) := by
  have c97053 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97054 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97055 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97056 := (publicBatchWrapper86_copies3033 a h).1
  have c97057 := (publicBatchWrapper86_copies3033 a h).2.1
  have c97058 := (publicBatchWrapper86_copies3033 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_16 := arithEq_of_rows h (row := 1585) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_16
  simp only [← c97053, ← c97054, ← c97055] at e_1585_16
  have e_1585_17 := arithEq_of_rows h (row := 1585) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_17
  simp only [← c97056, ← c97057, k2, ← c97058] at e_1585_17
  have hr := e_1585_17
  simp only [e_1585_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15657)) := by
  have c97059 := (publicBatchWrapper86_copies3033 a h).2.2.2.1
  have c97060 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.1
  have c97061 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.1
  have c97062 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.1
  have c97063 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.1
  have c97064 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_18 := arithEq_of_rows h (row := 1585) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_18
  simp only [← c97059, ← c97060, ← c97061] at e_1585_18
  have e_1585_19 := arithEq_of_rows h (row := 1585) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_19
  simp only [← c97062, ← c97063, k2, ← c97064] at e_1585_19
  have hr := e_1585_19
  simp only [e_1585_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1586 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15658)) := by
  have c97065 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.1
  have c97066 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97067 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97068 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97069 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97070 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1586_0 := arithEq_of_rows h (row := 1586) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_0
  simp only [← c97065, ← c97066, ← c97067] at e_1586_0
  have e_1586_1 := arithEq_of_rows h (row := 1586) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_1
  simp only [← c97068, ← c97069, k2, ← c97070] at e_1586_1
  have hr := e_1586_1
  simp only [e_1586_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1586 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15659)) := by
  have c97071 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97072 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97073 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97074 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97075 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97076 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1586_2 := arithEq_of_rows h (row := 1586) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_2
  simp only [← c97071, ← c97072, ← c97073] at e_1586_2
  have e_1586_3 := arithEq_of_rows h (row := 1586) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_3
  simp only [← c97074, ← c97075, k2, ← c97076] at e_1586_3
  have hr := e_1586_3
  simp only [e_1586_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1586 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15660)) := by
  have c97077 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97078 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97079 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97080 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97081 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97082 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1586_4 := arithEq_of_rows h (row := 1586) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_4
  simp only [← c97077, ← c97078, ← c97079] at e_1586_4
  have e_1586_5 := arithEq_of_rows h (row := 1586) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_5
  simp only [← c97080, ← c97081, k2, ← c97082] at e_1586_5
  have hr := e_1586_5
  simp only [e_1586_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1586 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15661)) := by
  have c97083 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97084 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97085 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97086 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97087 := (publicBatchWrapper86_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97088 := (publicBatchWrapper86_copies3034 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1586_6 := arithEq_of_rows h (row := 1586) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_6
  simp only [← c97083, ← c97084, ← c97085] at e_1586_6
  have e_1586_7 := arithEq_of_rows h (row := 1586) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_7
  simp only [← c97086, ← c97087, k2, ← c97088] at e_1586_7
  have hr := e_1586_7
  simp only [e_1586_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1586 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15662)) := by
  have c97089 := (publicBatchWrapper86_copies3034 a h).2.1
  have c97090 := (publicBatchWrapper86_copies3034 a h).2.2.1
  have c97091 := (publicBatchWrapper86_copies3034 a h).2.2.2.1
  have c97092 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.1
  have c97093 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.1
  have c97094 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1586_8 := arithEq_of_rows h (row := 1586) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_8
  simp only [← c97089, ← c97090, ← c97091] at e_1586_8
  have e_1586_9 := arithEq_of_rows h (row := 1586) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_9
  simp only [← c97092, ← c97093, k2, ← c97094] at e_1586_9
  have hr := e_1586_9
  simp only [e_1586_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1586 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15663)) := by
  have c97095 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.1
  have c97096 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.1
  have c97097 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.1
  have c97098 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97099 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97100 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1586_10 := arithEq_of_rows h (row := 1586) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_10
  simp only [← c97095, ← c97096, ← c97097] at e_1586_10
  have e_1586_11 := arithEq_of_rows h (row := 1586) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_11
  simp only [← c97098, ← c97099, k2, ← c97100] at e_1586_11
  have hr := e_1586_11
  simp only [e_1586_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1586 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15664)) := by
  have c97101 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97102 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97103 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97104 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97105 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97106 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1586_12 := arithEq_of_rows h (row := 1586) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_12
  simp only [← c97101, ← c97102, ← c97103] at e_1586_12
  have e_1586_13 := arithEq_of_rows h (row := 1586) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_13
  simp only [← c97104, ← c97105, k2, ← c97106] at e_1586_13
  have hr := e_1586_13
  simp only [e_1586_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1586 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15665)) := by
  have c97107 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97108 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97109 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97110 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97111 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97112 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1586_14 := arithEq_of_rows h (row := 1586) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_14
  simp only [← c97107, ← c97108, ← c97109] at e_1586_14
  have e_1586_15 := arithEq_of_rows h (row := 1586) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_15
  simp only [← c97110, ← c97111, k2, ← c97112] at e_1586_15
  have hr := e_1586_15
  simp only [e_1586_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1586 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15666)) := by
  have c97113 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97114 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97115 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97116 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97117 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97118 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1586_16 := arithEq_of_rows h (row := 1586) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_16
  simp only [← c97113, ← c97114, ← c97115] at e_1586_16
  have e_1586_17 := arithEq_of_rows h (row := 1586) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_17
  simp only [← c97116, ← c97117, k2, ← c97118] at e_1586_17
  have hr := e_1586_17
  simp only [e_1586_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1586 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15667)) := by
  have c97119 := (publicBatchWrapper86_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97120 := (publicBatchWrapper86_copies3035 a h).1
  have c97121 := (publicBatchWrapper86_copies3035 a h).2.1
  have c97122 := (publicBatchWrapper86_copies3035 a h).2.2.1
  have c97123 := (publicBatchWrapper86_copies3035 a h).2.2.2.1
  have c97124 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1586_18 := arithEq_of_rows h (row := 1586) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_18
  simp only [← c97119, ← c97120, ← c97121] at e_1586_18
  have e_1586_19 := arithEq_of_rows h (row := 1586) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_19
  simp only [← c97122, ← c97123, k2, ← c97124] at e_1586_19
  have hr := e_1586_19
  simp only [e_1586_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1587 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15668)) := by
  have c97125 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.1
  have c97126 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.1
  have c97127 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.1
  have c97128 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.1
  have c97129 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.1
  have c97130 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1587_0 := arithEq_of_rows h (row := 1587) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_0
  simp only [← c97125, ← c97126, ← c97127] at e_1587_0
  have e_1587_1 := arithEq_of_rows h (row := 1587) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_1
  simp only [← c97128, ← c97129, k2, ← c97130] at e_1587_1
  have hr := e_1587_1
  simp only [e_1587_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1587 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15669)) := by
  have c97131 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97132 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97133 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97134 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97135 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97136 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1587_2 := arithEq_of_rows h (row := 1587) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_2
  simp only [← c97131, ← c97132, ← c97133] at e_1587_2
  have e_1587_3 := arithEq_of_rows h (row := 1587) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_3
  simp only [← c97134, ← c97135, k2, ← c97136] at e_1587_3
  have hr := e_1587_3
  simp only [e_1587_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1587 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15670)) := by
  have c97137 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97138 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97139 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97140 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97141 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97142 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1587_4 := arithEq_of_rows h (row := 1587) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_4
  simp only [← c97137, ← c97138, ← c97139] at e_1587_4
  have e_1587_5 := arithEq_of_rows h (row := 1587) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_5
  simp only [← c97140, ← c97141, k2, ← c97142] at e_1587_5
  have hr := e_1587_5
  simp only [e_1587_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1587 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15671)) := by
  have c97143 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97144 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97145 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97146 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97147 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97148 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1587_6 := arithEq_of_rows h (row := 1587) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_6
  simp only [← c97143, ← c97144, ← c97145] at e_1587_6
  have e_1587_7 := arithEq_of_rows h (row := 1587) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_7
  simp only [← c97146, ← c97147, k2, ← c97148] at e_1587_7
  have hr := e_1587_7
  simp only [e_1587_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1587 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15672)) := by
  have c97149 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97150 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97151 := (publicBatchWrapper86_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97152 := (publicBatchWrapper86_copies3036 a h).1
  have c97153 := (publicBatchWrapper86_copies3036 a h).2.1
  have c97154 := (publicBatchWrapper86_copies3036 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1587_8 := arithEq_of_rows h (row := 1587) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_8
  simp only [← c97149, ← c97150, ← c97151] at e_1587_8
  have e_1587_9 := arithEq_of_rows h (row := 1587) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_9
  simp only [← c97152, ← c97153, k2, ← c97154] at e_1587_9
  have hr := e_1587_9
  simp only [e_1587_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1587 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15673)) := by
  have c97155 := (publicBatchWrapper86_copies3036 a h).2.2.2.1
  have c97156 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.1
  have c97157 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.1
  have c97158 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.1
  have c97159 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.1
  have c97160 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1587_10 := arithEq_of_rows h (row := 1587) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_10
  simp only [← c97155, ← c97156, ← c97157] at e_1587_10
  have e_1587_11 := arithEq_of_rows h (row := 1587) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_11
  simp only [← c97158, ← c97159, k2, ← c97160] at e_1587_11
  have hr := e_1587_11
  simp only [e_1587_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1587 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15674)) := by
  have c97161 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.1
  have c97162 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97163 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97164 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97165 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97166 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1587_12 := arithEq_of_rows h (row := 1587) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_12
  simp only [← c97161, ← c97162, ← c97163] at e_1587_12
  have e_1587_13 := arithEq_of_rows h (row := 1587) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_13
  simp only [← c97164, ← c97165, k2, ← c97166] at e_1587_13
  have hr := e_1587_13
  simp only [e_1587_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1587 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15675)) := by
  have c97167 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97168 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97169 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97170 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97171 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97172 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1587_14 := arithEq_of_rows h (row := 1587) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_14
  simp only [← c97167, ← c97168, ← c97169] at e_1587_14
  have e_1587_15 := arithEq_of_rows h (row := 1587) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_15
  simp only [← c97170, ← c97171, k2, ← c97172] at e_1587_15
  have hr := e_1587_15
  simp only [e_1587_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1587 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15676)) := by
  have c97173 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97174 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97175 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97176 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97177 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97178 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1587_16 := arithEq_of_rows h (row := 1587) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_16
  simp only [← c97173, ← c97174, ← c97175] at e_1587_16
  have e_1587_17 := arithEq_of_rows h (row := 1587) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_17
  simp only [← c97176, ← c97177, k2, ← c97178] at e_1587_17
  have hr := e_1587_17
  simp only [e_1587_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1587 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15677)) := by
  have c97179 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97180 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97181 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97182 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97183 := (publicBatchWrapper86_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97184 := (publicBatchWrapper86_copies3037 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1587_18 := arithEq_of_rows h (row := 1587) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_18
  simp only [← c97179, ← c97180, ← c97181] at e_1587_18
  have e_1587_19 := arithEq_of_rows h (row := 1587) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_19
  simp only [← c97182, ← c97183, k2, ← c97184] at e_1587_19
  have hr := e_1587_19
  simp only [e_1587_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1588 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15678)) := by
  have c97185 := (publicBatchWrapper86_copies3037 a h).2.1
  have c97186 := (publicBatchWrapper86_copies3037 a h).2.2.1
  have c97187 := (publicBatchWrapper86_copies3037 a h).2.2.2.1
  have c97188 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.1
  have c97189 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.1
  have c97190 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1588_0 := arithEq_of_rows h (row := 1588) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_0
  simp only [← c97185, ← c97186, ← c97187] at e_1588_0
  have e_1588_1 := arithEq_of_rows h (row := 1588) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_1
  simp only [← c97188, ← c97189, k2, ← c97190] at e_1588_1
  have hr := e_1588_1
  simp only [e_1588_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1588 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15679)) := by
  have c97191 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.1
  have c97192 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.1
  have c97193 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.1
  have c97194 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97195 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97196 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1588_2 := arithEq_of_rows h (row := 1588) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_2
  simp only [← c97191, ← c97192, ← c97193] at e_1588_2
  have e_1588_3 := arithEq_of_rows h (row := 1588) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_3
  simp only [← c97194, ← c97195, k2, ← c97196] at e_1588_3
  have hr := e_1588_3
  simp only [e_1588_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1588 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15680)) := by
  have c97197 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97198 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97199 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97200 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97201 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97202 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1588_4 := arithEq_of_rows h (row := 1588) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_4
  simp only [← c97197, ← c97198, ← c97199] at e_1588_4
  have e_1588_5 := arithEq_of_rows h (row := 1588) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_5
  simp only [← c97200, ← c97201, k2, ← c97202] at e_1588_5
  have hr := e_1588_5
  simp only [e_1588_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1588 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15681)) := by
  have c97203 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97204 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97205 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97206 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97207 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97208 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1588_6 := arithEq_of_rows h (row := 1588) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_6
  simp only [← c97203, ← c97204, ← c97205] at e_1588_6
  have e_1588_7 := arithEq_of_rows h (row := 1588) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_7
  simp only [← c97206, ← c97207, k2, ← c97208] at e_1588_7
  have hr := e_1588_7
  simp only [e_1588_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1588 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15682)) := by
  have c97209 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97210 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97211 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97212 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97213 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97214 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1588_8 := arithEq_of_rows h (row := 1588) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_8
  simp only [← c97209, ← c97210, ← c97211] at e_1588_8
  have e_1588_9 := arithEq_of_rows h (row := 1588) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_9
  simp only [← c97212, ← c97213, k2, ← c97214] at e_1588_9
  have hr := e_1588_9
  simp only [e_1588_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1588 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15683)) := by
  have c97215 := (publicBatchWrapper86_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97216 := (publicBatchWrapper86_copies3038 a h).1
  have c97217 := (publicBatchWrapper86_copies3038 a h).2.1
  have c97218 := (publicBatchWrapper86_copies3038 a h).2.2.1
  have c97219 := (publicBatchWrapper86_copies3038 a h).2.2.2.1
  have c97220 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1588_10 := arithEq_of_rows h (row := 1588) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_10
  simp only [← c97215, ← c97216, ← c97217] at e_1588_10
  have e_1588_11 := arithEq_of_rows h (row := 1588) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_11
  simp only [← c97218, ← c97219, k2, ← c97220] at e_1588_11
  have hr := e_1588_11
  simp only [e_1588_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1588 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15684)) := by
  have c97221 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.1
  have c97222 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.1
  have c97223 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.1
  have c97224 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.1
  have c97225 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.1
  have c97226 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1588_12 := arithEq_of_rows h (row := 1588) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_12
  simp only [← c97221, ← c97222, ← c97223] at e_1588_12
  have e_1588_13 := arithEq_of_rows h (row := 1588) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_13
  simp only [← c97224, ← c97225, k2, ← c97226] at e_1588_13
  have hr := e_1588_13
  simp only [e_1588_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1588 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15685)) := by
  have c97227 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97228 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97229 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97230 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97231 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97232 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1588_14 := arithEq_of_rows h (row := 1588) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_14
  simp only [← c97227, ← c97228, ← c97229] at e_1588_14
  have e_1588_15 := arithEq_of_rows h (row := 1588) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_15
  simp only [← c97230, ← c97231, k2, ← c97232] at e_1588_15
  have hr := e_1588_15
  simp only [e_1588_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1588 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15686)) := by
  have c97233 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97234 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97235 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97236 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97237 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97238 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1588_16 := arithEq_of_rows h (row := 1588) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_16
  simp only [← c97233, ← c97234, ← c97235] at e_1588_16
  have e_1588_17 := arithEq_of_rows h (row := 1588) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_17
  simp only [← c97236, ← c97237, k2, ← c97238] at e_1588_17
  have hr := e_1588_17
  simp only [e_1588_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1588 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15687)) := by
  have c97239 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97240 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97241 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97242 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97243 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97244 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1588_18 := arithEq_of_rows h (row := 1588) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_18
  simp only [← c97239, ← c97240, ← c97241] at e_1588_18
  have e_1588_19 := arithEq_of_rows h (row := 1588) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_19
  simp only [← c97242, ← c97243, k2, ← c97244] at e_1588_19
  have hr := e_1588_19
  simp only [e_1588_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1589 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15688)) := by
  have c97245 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97246 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97247 := (publicBatchWrapper86_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97248 := (publicBatchWrapper86_copies3039 a h).1
  have c97249 := (publicBatchWrapper86_copies3039 a h).2.1
  have c97250 := (publicBatchWrapper86_copies3039 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1589_0 := arithEq_of_rows h (row := 1589) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_0
  simp only [← c97245, ← c97246, ← c97247] at e_1589_0
  have e_1589_1 := arithEq_of_rows h (row := 1589) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_1
  simp only [← c97248, ← c97249, k2, ← c97250] at e_1589_1
  have hr := e_1589_1
  simp only [e_1589_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1589 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15689)) := by
  have c97251 := (publicBatchWrapper86_copies3039 a h).2.2.2.1
  have c97252 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.1
  have c97253 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.1
  have c97254 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.1
  have c97255 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.1
  have c97256 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1589_2 := arithEq_of_rows h (row := 1589) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_2
  simp only [← c97251, ← c97252, ← c97253] at e_1589_2
  have e_1589_3 := arithEq_of_rows h (row := 1589) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_3
  simp only [← c97254, ← c97255, k2, ← c97256] at e_1589_3
  have hr := e_1589_3
  simp only [e_1589_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1589 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15690)) := by
  have c97257 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.1
  have c97258 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97259 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97260 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97261 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97262 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1589_4 := arithEq_of_rows h (row := 1589) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_4
  simp only [← c97257, ← c97258, ← c97259] at e_1589_4
  have e_1589_5 := arithEq_of_rows h (row := 1589) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_5
  simp only [← c97260, ← c97261, k2, ← c97262] at e_1589_5
  have hr := e_1589_5
  simp only [e_1589_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1589 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15691)) := by
  have c97263 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97264 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97265 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97266 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97267 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97268 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1589_6 := arithEq_of_rows h (row := 1589) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_6
  simp only [← c97263, ← c97264, ← c97265] at e_1589_6
  have e_1589_7 := arithEq_of_rows h (row := 1589) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_7
  simp only [← c97266, ← c97267, k2, ← c97268] at e_1589_7
  have hr := e_1589_7
  simp only [e_1589_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1589 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15692)) := by
  have c97269 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97270 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97271 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97272 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97273 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97274 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1589_8 := arithEq_of_rows h (row := 1589) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_8
  simp only [← c97269, ← c97270, ← c97271] at e_1589_8
  have e_1589_9 := arithEq_of_rows h (row := 1589) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_9
  simp only [← c97272, ← c97273, k2, ← c97274] at e_1589_9
  have hr := e_1589_9
  simp only [e_1589_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1589 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15693)) := by
  have c97275 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97276 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97277 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97278 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97279 := (publicBatchWrapper86_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97280 := (publicBatchWrapper86_copies3040 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1589_10 := arithEq_of_rows h (row := 1589) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_10
  simp only [← c97275, ← c97276, ← c97277] at e_1589_10
  have e_1589_11 := arithEq_of_rows h (row := 1589) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_11
  simp only [← c97278, ← c97279, k2, ← c97280] at e_1589_11
  have hr := e_1589_11
  simp only [e_1589_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1589 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15694)) := by
  have c97281 := (publicBatchWrapper86_copies3040 a h).2.1
  have c97282 := (publicBatchWrapper86_copies3040 a h).2.2.1
  have c97283 := (publicBatchWrapper86_copies3040 a h).2.2.2.1
  have c97284 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.1
  have c97285 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.1
  have c97286 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1589_12 := arithEq_of_rows h (row := 1589) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_12
  simp only [← c97281, ← c97282, ← c97283] at e_1589_12
  have e_1589_13 := arithEq_of_rows h (row := 1589) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_13
  simp only [← c97284, ← c97285, k2, ← c97286] at e_1589_13
  have hr := e_1589_13
  simp only [e_1589_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1589 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15695)) := by
  have c97287 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.1
  have c97288 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.1
  have c97289 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.1
  have c97290 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97291 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97292 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1589_14 := arithEq_of_rows h (row := 1589) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_14
  simp only [← c97287, ← c97288, ← c97289] at e_1589_14
  have e_1589_15 := arithEq_of_rows h (row := 1589) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_15
  simp only [← c97290, ← c97291, k2, ← c97292] at e_1589_15
  have hr := e_1589_15
  simp only [e_1589_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1589 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15696)) := by
  have c97293 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97294 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97295 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97296 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97297 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97298 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1589_16 := arithEq_of_rows h (row := 1589) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_16
  simp only [← c97293, ← c97294, ← c97295] at e_1589_16
  have e_1589_17 := arithEq_of_rows h (row := 1589) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_17
  simp only [← c97296, ← c97297, k2, ← c97298] at e_1589_17
  have hr := e_1589_17
  simp only [e_1589_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1589 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15697)) := by
  have c97299 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97300 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97301 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97302 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97303 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97304 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1589_18 := arithEq_of_rows h (row := 1589) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_18
  simp only [← c97299, ← c97300, ← c97301] at e_1589_18
  have e_1589_19 := arithEq_of_rows h (row := 1589) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_19
  simp only [← c97302, ← c97303, k2, ← c97304] at e_1589_19
  have hr := e_1589_19
  simp only [e_1589_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1590 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15698)) := by
  have c97305 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97306 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97307 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97308 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97309 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97310 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1590_0 := arithEq_of_rows h (row := 1590) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_0
  simp only [← c97305, ← c97306, ← c97307] at e_1590_0
  have e_1590_1 := arithEq_of_rows h (row := 1590) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_1
  simp only [← c97308, ← c97309, k2, ← c97310] at e_1590_1
  have hr := e_1590_1
  simp only [e_1590_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1590 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15699)) := by
  have c97311 := (publicBatchWrapper86_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97312 := (publicBatchWrapper86_copies3041 a h).1
  have c97313 := (publicBatchWrapper86_copies3041 a h).2.1
  have c97314 := (publicBatchWrapper86_copies3041 a h).2.2.1
  have c97315 := (publicBatchWrapper86_copies3041 a h).2.2.2.1
  have c97316 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1590_2 := arithEq_of_rows h (row := 1590) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_2
  simp only [← c97311, ← c97312, ← c97313] at e_1590_2
  have e_1590_3 := arithEq_of_rows h (row := 1590) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_3
  simp only [← c97314, ← c97315, k2, ← c97316] at e_1590_3
  have hr := e_1590_3
  simp only [e_1590_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1590 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15700)) := by
  have c97317 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.1
  have c97318 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.1
  have c97319 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.1
  have c97320 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.1
  have c97321 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.1
  have c97322 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1590_4 := arithEq_of_rows h (row := 1590) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_4
  simp only [← c97317, ← c97318, ← c97319] at e_1590_4
  have e_1590_5 := arithEq_of_rows h (row := 1590) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_5
  simp only [← c97320, ← c97321, k2, ← c97322] at e_1590_5
  have hr := e_1590_5
  simp only [e_1590_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1590 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15701)) := by
  have c97323 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97324 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97325 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97326 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97327 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97328 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1590_6 := arithEq_of_rows h (row := 1590) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_6
  simp only [← c97323, ← c97324, ← c97325] at e_1590_6
  have e_1590_7 := arithEq_of_rows h (row := 1590) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_7
  simp only [← c97326, ← c97327, k2, ← c97328] at e_1590_7
  have hr := e_1590_7
  simp only [e_1590_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1590 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15702)) := by
  have c97329 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97330 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97331 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97332 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97333 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97334 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1590_8 := arithEq_of_rows h (row := 1590) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_8
  simp only [← c97329, ← c97330, ← c97331] at e_1590_8
  have e_1590_9 := arithEq_of_rows h (row := 1590) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_9
  simp only [← c97332, ← c97333, k2, ← c97334] at e_1590_9
  have hr := e_1590_9
  simp only [e_1590_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1590 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15703)) := by
  have c97335 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97336 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97337 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97338 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97339 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97340 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1590_10 := arithEq_of_rows h (row := 1590) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_10
  simp only [← c97335, ← c97336, ← c97337] at e_1590_10
  have e_1590_11 := arithEq_of_rows h (row := 1590) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_11
  simp only [← c97338, ← c97339, k2, ← c97340] at e_1590_11
  have hr := e_1590_11
  simp only [e_1590_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1590 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15704)) := by
  have c97341 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97342 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97343 := (publicBatchWrapper86_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97344 := (publicBatchWrapper86_copies3042 a h).1
  have c97345 := (publicBatchWrapper86_copies3042 a h).2.1
  have c97346 := (publicBatchWrapper86_copies3042 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1590_12 := arithEq_of_rows h (row := 1590) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_12
  simp only [← c97341, ← c97342, ← c97343] at e_1590_12
  have e_1590_13 := arithEq_of_rows h (row := 1590) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_13
  simp only [← c97344, ← c97345, k2, ← c97346] at e_1590_13
  have hr := e_1590_13
  simp only [e_1590_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1590 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15705)) := by
  have c97347 := (publicBatchWrapper86_copies3042 a h).2.2.2.1
  have c97348 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.1
  have c97349 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.1
  have c97350 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.1
  have c97351 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.1
  have c97352 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1590_14 := arithEq_of_rows h (row := 1590) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_14
  simp only [← c97347, ← c97348, ← c97349] at e_1590_14
  have e_1590_15 := arithEq_of_rows h (row := 1590) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_15
  simp only [← c97350, ← c97351, k2, ← c97352] at e_1590_15
  have hr := e_1590_15
  simp only [e_1590_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1590 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15706)) := by
  have c97353 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.1
  have c97354 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97355 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97356 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97357 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97358 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1590_16 := arithEq_of_rows h (row := 1590) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_16
  simp only [← c97353, ← c97354, ← c97355] at e_1590_16
  have e_1590_17 := arithEq_of_rows h (row := 1590) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_17
  simp only [← c97356, ← c97357, k2, ← c97358] at e_1590_17
  have hr := e_1590_17
  simp only [e_1590_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1590 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15707)) := by
  have c97359 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97360 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97361 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97362 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97363 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97364 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1590_18 := arithEq_of_rows h (row := 1590) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_18
  simp only [← c97359, ← c97360, ← c97361] at e_1590_18
  have e_1590_19 := arithEq_of_rows h (row := 1590) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_19
  simp only [← c97362, ← c97363, k2, ← c97364] at e_1590_19
  have hr := e_1590_19
  simp only [e_1590_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1591 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15708)) := by
  have c97365 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97366 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97367 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97368 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97369 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97370 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1591_0 := arithEq_of_rows h (row := 1591) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_0
  simp only [← c97365, ← c97366, ← c97367] at e_1591_0
  have e_1591_1 := arithEq_of_rows h (row := 1591) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_1
  simp only [← c97368, ← c97369, k2, ← c97370] at e_1591_1
  have hr := e_1591_1
  simp only [e_1591_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1591 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15709)) := by
  have c97371 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97372 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97373 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97374 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97375 := (publicBatchWrapper86_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97376 := (publicBatchWrapper86_copies3043 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1591_2 := arithEq_of_rows h (row := 1591) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_2
  simp only [← c97371, ← c97372, ← c97373] at e_1591_2
  have e_1591_3 := arithEq_of_rows h (row := 1591) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_3
  simp only [← c97374, ← c97375, k2, ← c97376] at e_1591_3
  have hr := e_1591_3
  simp only [e_1591_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1591 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15710)) := by
  have c97377 := (publicBatchWrapper86_copies3043 a h).2.1
  have c97378 := (publicBatchWrapper86_copies3043 a h).2.2.1
  have c97379 := (publicBatchWrapper86_copies3043 a h).2.2.2.1
  have c97380 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.1
  have c97381 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.1
  have c97382 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1591_4 := arithEq_of_rows h (row := 1591) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_4
  simp only [← c97377, ← c97378, ← c97379] at e_1591_4
  have e_1591_5 := arithEq_of_rows h (row := 1591) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_5
  simp only [← c97380, ← c97381, k2, ← c97382] at e_1591_5
  have hr := e_1591_5
  simp only [e_1591_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1591 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15711)) := by
  have c97383 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.1
  have c97384 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.1
  have c97385 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.1
  have c97386 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97387 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97388 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1591_6 := arithEq_of_rows h (row := 1591) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_6
  simp only [← c97383, ← c97384, ← c97385] at e_1591_6
  have e_1591_7 := arithEq_of_rows h (row := 1591) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_7
  simp only [← c97386, ← c97387, k2, ← c97388] at e_1591_7
  have hr := e_1591_7
  simp only [e_1591_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1591 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15712)) := by
  have c97389 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97390 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97391 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97392 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97393 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97394 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1591_8 := arithEq_of_rows h (row := 1591) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_8
  simp only [← c97389, ← c97390, ← c97391] at e_1591_8
  have e_1591_9 := arithEq_of_rows h (row := 1591) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_9
  simp only [← c97392, ← c97393, k2, ← c97394] at e_1591_9
  have hr := e_1591_9
  simp only [e_1591_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1591 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15713)) := by
  have c97395 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97396 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97397 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97398 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97399 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97400 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1591_10 := arithEq_of_rows h (row := 1591) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_10
  simp only [← c97395, ← c97396, ← c97397] at e_1591_10
  have e_1591_11 := arithEq_of_rows h (row := 1591) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_11
  simp only [← c97398, ← c97399, k2, ← c97400] at e_1591_11
  have hr := e_1591_11
  simp only [e_1591_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1591 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15714)) := by
  have c97401 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97402 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97403 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97404 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97405 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97406 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1591_12 := arithEq_of_rows h (row := 1591) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_12
  simp only [← c97401, ← c97402, ← c97403] at e_1591_12
  have e_1591_13 := arithEq_of_rows h (row := 1591) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_13
  simp only [← c97404, ← c97405, k2, ← c97406] at e_1591_13
  have hr := e_1591_13
  simp only [e_1591_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1591 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15715)) := by
  have c97407 := (publicBatchWrapper86_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97408 := (publicBatchWrapper86_copies3044 a h).1
  have c97409 := (publicBatchWrapper86_copies3044 a h).2.1
  have c97410 := (publicBatchWrapper86_copies3044 a h).2.2.1
  have c97411 := (publicBatchWrapper86_copies3044 a h).2.2.2.1
  have c97412 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1591_14 := arithEq_of_rows h (row := 1591) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_14
  simp only [← c97407, ← c97408, ← c97409] at e_1591_14
  have e_1591_15 := arithEq_of_rows h (row := 1591) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_15
  simp only [← c97410, ← c97411, k2, ← c97412] at e_1591_15
  have hr := e_1591_15
  simp only [e_1591_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1591 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15716)) := by
  have c97413 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.1
  have c97414 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.1
  have c97415 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.1
  have c97416 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.1
  have c97417 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.1
  have c97418 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1591_16 := arithEq_of_rows h (row := 1591) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_16
  simp only [← c97413, ← c97414, ← c97415] at e_1591_16
  have e_1591_17 := arithEq_of_rows h (row := 1591) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_17
  simp only [← c97416, ← c97417, k2, ← c97418] at e_1591_17
  have hr := e_1591_17
  simp only [e_1591_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1591 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15717)) := by
  have c97419 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97420 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97421 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97422 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97423 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97424 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1591_18 := arithEq_of_rows h (row := 1591) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_18
  simp only [← c97419, ← c97420, ← c97421] at e_1591_18
  have e_1591_19 := arithEq_of_rows h (row := 1591) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_19
  simp only [← c97422, ← c97423, k2, ← c97424] at e_1591_19
  have hr := e_1591_19
  simp only [e_1591_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1592 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15718)) := by
  have c97425 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97426 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97427 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97428 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97429 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97430 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1592_0 := arithEq_of_rows h (row := 1592) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_0
  simp only [← c97425, ← c97426, ← c97427] at e_1592_0
  have e_1592_1 := arithEq_of_rows h (row := 1592) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_1
  simp only [← c97428, ← c97429, k2, ← c97430] at e_1592_1
  have hr := e_1592_1
  simp only [e_1592_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1592 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15719)) := by
  have c97431 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97432 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97433 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97434 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97435 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97436 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1592_2 := arithEq_of_rows h (row := 1592) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_2
  simp only [← c97431, ← c97432, ← c97433] at e_1592_2
  have e_1592_3 := arithEq_of_rows h (row := 1592) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_3
  simp only [← c97434, ← c97435, k2, ← c97436] at e_1592_3
  have hr := e_1592_3
  simp only [e_1592_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1592 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15720)) := by
  have c97437 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97438 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97439 := (publicBatchWrapper86_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97440 := (publicBatchWrapper86_copies3045 a h).1
  have c97441 := (publicBatchWrapper86_copies3045 a h).2.1
  have c97442 := (publicBatchWrapper86_copies3045 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1592_4 := arithEq_of_rows h (row := 1592) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_4
  simp only [← c97437, ← c97438, ← c97439] at e_1592_4
  have e_1592_5 := arithEq_of_rows h (row := 1592) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_5
  simp only [← c97440, ← c97441, k2, ← c97442] at e_1592_5
  have hr := e_1592_5
  simp only [e_1592_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1592 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15721)) := by
  have c97443 := (publicBatchWrapper86_copies3045 a h).2.2.2.1
  have c97444 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.1
  have c97445 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.1
  have c97446 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.1
  have c97447 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.1
  have c97448 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1592_6 := arithEq_of_rows h (row := 1592) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_6
  simp only [← c97443, ← c97444, ← c97445] at e_1592_6
  have e_1592_7 := arithEq_of_rows h (row := 1592) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_7
  simp only [← c97446, ← c97447, k2, ← c97448] at e_1592_7
  have hr := e_1592_7
  simp only [e_1592_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1592 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15722)) := by
  have c97449 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.1
  have c97450 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97451 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97452 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97453 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97454 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1592_8 := arithEq_of_rows h (row := 1592) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_8
  simp only [← c97449, ← c97450, ← c97451] at e_1592_8
  have e_1592_9 := arithEq_of_rows h (row := 1592) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_9
  simp only [← c97452, ← c97453, k2, ← c97454] at e_1592_9
  have hr := e_1592_9
  simp only [e_1592_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1592 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15723)) := by
  have c97455 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97456 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97457 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97458 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97459 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97460 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1592_10 := arithEq_of_rows h (row := 1592) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_10
  simp only [← c97455, ← c97456, ← c97457] at e_1592_10
  have e_1592_11 := arithEq_of_rows h (row := 1592) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_11
  simp only [← c97458, ← c97459, k2, ← c97460] at e_1592_11
  have hr := e_1592_11
  simp only [e_1592_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1592 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15724)) := by
  have c97461 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97462 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97463 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97464 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97465 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97466 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1592_12 := arithEq_of_rows h (row := 1592) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_12
  simp only [← c97461, ← c97462, ← c97463] at e_1592_12
  have e_1592_13 := arithEq_of_rows h (row := 1592) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_13
  simp only [← c97464, ← c97465, k2, ← c97466] at e_1592_13
  have hr := e_1592_13
  simp only [e_1592_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1592 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15725)) := by
  have c97467 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97468 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97469 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97470 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97471 := (publicBatchWrapper86_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97472 := (publicBatchWrapper86_copies3046 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1592_14 := arithEq_of_rows h (row := 1592) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_14
  simp only [← c97467, ← c97468, ← c97469] at e_1592_14
  have e_1592_15 := arithEq_of_rows h (row := 1592) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_15
  simp only [← c97470, ← c97471, k2, ← c97472] at e_1592_15
  have hr := e_1592_15
  simp only [e_1592_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1592 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15726)) := by
  have c97473 := (publicBatchWrapper86_copies3046 a h).2.1
  have c97474 := (publicBatchWrapper86_copies3046 a h).2.2.1
  have c97475 := (publicBatchWrapper86_copies3046 a h).2.2.2.1
  have c97476 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.1
  have c97477 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.1
  have c97478 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1592_16 := arithEq_of_rows h (row := 1592) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_16
  simp only [← c97473, ← c97474, ← c97475] at e_1592_16
  have e_1592_17 := arithEq_of_rows h (row := 1592) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_17
  simp only [← c97476, ← c97477, k2, ← c97478] at e_1592_17
  have hr := e_1592_17
  simp only [e_1592_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1592 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15727)) := by
  have c97479 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.1
  have c97480 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.1
  have c97481 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.1
  have c97482 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97483 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97484 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1592_18 := arithEq_of_rows h (row := 1592) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_18
  simp only [← c97479, ← c97480, ← c97481] at e_1592_18
  have e_1592_19 := arithEq_of_rows h (row := 1592) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_19
  simp only [← c97482, ← c97483, k2, ← c97484] at e_1592_19
  have hr := e_1592_19
  simp only [e_1592_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1593 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15728)) := by
  have c97485 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97486 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97487 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97488 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97489 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97490 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1593_0 := arithEq_of_rows h (row := 1593) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_0
  simp only [← c97485, ← c97486, ← c97487] at e_1593_0
  have e_1593_1 := arithEq_of_rows h (row := 1593) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_1
  simp only [← c97488, ← c97489, k2, ← c97490] at e_1593_1
  have hr := e_1593_1
  simp only [e_1593_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1593 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15729)) := by
  have c97491 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97492 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97493 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97494 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97495 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97496 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1593_2 := arithEq_of_rows h (row := 1593) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_2
  simp only [← c97491, ← c97492, ← c97493] at e_1593_2
  have e_1593_3 := arithEq_of_rows h (row := 1593) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_3
  simp only [← c97494, ← c97495, k2, ← c97496] at e_1593_3
  have hr := e_1593_3
  simp only [e_1593_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1593 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15730)) := by
  have c97497 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97498 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97499 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97500 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97501 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97502 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1593_4 := arithEq_of_rows h (row := 1593) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_4
  simp only [← c97497, ← c97498, ← c97499] at e_1593_4
  have e_1593_5 := arithEq_of_rows h (row := 1593) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_5
  simp only [← c97500, ← c97501, k2, ← c97502] at e_1593_5
  have hr := e_1593_5
  simp only [e_1593_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1593 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15731)) := by
  have c97503 := (publicBatchWrapper86_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97504 := (publicBatchWrapper86_copies3047 a h).1
  have c97505 := (publicBatchWrapper86_copies3047 a h).2.1
  have c97506 := (publicBatchWrapper86_copies3047 a h).2.2.1
  have c97507 := (publicBatchWrapper86_copies3047 a h).2.2.2.1
  have c97508 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1593_6 := arithEq_of_rows h (row := 1593) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_6
  simp only [← c97503, ← c97504, ← c97505] at e_1593_6
  have e_1593_7 := arithEq_of_rows h (row := 1593) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_7
  simp only [← c97506, ← c97507, k2, ← c97508] at e_1593_7
  have hr := e_1593_7
  simp only [e_1593_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1593 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15732)) := by
  have c97509 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.1
  have c97510 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.1
  have c97511 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.1
  have c97512 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.1
  have c97513 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.1
  have c97514 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1593_8 := arithEq_of_rows h (row := 1593) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_8
  simp only [← c97509, ← c97510, ← c97511] at e_1593_8
  have e_1593_9 := arithEq_of_rows h (row := 1593) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_9
  simp only [← c97512, ← c97513, k2, ← c97514] at e_1593_9
  have hr := e_1593_9
  simp only [e_1593_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1593 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15733)) := by
  have c97515 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97516 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97517 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97518 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97519 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97520 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1593_10 := arithEq_of_rows h (row := 1593) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_10
  simp only [← c97515, ← c97516, ← c97517] at e_1593_10
  have e_1593_11 := arithEq_of_rows h (row := 1593) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_11
  simp only [← c97518, ← c97519, k2, ← c97520] at e_1593_11
  have hr := e_1593_11
  simp only [e_1593_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1593 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15734)) := by
  have c97521 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97522 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97523 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97524 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97525 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97526 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1593_12 := arithEq_of_rows h (row := 1593) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_12
  simp only [← c97521, ← c97522, ← c97523] at e_1593_12
  have e_1593_13 := arithEq_of_rows h (row := 1593) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_13
  simp only [← c97524, ← c97525, k2, ← c97526] at e_1593_13
  have hr := e_1593_13
  simp only [e_1593_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1593 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15735)) := by
  have c97527 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97528 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97529 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97530 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97531 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97532 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1593_14 := arithEq_of_rows h (row := 1593) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_14
  simp only [← c97527, ← c97528, ← c97529] at e_1593_14
  have e_1593_15 := arithEq_of_rows h (row := 1593) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_15
  simp only [← c97530, ← c97531, k2, ← c97532] at e_1593_15
  have hr := e_1593_15
  simp only [e_1593_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1593 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15736)) := by
  have c97533 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97534 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97535 := (publicBatchWrapper86_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97536 := (publicBatchWrapper86_copies3048 a h).1
  have c97537 := (publicBatchWrapper86_copies3048 a h).2.1
  have c97538 := (publicBatchWrapper86_copies3048 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1593_16 := arithEq_of_rows h (row := 1593) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_16
  simp only [← c97533, ← c97534, ← c97535] at e_1593_16
  have e_1593_17 := arithEq_of_rows h (row := 1593) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_17
  simp only [← c97536, ← c97537, k2, ← c97538] at e_1593_17
  have hr := e_1593_17
  simp only [e_1593_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1593 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15737)) := by
  have c97539 := (publicBatchWrapper86_copies3048 a h).2.2.2.1
  have c97540 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.1
  have c97541 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.1
  have c97542 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.1
  have c97543 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.1
  have c97544 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1593_18 := arithEq_of_rows h (row := 1593) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_18
  simp only [← c97539, ← c97540, ← c97541] at e_1593_18
  have e_1593_19 := arithEq_of_rows h (row := 1593) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_19
  simp only [← c97542, ← c97543, k2, ← c97544] at e_1593_19
  have hr := e_1593_19
  simp only [e_1593_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1594 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15738)) := by
  have c97545 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.1
  have c97546 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97547 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97548 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97549 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97550 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1594_0 := arithEq_of_rows h (row := 1594) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_0
  simp only [← c97545, ← c97546, ← c97547] at e_1594_0
  have e_1594_1 := arithEq_of_rows h (row := 1594) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_1
  simp only [← c97548, ← c97549, k2, ← c97550] at e_1594_1
  have hr := e_1594_1
  simp only [e_1594_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1594 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15739)) := by
  have c97551 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97552 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97553 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97554 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97555 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97556 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1594_2 := arithEq_of_rows h (row := 1594) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_2
  simp only [← c97551, ← c97552, ← c97553] at e_1594_2
  have e_1594_3 := arithEq_of_rows h (row := 1594) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_3
  simp only [← c97554, ← c97555, k2, ← c97556] at e_1594_3
  have hr := e_1594_3
  simp only [e_1594_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1594 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15740)) := by
  have c97557 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97558 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97559 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97560 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97561 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97562 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1594_4 := arithEq_of_rows h (row := 1594) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_4
  simp only [← c97557, ← c97558, ← c97559] at e_1594_4
  have e_1594_5 := arithEq_of_rows h (row := 1594) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_5
  simp only [← c97560, ← c97561, k2, ← c97562] at e_1594_5
  have hr := e_1594_5
  simp only [e_1594_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1594 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15741)) := by
  have c97563 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97564 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97565 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97566 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97567 := (publicBatchWrapper86_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97568 := (publicBatchWrapper86_copies3049 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1594_6 := arithEq_of_rows h (row := 1594) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_6
  simp only [← c97563, ← c97564, ← c97565] at e_1594_6
  have e_1594_7 := arithEq_of_rows h (row := 1594) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_7
  simp only [← c97566, ← c97567, k2, ← c97568] at e_1594_7
  have hr := e_1594_7
  simp only [e_1594_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1594 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15742)) := by
  have c97569 := (publicBatchWrapper86_copies3049 a h).2.1
  have c97570 := (publicBatchWrapper86_copies3049 a h).2.2.1
  have c97571 := (publicBatchWrapper86_copies3049 a h).2.2.2.1
  have c97572 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.1
  have c97573 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.1
  have c97574 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1594_8 := arithEq_of_rows h (row := 1594) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_8
  simp only [← c97569, ← c97570, ← c97571] at e_1594_8
  have e_1594_9 := arithEq_of_rows h (row := 1594) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_9
  simp only [← c97572, ← c97573, k2, ← c97574] at e_1594_9
  have hr := e_1594_9
  simp only [e_1594_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1594 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15743)) := by
  have c97575 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.1
  have c97576 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.1
  have c97577 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.1
  have c97578 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97579 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97580 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1594_10 := arithEq_of_rows h (row := 1594) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_10
  simp only [← c97575, ← c97576, ← c97577] at e_1594_10
  have e_1594_11 := arithEq_of_rows h (row := 1594) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_11
  simp only [← c97578, ← c97579, k2, ← c97580] at e_1594_11
  have hr := e_1594_11
  simp only [e_1594_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1594 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15744)) := by
  have c97581 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97582 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97583 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97584 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97585 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97586 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1594_12 := arithEq_of_rows h (row := 1594) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_12
  simp only [← c97581, ← c97582, ← c97583] at e_1594_12
  have e_1594_13 := arithEq_of_rows h (row := 1594) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_13
  simp only [← c97584, ← c97585, k2, ← c97586] at e_1594_13
  have hr := e_1594_13
  simp only [e_1594_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1594 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15745)) := by
  have c97587 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97588 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97589 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97590 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97591 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97592 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1594_14 := arithEq_of_rows h (row := 1594) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_14
  simp only [← c97587, ← c97588, ← c97589] at e_1594_14
  have e_1594_15 := arithEq_of_rows h (row := 1594) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_15
  simp only [← c97590, ← c97591, k2, ← c97592] at e_1594_15
  have hr := e_1594_15
  simp only [e_1594_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1594 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15746)) := by
  have c97593 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97594 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97595 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97596 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97597 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97598 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1594_16 := arithEq_of_rows h (row := 1594) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_16
  simp only [← c97593, ← c97594, ← c97595] at e_1594_16
  have e_1594_17 := arithEq_of_rows h (row := 1594) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_17
  simp only [← c97596, ← c97597, k2, ← c97598] at e_1594_17
  have hr := e_1594_17
  simp only [e_1594_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1594 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15747)) := by
  have c97599 := (publicBatchWrapper86_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97600 := (publicBatchWrapper86_copies3050 a h).1
  have c97601 := (publicBatchWrapper86_copies3050 a h).2.1
  have c97602 := (publicBatchWrapper86_copies3050 a h).2.2.1
  have c97603 := (publicBatchWrapper86_copies3050 a h).2.2.2.1
  have c97604 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1594_18 := arithEq_of_rows h (row := 1594) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_18
  simp only [← c97599, ← c97600, ← c97601] at e_1594_18
  have e_1594_19 := arithEq_of_rows h (row := 1594) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_19
  simp only [← c97602, ← c97603, k2, ← c97604] at e_1594_19
  have hr := e_1594_19
  simp only [e_1594_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1595 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15748)) := by
  have c97605 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.1
  have c97606 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.1
  have c97607 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.1
  have c97608 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.1
  have c97609 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.1
  have c97610 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1595_0 := arithEq_of_rows h (row := 1595) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_0
  simp only [← c97605, ← c97606, ← c97607] at e_1595_0
  have e_1595_1 := arithEq_of_rows h (row := 1595) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_1
  simp only [← c97608, ← c97609, k2, ← c97610] at e_1595_1
  have hr := e_1595_1
  simp only [e_1595_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1595 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15749)) := by
  have c97611 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97612 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97613 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97614 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97615 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97616 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1595_2 := arithEq_of_rows h (row := 1595) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_2
  simp only [← c97611, ← c97612, ← c97613] at e_1595_2
  have e_1595_3 := arithEq_of_rows h (row := 1595) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_3
  simp only [← c97614, ← c97615, k2, ← c97616] at e_1595_3
  have hr := e_1595_3
  simp only [e_1595_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1595 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15750)) := by
  have c97617 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97618 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97619 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97620 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97621 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97622 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1595_4 := arithEq_of_rows h (row := 1595) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_4
  simp only [← c97617, ← c97618, ← c97619] at e_1595_4
  have e_1595_5 := arithEq_of_rows h (row := 1595) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_5
  simp only [← c97620, ← c97621, k2, ← c97622] at e_1595_5
  have hr := e_1595_5
  simp only [e_1595_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1595 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15751)) := by
  have c97623 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97624 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97625 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97626 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97627 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97628 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1595_6 := arithEq_of_rows h (row := 1595) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_6
  simp only [← c97623, ← c97624, ← c97625] at e_1595_6
  have e_1595_7 := arithEq_of_rows h (row := 1595) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_7
  simp only [← c97626, ← c97627, k2, ← c97628] at e_1595_7
  have hr := e_1595_7
  simp only [e_1595_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1595 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15752)) := by
  have c97629 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97630 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97631 := (publicBatchWrapper86_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97632 := (publicBatchWrapper86_copies3051 a h).1
  have c97633 := (publicBatchWrapper86_copies3051 a h).2.1
  have c97634 := (publicBatchWrapper86_copies3051 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1595_8 := arithEq_of_rows h (row := 1595) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_8
  simp only [← c97629, ← c97630, ← c97631] at e_1595_8
  have e_1595_9 := arithEq_of_rows h (row := 1595) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_9
  simp only [← c97632, ← c97633, k2, ← c97634] at e_1595_9
  have hr := e_1595_9
  simp only [e_1595_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1595 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15753)) := by
  have c97635 := (publicBatchWrapper86_copies3051 a h).2.2.2.1
  have c97636 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.1
  have c97637 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.1
  have c97638 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.1
  have c97639 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.1
  have c97640 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1595_10 := arithEq_of_rows h (row := 1595) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_10
  simp only [← c97635, ← c97636, ← c97637] at e_1595_10
  have e_1595_11 := arithEq_of_rows h (row := 1595) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_11
  simp only [← c97638, ← c97639, k2, ← c97640] at e_1595_11
  have hr := e_1595_11
  simp only [e_1595_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1595 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15754)) := by
  have c97641 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.1
  have c97642 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97643 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97644 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97645 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97646 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1595_12 := arithEq_of_rows h (row := 1595) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_12
  simp only [← c97641, ← c97642, ← c97643] at e_1595_12
  have e_1595_13 := arithEq_of_rows h (row := 1595) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_13
  simp only [← c97644, ← c97645, k2, ← c97646] at e_1595_13
  have hr := e_1595_13
  simp only [e_1595_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1595 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15755)) := by
  have c97647 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97648 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97649 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97650 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97651 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97652 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1595_14 := arithEq_of_rows h (row := 1595) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_14
  simp only [← c97647, ← c97648, ← c97649] at e_1595_14
  have e_1595_15 := arithEq_of_rows h (row := 1595) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_15
  simp only [← c97650, ← c97651, k2, ← c97652] at e_1595_15
  have hr := e_1595_15
  simp only [e_1595_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1595 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15756)) := by
  have c97653 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97654 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97655 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97656 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97657 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97658 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1595_16 := arithEq_of_rows h (row := 1595) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_16
  simp only [← c97653, ← c97654, ← c97655] at e_1595_16
  have e_1595_17 := arithEq_of_rows h (row := 1595) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_17
  simp only [← c97656, ← c97657, k2, ← c97658] at e_1595_17
  have hr := e_1595_17
  simp only [e_1595_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1595 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15757)) := by
  have c97659 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97660 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97661 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97662 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97663 := (publicBatchWrapper86_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97664 := (publicBatchWrapper86_copies3052 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1595_18 := arithEq_of_rows h (row := 1595) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_18
  simp only [← c97659, ← c97660, ← c97661] at e_1595_18
  have e_1595_19 := arithEq_of_rows h (row := 1595) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_19
  simp only [← c97662, ← c97663, k2, ← c97664] at e_1595_19
  have hr := e_1595_19
  simp only [e_1595_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1596 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15758)) := by
  have c97665 := (publicBatchWrapper86_copies3052 a h).2.1
  have c97666 := (publicBatchWrapper86_copies3052 a h).2.2.1
  have c97667 := (publicBatchWrapper86_copies3052 a h).2.2.2.1
  have c97668 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.1
  have c97669 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.1
  have c97670 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1596_0 := arithEq_of_rows h (row := 1596) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_0
  simp only [← c97665, ← c97666, ← c97667] at e_1596_0
  have e_1596_1 := arithEq_of_rows h (row := 1596) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_1
  simp only [← c97668, ← c97669, k2, ← c97670] at e_1596_1
  have hr := e_1596_1
  simp only [e_1596_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1596 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15759)) := by
  have c97671 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.1
  have c97672 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.1
  have c97673 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.1
  have c97674 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97675 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97676 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1596_2 := arithEq_of_rows h (row := 1596) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_2
  simp only [← c97671, ← c97672, ← c97673] at e_1596_2
  have e_1596_3 := arithEq_of_rows h (row := 1596) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_3
  simp only [← c97674, ← c97675, k2, ← c97676] at e_1596_3
  have hr := e_1596_3
  simp only [e_1596_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1596 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15760)) := by
  have c97677 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97678 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97679 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97680 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97681 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97682 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1596_4 := arithEq_of_rows h (row := 1596) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_4
  simp only [← c97677, ← c97678, ← c97679] at e_1596_4
  have e_1596_5 := arithEq_of_rows h (row := 1596) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_5
  simp only [← c97680, ← c97681, k2, ← c97682] at e_1596_5
  have hr := e_1596_5
  simp only [e_1596_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1596 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15761)) := by
  have c97683 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97684 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97685 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97686 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97687 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97688 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1596_6 := arithEq_of_rows h (row := 1596) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_6
  simp only [← c97683, ← c97684, ← c97685] at e_1596_6
  have e_1596_7 := arithEq_of_rows h (row := 1596) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_7
  simp only [← c97686, ← c97687, k2, ← c97688] at e_1596_7
  have hr := e_1596_7
  simp only [e_1596_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1596 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15762)) := by
  have c97689 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97690 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97691 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97692 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97693 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97694 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1596_8 := arithEq_of_rows h (row := 1596) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_8
  simp only [← c97689, ← c97690, ← c97691] at e_1596_8
  have e_1596_9 := arithEq_of_rows h (row := 1596) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_9
  simp only [← c97692, ← c97693, k2, ← c97694] at e_1596_9
  have hr := e_1596_9
  simp only [e_1596_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1596 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15763)) := by
  have c97695 := (publicBatchWrapper86_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97696 := (publicBatchWrapper86_copies3053 a h).1
  have c97697 := (publicBatchWrapper86_copies3053 a h).2.1
  have c97698 := (publicBatchWrapper86_copies3053 a h).2.2.1
  have c97699 := (publicBatchWrapper86_copies3053 a h).2.2.2.1
  have c97700 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1596_10 := arithEq_of_rows h (row := 1596) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_10
  simp only [← c97695, ← c97696, ← c97697] at e_1596_10
  have e_1596_11 := arithEq_of_rows h (row := 1596) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_11
  simp only [← c97698, ← c97699, k2, ← c97700] at e_1596_11
  have hr := e_1596_11
  simp only [e_1596_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1596 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15764)) := by
  have c97701 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.1
  have c97702 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.1
  have c97703 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.1
  have c97704 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.1
  have c97705 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.1
  have c97706 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1596_12 := arithEq_of_rows h (row := 1596) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_12
  simp only [← c97701, ← c97702, ← c97703] at e_1596_12
  have e_1596_13 := arithEq_of_rows h (row := 1596) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_13
  simp only [← c97704, ← c97705, k2, ← c97706] at e_1596_13
  have hr := e_1596_13
  simp only [e_1596_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1596 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15765)) := by
  have c97707 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97708 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97709 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97710 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97711 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97712 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1596_14 := arithEq_of_rows h (row := 1596) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_14
  simp only [← c97707, ← c97708, ← c97709] at e_1596_14
  have e_1596_15 := arithEq_of_rows h (row := 1596) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_15
  simp only [← c97710, ← c97711, k2, ← c97712] at e_1596_15
  have hr := e_1596_15
  simp only [e_1596_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1596 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15766)) := by
  have c97713 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97714 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97715 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97716 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97717 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97718 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1596_16 := arithEq_of_rows h (row := 1596) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_16
  simp only [← c97713, ← c97714, ← c97715] at e_1596_16
  have e_1596_17 := arithEq_of_rows h (row := 1596) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_17
  simp only [← c97716, ← c97717, k2, ← c97718] at e_1596_17
  have hr := e_1596_17
  simp only [e_1596_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1596 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15767)) := by
  have c97719 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97720 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97721 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97722 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97723 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97724 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1596_18 := arithEq_of_rows h (row := 1596) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_18
  simp only [← c97719, ← c97720, ← c97721] at e_1596_18
  have e_1596_19 := arithEq_of_rows h (row := 1596) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_19
  simp only [← c97722, ← c97723, k2, ← c97724] at e_1596_19
  have hr := e_1596_19
  simp only [e_1596_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1597 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15768)) := by
  have c97725 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97726 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97727 := (publicBatchWrapper86_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97728 := (publicBatchWrapper86_copies3054 a h).1
  have c97729 := (publicBatchWrapper86_copies3054 a h).2.1
  have c97730 := (publicBatchWrapper86_copies3054 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1597_0 := arithEq_of_rows h (row := 1597) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_0
  simp only [← c97725, ← c97726, ← c97727] at e_1597_0
  have e_1597_1 := arithEq_of_rows h (row := 1597) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_1
  simp only [← c97728, ← c97729, k2, ← c97730] at e_1597_1
  have hr := e_1597_1
  simp only [e_1597_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1597 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15769)) := by
  have c97731 := (publicBatchWrapper86_copies3054 a h).2.2.2.1
  have c97732 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.1
  have c97733 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.1
  have c97734 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.1
  have c97735 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.1
  have c97736 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1597_2 := arithEq_of_rows h (row := 1597) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_2
  simp only [← c97731, ← c97732, ← c97733] at e_1597_2
  have e_1597_3 := arithEq_of_rows h (row := 1597) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_3
  simp only [← c97734, ← c97735, k2, ← c97736] at e_1597_3
  have hr := e_1597_3
  simp only [e_1597_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1597 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15810)) := by
  have c97737 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.1
  have c97738 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97739 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97740 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97741 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97742 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1597_4 := arithEq_of_rows h (row := 1597) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_4
  simp only [← c97737, ← c97738, ← c97739] at e_1597_4
  have e_1597_5 := arithEq_of_rows h (row := 1597) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_5
  simp only [← c97740, ← c97741, k2, ← c97742] at e_1597_5
  have hr := e_1597_5
  simp only [e_1597_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1597 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15811)) := by
  have c97743 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97744 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97745 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97746 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97747 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97748 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1597_6 := arithEq_of_rows h (row := 1597) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_6
  simp only [← c97743, ← c97744, ← c97745] at e_1597_6
  have e_1597_7 := arithEq_of_rows h (row := 1597) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_7
  simp only [← c97746, ← c97747, k2, ← c97748] at e_1597_7
  have hr := e_1597_7
  simp only [e_1597_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1597 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15812)) := by
  have c97749 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97750 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97751 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97752 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97753 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97754 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1597_8 := arithEq_of_rows h (row := 1597) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_8
  simp only [← c97749, ← c97750, ← c97751] at e_1597_8
  have e_1597_9 := arithEq_of_rows h (row := 1597) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_9
  simp only [← c97752, ← c97753, k2, ← c97754] at e_1597_9
  have hr := e_1597_9
  simp only [e_1597_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1597 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15813)) := by
  have c97755 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97756 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97757 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97758 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97759 := (publicBatchWrapper86_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97760 := (publicBatchWrapper86_copies3055 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1597_10 := arithEq_of_rows h (row := 1597) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_10
  simp only [← c97755, ← c97756, ← c97757] at e_1597_10
  have e_1597_11 := arithEq_of_rows h (row := 1597) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_11
  simp only [← c97758, ← c97759, k2, ← c97760] at e_1597_11
  have hr := e_1597_11
  simp only [e_1597_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1597 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15814)) := by
  have c97761 := (publicBatchWrapper86_copies3055 a h).2.1
  have c97762 := (publicBatchWrapper86_copies3055 a h).2.2.1
  have c97763 := (publicBatchWrapper86_copies3055 a h).2.2.2.1
  have c97764 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.1
  have c97765 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.1
  have c97766 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1597_12 := arithEq_of_rows h (row := 1597) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_12
  simp only [← c97761, ← c97762, ← c97763] at e_1597_12
  have e_1597_13 := arithEq_of_rows h (row := 1597) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_13
  simp only [← c97764, ← c97765, k2, ← c97766] at e_1597_13
  have hr := e_1597_13
  simp only [e_1597_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1597 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15815)) := by
  have c97767 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.1
  have c97768 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.1
  have c97769 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.1
  have c97770 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97771 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97772 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1597_14 := arithEq_of_rows h (row := 1597) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_14
  simp only [← c97767, ← c97768, ← c97769] at e_1597_14
  have e_1597_15 := arithEq_of_rows h (row := 1597) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_15
  simp only [← c97770, ← c97771, k2, ← c97772] at e_1597_15
  have hr := e_1597_15
  simp only [e_1597_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1597 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15816)) := by
  have c97773 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97774 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97775 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97776 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97777 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97778 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1597_16 := arithEq_of_rows h (row := 1597) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_16
  simp only [← c97773, ← c97774, ← c97775] at e_1597_16
  have e_1597_17 := arithEq_of_rows h (row := 1597) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_17
  simp only [← c97776, ← c97777, k2, ← c97778] at e_1597_17
  have hr := e_1597_17
  simp only [e_1597_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1597 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15817)) := by
  have c97779 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97780 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97781 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97782 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97783 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97784 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1597_18 := arithEq_of_rows h (row := 1597) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_18
  simp only [← c97779, ← c97780, ← c97781] at e_1597_18
  have e_1597_19 := arithEq_of_rows h (row := 1597) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_19
  simp only [← c97782, ← c97783, k2, ← c97784] at e_1597_19
  have hr := e_1597_19
  simp only [e_1597_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1598 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15818)) := by
  have c97785 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97786 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97787 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97788 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97789 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97790 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1598_0 := arithEq_of_rows h (row := 1598) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_0
  simp only [← c97785, ← c97786, ← c97787] at e_1598_0
  have e_1598_1 := arithEq_of_rows h (row := 1598) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_1
  simp only [← c97788, ← c97789, k2, ← c97790] at e_1598_1
  have hr := e_1598_1
  simp only [e_1598_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1598 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15819)) := by
  have c97791 := (publicBatchWrapper86_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97792 := (publicBatchWrapper86_copies3056 a h).1
  have c97793 := (publicBatchWrapper86_copies3056 a h).2.1
  have c97794 := (publicBatchWrapper86_copies3056 a h).2.2.1
  have c97795 := (publicBatchWrapper86_copies3056 a h).2.2.2.1
  have c97796 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1598_2 := arithEq_of_rows h (row := 1598) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_2
  simp only [← c97791, ← c97792, ← c97793] at e_1598_2
  have e_1598_3 := arithEq_of_rows h (row := 1598) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_3
  simp only [← c97794, ← c97795, k2, ← c97796] at e_1598_3
  have hr := e_1598_3
  simp only [e_1598_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1598 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15820)) := by
  have c97797 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.1
  have c97798 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.1
  have c97799 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.1
  have c97800 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.1
  have c97801 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.1
  have c97802 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1598_4 := arithEq_of_rows h (row := 1598) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_4
  simp only [← c97797, ← c97798, ← c97799] at e_1598_4
  have e_1598_5 := arithEq_of_rows h (row := 1598) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_5
  simp only [← c97800, ← c97801, k2, ← c97802] at e_1598_5
  have hr := e_1598_5
  simp only [e_1598_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1598 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15821)) := by
  have c97803 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97804 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97805 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97806 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97807 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97808 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1598_6 := arithEq_of_rows h (row := 1598) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_6
  simp only [← c97803, ← c97804, ← c97805] at e_1598_6
  have e_1598_7 := arithEq_of_rows h (row := 1598) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_7
  simp only [← c97806, ← c97807, k2, ← c97808] at e_1598_7
  have hr := e_1598_7
  simp only [e_1598_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1598 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15822)) := by
  have c97809 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97810 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97811 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97812 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97813 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97814 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1598_8 := arithEq_of_rows h (row := 1598) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_8
  simp only [← c97809, ← c97810, ← c97811] at e_1598_8
  have e_1598_9 := arithEq_of_rows h (row := 1598) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_9
  simp only [← c97812, ← c97813, k2, ← c97814] at e_1598_9
  have hr := e_1598_9
  simp only [e_1598_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1598 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15823)) := by
  have c97815 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97816 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97817 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97818 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97819 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97820 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1598_10 := arithEq_of_rows h (row := 1598) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_10
  simp only [← c97815, ← c97816, ← c97817] at e_1598_10
  have e_1598_11 := arithEq_of_rows h (row := 1598) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_11
  simp only [← c97818, ← c97819, k2, ← c97820] at e_1598_11
  have hr := e_1598_11
  simp only [e_1598_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1598 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15824)) := by
  have c97821 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97822 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97823 := (publicBatchWrapper86_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97824 := (publicBatchWrapper86_copies3057 a h).1
  have c97825 := (publicBatchWrapper86_copies3057 a h).2.1
  have c97826 := (publicBatchWrapper86_copies3057 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1598_12 := arithEq_of_rows h (row := 1598) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_12
  simp only [← c97821, ← c97822, ← c97823] at e_1598_12
  have e_1598_13 := arithEq_of_rows h (row := 1598) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_13
  simp only [← c97824, ← c97825, k2, ← c97826] at e_1598_13
  have hr := e_1598_13
  simp only [e_1598_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1598 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15825)) := by
  have c97827 := (publicBatchWrapper86_copies3057 a h).2.2.2.1
  have c97828 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.1
  have c97829 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.1
  have c97830 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.1
  have c97831 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.1
  have c97832 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1598_14 := arithEq_of_rows h (row := 1598) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_14
  simp only [← c97827, ← c97828, ← c97829] at e_1598_14
  have e_1598_15 := arithEq_of_rows h (row := 1598) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_15
  simp only [← c97830, ← c97831, k2, ← c97832] at e_1598_15
  have hr := e_1598_15
  simp only [e_1598_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1598 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15826)) := by
  have c97833 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.1
  have c97834 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97835 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97836 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97837 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97838 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1598_16 := arithEq_of_rows h (row := 1598) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_16
  simp only [← c97833, ← c97834, ← c97835] at e_1598_16
  have e_1598_17 := arithEq_of_rows h (row := 1598) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_17
  simp only [← c97836, ← c97837, k2, ← c97838] at e_1598_17
  have hr := e_1598_17
  simp only [e_1598_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1598 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15827)) := by
  have c97839 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97840 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97841 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97842 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97843 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97844 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1598_18 := arithEq_of_rows h (row := 1598) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_18
  simp only [← c97839, ← c97840, ← c97841] at e_1598_18
  have e_1598_19 := arithEq_of_rows h (row := 1598) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_19
  simp only [← c97842, ← c97843, k2, ← c97844] at e_1598_19
  have hr := e_1598_19
  simp only [e_1598_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1599 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15828)) := by
  have c97845 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97846 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97847 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97848 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97849 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97850 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1599_0 := arithEq_of_rows h (row := 1599) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_0
  simp only [← c97845, ← c97846, ← c97847] at e_1599_0
  have e_1599_1 := arithEq_of_rows h (row := 1599) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_1
  simp only [← c97848, ← c97849, k2, ← c97850] at e_1599_1
  have hr := e_1599_1
  simp only [e_1599_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1599 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15829)) := by
  have c97851 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97852 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97853 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97854 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97855 := (publicBatchWrapper86_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97856 := (publicBatchWrapper86_copies3058 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1599_2 := arithEq_of_rows h (row := 1599) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_2
  simp only [← c97851, ← c97852, ← c97853] at e_1599_2
  have e_1599_3 := arithEq_of_rows h (row := 1599) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_3
  simp only [← c97854, ← c97855, k2, ← c97856] at e_1599_3
  have hr := e_1599_3
  simp only [e_1599_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1599 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15830)) := by
  have c97857 := (publicBatchWrapper86_copies3058 a h).2.1
  have c97858 := (publicBatchWrapper86_copies3058 a h).2.2.1
  have c97859 := (publicBatchWrapper86_copies3058 a h).2.2.2.1
  have c97860 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.1
  have c97861 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.1
  have c97862 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1599_4 := arithEq_of_rows h (row := 1599) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_4
  simp only [← c97857, ← c97858, ← c97859] at e_1599_4
  have e_1599_5 := arithEq_of_rows h (row := 1599) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_5
  simp only [← c97860, ← c97861, k2, ← c97862] at e_1599_5
  have hr := e_1599_5
  simp only [e_1599_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1599 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15831)) := by
  have c97863 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.1
  have c97864 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.1
  have c97865 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.1
  have c97866 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97867 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97868 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1599_6 := arithEq_of_rows h (row := 1599) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_6
  simp only [← c97863, ← c97864, ← c97865] at e_1599_6
  have e_1599_7 := arithEq_of_rows h (row := 1599) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_7
  simp only [← c97866, ← c97867, k2, ← c97868] at e_1599_7
  have hr := e_1599_7
  simp only [e_1599_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1599 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15832)) := by
  have c97869 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97870 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97871 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97872 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97873 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97874 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1599_8 := arithEq_of_rows h (row := 1599) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_8
  simp only [← c97869, ← c97870, ← c97871] at e_1599_8
  have e_1599_9 := arithEq_of_rows h (row := 1599) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_9
  simp only [← c97872, ← c97873, k2, ← c97874] at e_1599_9
  have hr := e_1599_9
  simp only [e_1599_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1599 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15833)) := by
  have c97875 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97876 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97877 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97878 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97879 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97880 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1599_10 := arithEq_of_rows h (row := 1599) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_10
  simp only [← c97875, ← c97876, ← c97877] at e_1599_10
  have e_1599_11 := arithEq_of_rows h (row := 1599) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_11
  simp only [← c97878, ← c97879, k2, ← c97880] at e_1599_11
  have hr := e_1599_11
  simp only [e_1599_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1599 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15834)) := by
  have c97881 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97882 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97883 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97884 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97885 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97886 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1599_12 := arithEq_of_rows h (row := 1599) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_12
  simp only [← c97881, ← c97882, ← c97883] at e_1599_12
  have e_1599_13 := arithEq_of_rows h (row := 1599) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_13
  simp only [← c97884, ← c97885, k2, ← c97886] at e_1599_13
  have hr := e_1599_13
  simp only [e_1599_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1599 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15835)) := by
  have c97887 := (publicBatchWrapper86_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97888 := (publicBatchWrapper86_copies3059 a h).1
  have c97889 := (publicBatchWrapper86_copies3059 a h).2.1
  have c97890 := (publicBatchWrapper86_copies3059 a h).2.2.1
  have c97891 := (publicBatchWrapper86_copies3059 a h).2.2.2.1
  have c97892 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1599_14 := arithEq_of_rows h (row := 1599) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_14
  simp only [← c97887, ← c97888, ← c97889] at e_1599_14
  have e_1599_15 := arithEq_of_rows h (row := 1599) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_15
  simp only [← c97890, ← c97891, k2, ← c97892] at e_1599_15
  have hr := e_1599_15
  simp only [e_1599_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1599 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15836)) := by
  have c97893 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.1
  have c97894 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.1
  have c97895 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.1
  have c97896 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.1
  have c97897 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.1
  have c97898 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1599_16 := arithEq_of_rows h (row := 1599) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_16
  simp only [← c97893, ← c97894, ← c97895] at e_1599_16
  have e_1599_17 := arithEq_of_rows h (row := 1599) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_17
  simp only [← c97896, ← c97897, k2, ← c97898] at e_1599_17
  have hr := e_1599_17
  simp only [e_1599_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1599 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15837)) := by
  have c97899 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97900 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97901 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97902 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97903 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97904 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1599_18 := arithEq_of_rows h (row := 1599) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_18
  simp only [← c97899, ← c97900, ← c97901] at e_1599_18
  have e_1599_19 := arithEq_of_rows h (row := 1599) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_19
  simp only [← c97902, ← c97903, k2, ← c97904] at e_1599_19
  have hr := e_1599_19
  simp only [e_1599_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1600 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15838)) := by
  have c97905 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97906 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97907 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97908 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97909 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97910 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1600_0 := arithEq_of_rows h (row := 1600) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_0
  simp only [← c97905, ← c97906, ← c97907] at e_1600_0
  have e_1600_1 := arithEq_of_rows h (row := 1600) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_1
  simp only [← c97908, ← c97909, k2, ← c97910] at e_1600_1
  have hr := e_1600_1
  simp only [e_1600_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1600 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15839)) := by
  have c97911 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97912 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97913 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97914 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97915 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97916 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1600_2 := arithEq_of_rows h (row := 1600) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_2
  simp only [← c97911, ← c97912, ← c97913] at e_1600_2
  have e_1600_3 := arithEq_of_rows h (row := 1600) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_3
  simp only [← c97914, ← c97915, k2, ← c97916] at e_1600_3
  have hr := e_1600_3
  simp only [e_1600_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1600 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15840)) := by
  have c97917 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97918 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97919 := (publicBatchWrapper86_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97920 := (publicBatchWrapper86_copies3060 a h).1
  have c97921 := (publicBatchWrapper86_copies3060 a h).2.1
  have c97922 := (publicBatchWrapper86_copies3060 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1600_4 := arithEq_of_rows h (row := 1600) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_4
  simp only [← c97917, ← c97918, ← c97919] at e_1600_4
  have e_1600_5 := arithEq_of_rows h (row := 1600) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_5
  simp only [← c97920, ← c97921, k2, ← c97922] at e_1600_5
  have hr := e_1600_5
  simp only [e_1600_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1600 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15841)) := by
  have c97923 := (publicBatchWrapper86_copies3060 a h).2.2.2.1
  have c97924 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.1
  have c97925 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.1
  have c97926 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.1
  have c97927 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.1
  have c97928 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1600_6 := arithEq_of_rows h (row := 1600) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_6
  simp only [← c97923, ← c97924, ← c97925] at e_1600_6
  have e_1600_7 := arithEq_of_rows h (row := 1600) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_7
  simp only [← c97926, ← c97927, k2, ← c97928] at e_1600_7
  have hr := e_1600_7
  simp only [e_1600_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1600 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15842)) := by
  have c97929 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.1
  have c97930 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97931 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97932 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97933 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97934 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1600_8 := arithEq_of_rows h (row := 1600) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_8
  simp only [← c97929, ← c97930, ← c97931] at e_1600_8
  have e_1600_9 := arithEq_of_rows h (row := 1600) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_9
  simp only [← c97932, ← c97933, k2, ← c97934] at e_1600_9
  have hr := e_1600_9
  simp only [e_1600_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1600 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15843)) := by
  have c97935 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97936 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97937 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97938 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97939 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97940 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1600_10 := arithEq_of_rows h (row := 1600) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_10
  simp only [← c97935, ← c97936, ← c97937] at e_1600_10
  have e_1600_11 := arithEq_of_rows h (row := 1600) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_11
  simp only [← c97938, ← c97939, k2, ← c97940] at e_1600_11
  have hr := e_1600_11
  simp only [e_1600_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1600 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15844)) := by
  have c97941 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97942 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97943 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97944 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97945 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97946 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1600_12 := arithEq_of_rows h (row := 1600) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_12
  simp only [← c97941, ← c97942, ← c97943] at e_1600_12
  have e_1600_13 := arithEq_of_rows h (row := 1600) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_13
  simp only [← c97944, ← c97945, k2, ← c97946] at e_1600_13
  have hr := e_1600_13
  simp only [e_1600_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1600 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15845)) := by
  have c97947 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97948 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97949 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97950 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97951 := (publicBatchWrapper86_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97952 := (publicBatchWrapper86_copies3061 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1600_14 := arithEq_of_rows h (row := 1600) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_14
  simp only [← c97947, ← c97948, ← c97949] at e_1600_14
  have e_1600_15 := arithEq_of_rows h (row := 1600) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_15
  simp only [← c97950, ← c97951, k2, ← c97952] at e_1600_15
  have hr := e_1600_15
  simp only [e_1600_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1600 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15846)) := by
  have c97953 := (publicBatchWrapper86_copies3061 a h).2.1
  have c97954 := (publicBatchWrapper86_copies3061 a h).2.2.1
  have c97955 := (publicBatchWrapper86_copies3061 a h).2.2.2.1
  have c97956 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.1
  have c97957 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.1
  have c97958 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1600_16 := arithEq_of_rows h (row := 1600) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_16
  simp only [← c97953, ← c97954, ← c97955] at e_1600_16
  have e_1600_17 := arithEq_of_rows h (row := 1600) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_17
  simp only [← c97956, ← c97957, k2, ← c97958] at e_1600_17
  have hr := e_1600_17
  simp only [e_1600_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1600 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15847)) := by
  have c97959 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.1
  have c97960 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.1
  have c97961 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.1
  have c97962 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97963 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97964 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1600_18 := arithEq_of_rows h (row := 1600) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_18
  simp only [← c97959, ← c97960, ← c97961] at e_1600_18
  have e_1600_19 := arithEq_of_rows h (row := 1600) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_19
  simp only [← c97962, ← c97963, k2, ← c97964] at e_1600_19
  have hr := e_1600_19
  simp only [e_1600_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1601 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15848)) := by
  have c97965 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97966 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97967 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97968 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97969 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97970 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1601_0 := arithEq_of_rows h (row := 1601) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_0
  simp only [← c97965, ← c97966, ← c97967] at e_1601_0
  have e_1601_1 := arithEq_of_rows h (row := 1601) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_1
  simp only [← c97968, ← c97969, k2, ← c97970] at e_1601_1
  have hr := e_1601_1
  simp only [e_1601_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1601 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15849)) := by
  have c97971 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97972 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97973 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97974 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97975 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97976 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1601_2 := arithEq_of_rows h (row := 1601) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_2
  simp only [← c97971, ← c97972, ← c97973] at e_1601_2
  have e_1601_3 := arithEq_of_rows h (row := 1601) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_3
  simp only [← c97974, ← c97975, k2, ← c97976] at e_1601_3
  have hr := e_1601_3
  simp only [e_1601_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1601 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15850)) := by
  have c97977 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97978 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97979 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97980 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97981 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97982 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1601_4 := arithEq_of_rows h (row := 1601) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_4
  simp only [← c97977, ← c97978, ← c97979] at e_1601_4
  have e_1601_5 := arithEq_of_rows h (row := 1601) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_5
  simp only [← c97980, ← c97981, k2, ← c97982] at e_1601_5
  have hr := e_1601_5
  simp only [e_1601_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1601 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15851)) := by
  have c97983 := (publicBatchWrapper86_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97984 := (publicBatchWrapper86_copies3062 a h).1
  have c97985 := (publicBatchWrapper86_copies3062 a h).2.1
  have c97986 := (publicBatchWrapper86_copies3062 a h).2.2.1
  have c97987 := (publicBatchWrapper86_copies3062 a h).2.2.2.1
  have c97988 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1601_6 := arithEq_of_rows h (row := 1601) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_6
  simp only [← c97983, ← c97984, ← c97985] at e_1601_6
  have e_1601_7 := arithEq_of_rows h (row := 1601) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_7
  simp only [← c97986, ← c97987, k2, ← c97988] at e_1601_7
  have hr := e_1601_7
  simp only [e_1601_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1601 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15852)) := by
  have c97989 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.1
  have c97990 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.1
  have c97991 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.1
  have c97992 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.1
  have c97993 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.1
  have c97994 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1601_8 := arithEq_of_rows h (row := 1601) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_8
  simp only [← c97989, ← c97990, ← c97991] at e_1601_8
  have e_1601_9 := arithEq_of_rows h (row := 1601) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_9
  simp only [← c97992, ← c97993, k2, ← c97994] at e_1601_9
  have hr := e_1601_9
  simp only [e_1601_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1601 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15853)) := by
  have c97995 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97996 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97997 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97998 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97999 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98000 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1601_10 := arithEq_of_rows h (row := 1601) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_10
  simp only [← c97995, ← c97996, ← c97997] at e_1601_10
  have e_1601_11 := arithEq_of_rows h (row := 1601) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_11
  simp only [← c97998, ← c97999, k2, ← c98000] at e_1601_11
  have hr := e_1601_11
  simp only [e_1601_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1601 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15854)) := by
  have c98001 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98002 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98003 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98004 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98005 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98006 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1601_12 := arithEq_of_rows h (row := 1601) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_12
  simp only [← c98001, ← c98002, ← c98003] at e_1601_12
  have e_1601_13 := arithEq_of_rows h (row := 1601) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_13
  simp only [← c98004, ← c98005, k2, ← c98006] at e_1601_13
  have hr := e_1601_13
  simp only [e_1601_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1601 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15855)) := by
  have c98007 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98008 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98009 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98010 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98011 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98012 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1601_14 := arithEq_of_rows h (row := 1601) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_14
  simp only [← c98007, ← c98008, ← c98009] at e_1601_14
  have e_1601_15 := arithEq_of_rows h (row := 1601) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_15
  simp only [← c98010, ← c98011, k2, ← c98012] at e_1601_15
  have hr := e_1601_15
  simp only [e_1601_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1601 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15856)) := by
  have c98013 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98014 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98015 := (publicBatchWrapper86_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98016 := (publicBatchWrapper86_copies3063 a h).1
  have c98017 := (publicBatchWrapper86_copies3063 a h).2.1
  have c98018 := (publicBatchWrapper86_copies3063 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1601_16 := arithEq_of_rows h (row := 1601) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_16
  simp only [← c98013, ← c98014, ← c98015] at e_1601_16
  have e_1601_17 := arithEq_of_rows h (row := 1601) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_17
  simp only [← c98016, ← c98017, k2, ← c98018] at e_1601_17
  have hr := e_1601_17
  simp only [e_1601_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1601 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15857)) := by
  have c98019 := (publicBatchWrapper86_copies3063 a h).2.2.2.1
  have c98020 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.1
  have c98021 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.1
  have c98022 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.1
  have c98023 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.1
  have c98024 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1601_18 := arithEq_of_rows h (row := 1601) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_18
  simp only [← c98019, ← c98020, ← c98021] at e_1601_18
  have e_1601_19 := arithEq_of_rows h (row := 1601) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_19
  simp only [← c98022, ← c98023, k2, ← c98024] at e_1601_19
  have hr := e_1601_19
  simp only [e_1601_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1602 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15858)) := by
  have c98025 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.1
  have c98026 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98027 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98028 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98029 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98030 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1602_0 := arithEq_of_rows h (row := 1602) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_0
  simp only [← c98025, ← c98026, ← c98027] at e_1602_0
  have e_1602_1 := arithEq_of_rows h (row := 1602) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_1
  simp only [← c98028, ← c98029, k2, ← c98030] at e_1602_1
  have hr := e_1602_1
  simp only [e_1602_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1602 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15859)) := by
  have c98031 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98032 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98033 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98034 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98035 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98036 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1602_2 := arithEq_of_rows h (row := 1602) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_2
  simp only [← c98031, ← c98032, ← c98033] at e_1602_2
  have e_1602_3 := arithEq_of_rows h (row := 1602) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_3
  simp only [← c98034, ← c98035, k2, ← c98036] at e_1602_3
  have hr := e_1602_3
  simp only [e_1602_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1602 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15860)) := by
  have c98037 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98038 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98039 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98040 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98041 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98042 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1602_4 := arithEq_of_rows h (row := 1602) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_4
  simp only [← c98037, ← c98038, ← c98039] at e_1602_4
  have e_1602_5 := arithEq_of_rows h (row := 1602) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_5
  simp only [← c98040, ← c98041, k2, ← c98042] at e_1602_5
  have hr := e_1602_5
  simp only [e_1602_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1602 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15861)) := by
  have c98043 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98044 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98045 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98046 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98047 := (publicBatchWrapper86_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98048 := (publicBatchWrapper86_copies3064 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1602_6 := arithEq_of_rows h (row := 1602) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_6
  simp only [← c98043, ← c98044, ← c98045] at e_1602_6
  have e_1602_7 := arithEq_of_rows h (row := 1602) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_7
  simp only [← c98046, ← c98047, k2, ← c98048] at e_1602_7
  have hr := e_1602_7
  simp only [e_1602_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1602 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15862)) := by
  have c98049 := (publicBatchWrapper86_copies3064 a h).2.1
  have c98050 := (publicBatchWrapper86_copies3064 a h).2.2.1
  have c98051 := (publicBatchWrapper86_copies3064 a h).2.2.2.1
  have c98052 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.1
  have c98053 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.1
  have c98054 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1602_8 := arithEq_of_rows h (row := 1602) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_8
  simp only [← c98049, ← c98050, ← c98051] at e_1602_8
  have e_1602_9 := arithEq_of_rows h (row := 1602) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_9
  simp only [← c98052, ← c98053, k2, ← c98054] at e_1602_9
  have hr := e_1602_9
  simp only [e_1602_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1602 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15863)) := by
  have c98055 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.1
  have c98056 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.1
  have c98057 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.1
  have c98058 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98059 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98060 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1602_10 := arithEq_of_rows h (row := 1602) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_10
  simp only [← c98055, ← c98056, ← c98057] at e_1602_10
  have e_1602_11 := arithEq_of_rows h (row := 1602) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_11
  simp only [← c98058, ← c98059, k2, ← c98060] at e_1602_11
  have hr := e_1602_11
  simp only [e_1602_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1602 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15864)) := by
  have c98061 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98062 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98063 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98064 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98065 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98066 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1602_12 := arithEq_of_rows h (row := 1602) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_12
  simp only [← c98061, ← c98062, ← c98063] at e_1602_12
  have e_1602_13 := arithEq_of_rows h (row := 1602) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_13
  simp only [← c98064, ← c98065, k2, ← c98066] at e_1602_13
  have hr := e_1602_13
  simp only [e_1602_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1602 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15865)) := by
  have c98067 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98068 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98069 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98070 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98071 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98072 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1602_14 := arithEq_of_rows h (row := 1602) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_14
  simp only [← c98067, ← c98068, ← c98069] at e_1602_14
  have e_1602_15 := arithEq_of_rows h (row := 1602) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_15
  simp only [← c98070, ← c98071, k2, ← c98072] at e_1602_15
  have hr := e_1602_15
  simp only [e_1602_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1602 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15866)) := by
  have c98073 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98074 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98075 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98076 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98077 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98078 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1602_16 := arithEq_of_rows h (row := 1602) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_16
  simp only [← c98073, ← c98074, ← c98075] at e_1602_16
  have e_1602_17 := arithEq_of_rows h (row := 1602) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_17
  simp only [← c98076, ← c98077, k2, ← c98078] at e_1602_17
  have hr := e_1602_17
  simp only [e_1602_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1602 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15867)) := by
  have c98079 := (publicBatchWrapper86_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98080 := (publicBatchWrapper86_copies3065 a h).1
  have c98081 := (publicBatchWrapper86_copies3065 a h).2.1
  have c98082 := (publicBatchWrapper86_copies3065 a h).2.2.1
  have c98083 := (publicBatchWrapper86_copies3065 a h).2.2.2.1
  have c98084 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1602_18 := arithEq_of_rows h (row := 1602) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_18
  simp only [← c98079, ← c98080, ← c98081] at e_1602_18
  have e_1602_19 := arithEq_of_rows h (row := 1602) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_19
  simp only [← c98082, ← c98083, k2, ← c98084] at e_1602_19
  have hr := e_1602_19
  simp only [e_1602_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1603 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15868)) := by
  have c98085 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.1
  have c98086 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.1
  have c98087 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.1
  have c98088 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.1
  have c98089 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.1
  have c98090 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1603_0 := arithEq_of_rows h (row := 1603) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_0
  simp only [← c98085, ← c98086, ← c98087] at e_1603_0
  have e_1603_1 := arithEq_of_rows h (row := 1603) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_1
  simp only [← c98088, ← c98089, k2, ← c98090] at e_1603_1
  have hr := e_1603_1
  simp only [e_1603_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1603 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15869)) := by
  have c98091 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98092 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98093 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98094 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98095 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98096 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1603_2 := arithEq_of_rows h (row := 1603) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_2
  simp only [← c98091, ← c98092, ← c98093] at e_1603_2
  have e_1603_3 := arithEq_of_rows h (row := 1603) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_3
  simp only [← c98094, ← c98095, k2, ← c98096] at e_1603_3
  have hr := e_1603_3
  simp only [e_1603_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1603 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15870)) := by
  have c98097 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98098 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98099 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98100 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98101 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98102 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1603_4 := arithEq_of_rows h (row := 1603) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_4
  simp only [← c98097, ← c98098, ← c98099] at e_1603_4
  have e_1603_5 := arithEq_of_rows h (row := 1603) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_5
  simp only [← c98100, ← c98101, k2, ← c98102] at e_1603_5
  have hr := e_1603_5
  simp only [e_1603_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1603 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15871)) := by
  have c98103 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98104 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98105 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98106 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98107 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98108 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1603_6 := arithEq_of_rows h (row := 1603) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_6
  simp only [← c98103, ← c98104, ← c98105] at e_1603_6
  have e_1603_7 := arithEq_of_rows h (row := 1603) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_7
  simp only [← c98106, ← c98107, k2, ← c98108] at e_1603_7
  have hr := e_1603_7
  simp only [e_1603_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1603 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15872)) := by
  have c98109 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98110 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98111 := (publicBatchWrapper86_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98112 := (publicBatchWrapper86_copies3066 a h).1
  have c98113 := (publicBatchWrapper86_copies3066 a h).2.1
  have c98114 := (publicBatchWrapper86_copies3066 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1603_8 := arithEq_of_rows h (row := 1603) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_8
  simp only [← c98109, ← c98110, ← c98111] at e_1603_8
  have e_1603_9 := arithEq_of_rows h (row := 1603) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_9
  simp only [← c98112, ← c98113, k2, ← c98114] at e_1603_9
  have hr := e_1603_9
  simp only [e_1603_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1603 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15873)) := by
  have c98115 := (publicBatchWrapper86_copies3066 a h).2.2.2.1
  have c98116 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.1
  have c98117 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.1
  have c98118 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.1
  have c98119 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.1
  have c98120 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1603_10 := arithEq_of_rows h (row := 1603) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_10
  simp only [← c98115, ← c98116, ← c98117] at e_1603_10
  have e_1603_11 := arithEq_of_rows h (row := 1603) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_11
  simp only [← c98118, ← c98119, k2, ← c98120] at e_1603_11
  have hr := e_1603_11
  simp only [e_1603_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1603 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15874)) := by
  have c98121 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.1
  have c98122 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98123 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98124 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98125 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98126 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1603_12 := arithEq_of_rows h (row := 1603) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_12
  simp only [← c98121, ← c98122, ← c98123] at e_1603_12
  have e_1603_13 := arithEq_of_rows h (row := 1603) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_13
  simp only [← c98124, ← c98125, k2, ← c98126] at e_1603_13
  have hr := e_1603_13
  simp only [e_1603_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1603 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15875)) := by
  have c98127 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98128 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98129 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98130 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98131 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98132 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1603_14 := arithEq_of_rows h (row := 1603) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_14
  simp only [← c98127, ← c98128, ← c98129] at e_1603_14
  have e_1603_15 := arithEq_of_rows h (row := 1603) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_15
  simp only [← c98130, ← c98131, k2, ← c98132] at e_1603_15
  have hr := e_1603_15
  simp only [e_1603_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1603 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15876)) := by
  have c98133 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98134 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98135 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98136 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98137 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98138 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1603_16 := arithEq_of_rows h (row := 1603) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_16
  simp only [← c98133, ← c98134, ← c98135] at e_1603_16
  have e_1603_17 := arithEq_of_rows h (row := 1603) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_17
  simp only [← c98136, ← c98137, k2, ← c98138] at e_1603_17
  have hr := e_1603_17
  simp only [e_1603_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1603 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15877)) := by
  have c98139 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98140 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98141 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98142 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98143 := (publicBatchWrapper86_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98144 := (publicBatchWrapper86_copies3067 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1603_18 := arithEq_of_rows h (row := 1603) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_18
  simp only [← c98139, ← c98140, ← c98141] at e_1603_18
  have e_1603_19 := arithEq_of_rows h (row := 1603) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_19
  simp only [← c98142, ← c98143, k2, ← c98144] at e_1603_19
  have hr := e_1603_19
  simp only [e_1603_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1604 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15878)) := by
  have c98145 := (publicBatchWrapper86_copies3067 a h).2.1
  have c98146 := (publicBatchWrapper86_copies3067 a h).2.2.1
  have c98147 := (publicBatchWrapper86_copies3067 a h).2.2.2.1
  have c98148 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.1
  have c98149 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.1
  have c98150 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1604_0 := arithEq_of_rows h (row := 1604) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_0
  simp only [← c98145, ← c98146, ← c98147] at e_1604_0
  have e_1604_1 := arithEq_of_rows h (row := 1604) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_1
  simp only [← c98148, ← c98149, k2, ← c98150] at e_1604_1
  have hr := e_1604_1
  simp only [e_1604_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1604 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15879)) := by
  have c98151 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.1
  have c98152 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.1
  have c98153 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.1
  have c98154 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98155 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98156 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1604_2 := arithEq_of_rows h (row := 1604) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_2
  simp only [← c98151, ← c98152, ← c98153] at e_1604_2
  have e_1604_3 := arithEq_of_rows h (row := 1604) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_3
  simp only [← c98154, ← c98155, k2, ← c98156] at e_1604_3
  have hr := e_1604_3
  simp only [e_1604_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1604 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15880)) := by
  have c98157 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98158 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98159 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98160 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98161 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98162 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1604_4 := arithEq_of_rows h (row := 1604) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_4
  simp only [← c98157, ← c98158, ← c98159] at e_1604_4
  have e_1604_5 := arithEq_of_rows h (row := 1604) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_5
  simp only [← c98160, ← c98161, k2, ← c98162] at e_1604_5
  have hr := e_1604_5
  simp only [e_1604_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1604 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15881)) := by
  have c98163 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98164 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98165 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98166 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98167 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98168 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1604_6 := arithEq_of_rows h (row := 1604) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_6
  simp only [← c98163, ← c98164, ← c98165] at e_1604_6
  have e_1604_7 := arithEq_of_rows h (row := 1604) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_7
  simp only [← c98166, ← c98167, k2, ← c98168] at e_1604_7
  have hr := e_1604_7
  simp only [e_1604_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1604 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15882)) := by
  have c98169 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98170 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98171 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98172 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98173 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98174 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1604_8 := arithEq_of_rows h (row := 1604) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_8
  simp only [← c98169, ← c98170, ← c98171] at e_1604_8
  have e_1604_9 := arithEq_of_rows h (row := 1604) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_9
  simp only [← c98172, ← c98173, k2, ← c98174] at e_1604_9
  have hr := e_1604_9
  simp only [e_1604_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1604 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15883)) := by
  have c98175 := (publicBatchWrapper86_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98176 := (publicBatchWrapper86_copies3068 a h).1
  have c98177 := (publicBatchWrapper86_copies3068 a h).2.1
  have c98178 := (publicBatchWrapper86_copies3068 a h).2.2.1
  have c98179 := (publicBatchWrapper86_copies3068 a h).2.2.2.1
  have c98180 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1604_10 := arithEq_of_rows h (row := 1604) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_10
  simp only [← c98175, ← c98176, ← c98177] at e_1604_10
  have e_1604_11 := arithEq_of_rows h (row := 1604) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_11
  simp only [← c98178, ← c98179, k2, ← c98180] at e_1604_11
  have hr := e_1604_11
  simp only [e_1604_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1604 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15884)) := by
  have c98181 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.1
  have c98182 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.1
  have c98183 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.1
  have c98184 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.1
  have c98185 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.1
  have c98186 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1604_12 := arithEq_of_rows h (row := 1604) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_12
  simp only [← c98181, ← c98182, ← c98183] at e_1604_12
  have e_1604_13 := arithEq_of_rows h (row := 1604) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_13
  simp only [← c98184, ← c98185, k2, ← c98186] at e_1604_13
  have hr := e_1604_13
  simp only [e_1604_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1604 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15885)) := by
  have c98187 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98188 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98189 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98190 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98191 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98192 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1604_14 := arithEq_of_rows h (row := 1604) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_14
  simp only [← c98187, ← c98188, ← c98189] at e_1604_14
  have e_1604_15 := arithEq_of_rows h (row := 1604) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_15
  simp only [← c98190, ← c98191, k2, ← c98192] at e_1604_15
  have hr := e_1604_15
  simp only [e_1604_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1604 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15886)) := by
  have c98193 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98194 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98195 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98196 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98197 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98198 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1604_16 := arithEq_of_rows h (row := 1604) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_16
  simp only [← c98193, ← c98194, ← c98195] at e_1604_16
  have e_1604_17 := arithEq_of_rows h (row := 1604) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_17
  simp only [← c98196, ← c98197, k2, ← c98198] at e_1604_17
  have hr := e_1604_17
  simp only [e_1604_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1604 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15887)) := by
  have c98199 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98200 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98201 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98202 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98203 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98204 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1604_18 := arithEq_of_rows h (row := 1604) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_18
  simp only [← c98199, ← c98200, ← c98201] at e_1604_18
  have e_1604_19 := arithEq_of_rows h (row := 1604) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_19
  simp only [← c98202, ← c98203, k2, ← c98204] at e_1604_19
  have hr := e_1604_19
  simp only [e_1604_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1605 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15888)) := by
  have c98205 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98206 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98207 := (publicBatchWrapper86_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98208 := (publicBatchWrapper86_copies3069 a h).1
  have c98209 := (publicBatchWrapper86_copies3069 a h).2.1
  have c98210 := (publicBatchWrapper86_copies3069 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1605_0 := arithEq_of_rows h (row := 1605) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_0
  simp only [← c98205, ← c98206, ← c98207] at e_1605_0
  have e_1605_1 := arithEq_of_rows h (row := 1605) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_1
  simp only [← c98208, ← c98209, k2, ← c98210] at e_1605_1
  have hr := e_1605_1
  simp only [e_1605_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1605 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15889)) := by
  have c98211 := (publicBatchWrapper86_copies3069 a h).2.2.2.1
  have c98212 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.1
  have c98213 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.1
  have c98214 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.1
  have c98215 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.1
  have c98216 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1605_2 := arithEq_of_rows h (row := 1605) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_2
  simp only [← c98211, ← c98212, ← c98213] at e_1605_2
  have e_1605_3 := arithEq_of_rows h (row := 1605) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_3
  simp only [← c98214, ← c98215, k2, ← c98216] at e_1605_3
  have hr := e_1605_3
  simp only [e_1605_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1605 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15890)) := by
  have c98217 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.1
  have c98218 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98219 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98220 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98221 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98222 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1605_4 := arithEq_of_rows h (row := 1605) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_4
  simp only [← c98217, ← c98218, ← c98219] at e_1605_4
  have e_1605_5 := arithEq_of_rows h (row := 1605) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_5
  simp only [← c98220, ← c98221, k2, ← c98222] at e_1605_5
  have hr := e_1605_5
  simp only [e_1605_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1605 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15891)) := by
  have c98223 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98224 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98225 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98226 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98227 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98228 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1605_6 := arithEq_of_rows h (row := 1605) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_6
  simp only [← c98223, ← c98224, ← c98225] at e_1605_6
  have e_1605_7 := arithEq_of_rows h (row := 1605) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_7
  simp only [← c98226, ← c98227, k2, ← c98228] at e_1605_7
  have hr := e_1605_7
  simp only [e_1605_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1605 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15892)) := by
  have c98229 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98230 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98231 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98232 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98233 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98234 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1605_8 := arithEq_of_rows h (row := 1605) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_8
  simp only [← c98229, ← c98230, ← c98231] at e_1605_8
  have e_1605_9 := arithEq_of_rows h (row := 1605) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_9
  simp only [← c98232, ← c98233, k2, ← c98234] at e_1605_9
  have hr := e_1605_9
  simp only [e_1605_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1605 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15893)) := by
  have c98235 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98236 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98237 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98238 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98239 := (publicBatchWrapper86_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98240 := (publicBatchWrapper86_copies3070 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1605_10 := arithEq_of_rows h (row := 1605) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_10
  simp only [← c98235, ← c98236, ← c98237] at e_1605_10
  have e_1605_11 := arithEq_of_rows h (row := 1605) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_11
  simp only [← c98238, ← c98239, k2, ← c98240] at e_1605_11
  have hr := e_1605_11
  simp only [e_1605_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1605 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15894)) := by
  have c98241 := (publicBatchWrapper86_copies3070 a h).2.1
  have c98242 := (publicBatchWrapper86_copies3070 a h).2.2.1
  have c98243 := (publicBatchWrapper86_copies3070 a h).2.2.2.1
  have c98244 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.1
  have c98245 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.1
  have c98246 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1605_12 := arithEq_of_rows h (row := 1605) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_12
  simp only [← c98241, ← c98242, ← c98243] at e_1605_12
  have e_1605_13 := arithEq_of_rows h (row := 1605) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_13
  simp only [← c98244, ← c98245, k2, ← c98246] at e_1605_13
  have hr := e_1605_13
  simp only [e_1605_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1605 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15895)) := by
  have c98247 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.1
  have c98248 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.1
  have c98249 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.1
  have c98250 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98251 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98252 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1605_14 := arithEq_of_rows h (row := 1605) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_14
  simp only [← c98247, ← c98248, ← c98249] at e_1605_14
  have e_1605_15 := arithEq_of_rows h (row := 1605) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_15
  simp only [← c98250, ← c98251, k2, ← c98252] at e_1605_15
  have hr := e_1605_15
  simp only [e_1605_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1605 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15896)) := by
  have c98253 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98254 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98255 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98256 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98257 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98258 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1605_16 := arithEq_of_rows h (row := 1605) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_16
  simp only [← c98253, ← c98254, ← c98255] at e_1605_16
  have e_1605_17 := arithEq_of_rows h (row := 1605) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_17
  simp only [← c98256, ← c98257, k2, ← c98258] at e_1605_17
  have hr := e_1605_17
  simp only [e_1605_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1605 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15897)) := by
  have c98259 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98260 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98261 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98262 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98263 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98264 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1605_18 := arithEq_of_rows h (row := 1605) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_18
  simp only [← c98259, ← c98260, ← c98261] at e_1605_18
  have e_1605_19 := arithEq_of_rows h (row := 1605) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_19
  simp only [← c98262, ← c98263, k2, ← c98264] at e_1605_19
  have hr := e_1605_19
  simp only [e_1605_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1606 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15898)) := by
  have c98265 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98266 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98267 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98268 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98269 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98270 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1606_0 := arithEq_of_rows h (row := 1606) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_0
  simp only [← c98265, ← c98266, ← c98267] at e_1606_0
  have e_1606_1 := arithEq_of_rows h (row := 1606) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_1
  simp only [← c98268, ← c98269, k2, ← c98270] at e_1606_1
  have hr := e_1606_1
  simp only [e_1606_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1606 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15899)) := by
  have c98271 := (publicBatchWrapper86_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98272 := (publicBatchWrapper86_copies3071 a h).1
  have c98273 := (publicBatchWrapper86_copies3071 a h).2.1
  have c98274 := (publicBatchWrapper86_copies3071 a h).2.2.1
  have c98275 := (publicBatchWrapper86_copies3071 a h).2.2.2.1
  have c98276 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1606_2 := arithEq_of_rows h (row := 1606) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_2
  simp only [← c98271, ← c98272, ← c98273] at e_1606_2
  have e_1606_3 := arithEq_of_rows h (row := 1606) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_3
  simp only [← c98274, ← c98275, k2, ← c98276] at e_1606_3
  have hr := e_1606_3
  simp only [e_1606_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1606 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15900)) := by
  have c98277 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.1
  have c98278 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.1
  have c98279 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.1
  have c98280 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.1
  have c98281 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.1
  have c98282 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1606_4 := arithEq_of_rows h (row := 1606) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_4
  simp only [← c98277, ← c98278, ← c98279] at e_1606_4
  have e_1606_5 := arithEq_of_rows h (row := 1606) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_5
  simp only [← c98280, ← c98281, k2, ← c98282] at e_1606_5
  have hr := e_1606_5
  simp only [e_1606_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1606 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15901)) := by
  have c98283 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98284 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98285 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98286 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98287 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98288 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1606_6 := arithEq_of_rows h (row := 1606) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_6
  simp only [← c98283, ← c98284, ← c98285] at e_1606_6
  have e_1606_7 := arithEq_of_rows h (row := 1606) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_7
  simp only [← c98286, ← c98287, k2, ← c98288] at e_1606_7
  have hr := e_1606_7
  simp only [e_1606_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1606 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15902)) := by
  have c98289 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98290 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98291 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98292 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98293 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98294 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1606_8 := arithEq_of_rows h (row := 1606) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_8
  simp only [← c98289, ← c98290, ← c98291] at e_1606_8
  have e_1606_9 := arithEq_of_rows h (row := 1606) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_9
  simp only [← c98292, ← c98293, k2, ← c98294] at e_1606_9
  have hr := e_1606_9
  simp only [e_1606_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1606 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15903)) := by
  have c98295 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98296 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98297 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98298 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98299 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98300 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1606_10 := arithEq_of_rows h (row := 1606) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_10
  simp only [← c98295, ← c98296, ← c98297] at e_1606_10
  have e_1606_11 := arithEq_of_rows h (row := 1606) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_11
  simp only [← c98298, ← c98299, k2, ← c98300] at e_1606_11
  have hr := e_1606_11
  simp only [e_1606_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1606 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15904)) := by
  have c98301 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98302 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98303 := (publicBatchWrapper86_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98304 := (publicBatchWrapper86_copies3072 a h).1
  have c98305 := (publicBatchWrapper86_copies3072 a h).2.1
  have c98306 := (publicBatchWrapper86_copies3072 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1606_12 := arithEq_of_rows h (row := 1606) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_12
  simp only [← c98301, ← c98302, ← c98303] at e_1606_12
  have e_1606_13 := arithEq_of_rows h (row := 1606) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_13
  simp only [← c98304, ← c98305, k2, ← c98306] at e_1606_13
  have hr := e_1606_13
  simp only [e_1606_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1606 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15905)) := by
  have c98307 := (publicBatchWrapper86_copies3072 a h).2.2.2.1
  have c98308 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.1
  have c98309 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.1
  have c98310 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.1
  have c98311 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.1
  have c98312 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1606_14 := arithEq_of_rows h (row := 1606) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_14
  simp only [← c98307, ← c98308, ← c98309] at e_1606_14
  have e_1606_15 := arithEq_of_rows h (row := 1606) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_15
  simp only [← c98310, ← c98311, k2, ← c98312] at e_1606_15
  have hr := e_1606_15
  simp only [e_1606_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1606 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15906)) := by
  have c98313 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.1
  have c98314 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98315 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98316 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98317 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98318 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1606_16 := arithEq_of_rows h (row := 1606) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_16
  simp only [← c98313, ← c98314, ← c98315] at e_1606_16
  have e_1606_17 := arithEq_of_rows h (row := 1606) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_17
  simp only [← c98316, ← c98317, k2, ← c98318] at e_1606_17
  have hr := e_1606_17
  simp only [e_1606_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1606 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15907)) := by
  have c98319 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98320 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98321 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98322 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98323 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98324 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1606_18 := arithEq_of_rows h (row := 1606) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_18
  simp only [← c98319, ← c98320, ← c98321] at e_1606_18
  have e_1606_19 := arithEq_of_rows h (row := 1606) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_19
  simp only [← c98322, ← c98323, k2, ← c98324] at e_1606_19
  have hr := e_1606_19
  simp only [e_1606_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1607 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15908)) := by
  have c98325 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98326 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98327 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98328 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98329 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98330 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1607_0 := arithEq_of_rows h (row := 1607) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_0
  simp only [← c98325, ← c98326, ← c98327] at e_1607_0
  have e_1607_1 := arithEq_of_rows h (row := 1607) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_1
  simp only [← c98328, ← c98329, k2, ← c98330] at e_1607_1
  have hr := e_1607_1
  simp only [e_1607_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1607 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15909)) := by
  have c98331 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98332 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98333 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98334 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98335 := (publicBatchWrapper86_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98336 := (publicBatchWrapper86_copies3073 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1607_2 := arithEq_of_rows h (row := 1607) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_2
  simp only [← c98331, ← c98332, ← c98333] at e_1607_2
  have e_1607_3 := arithEq_of_rows h (row := 1607) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_3
  simp only [← c98334, ← c98335, k2, ← c98336] at e_1607_3
  have hr := e_1607_3
  simp only [e_1607_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1607 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15910)) := by
  have c98337 := (publicBatchWrapper86_copies3073 a h).2.1
  have c98338 := (publicBatchWrapper86_copies3073 a h).2.2.1
  have c98339 := (publicBatchWrapper86_copies3073 a h).2.2.2.1
  have c98340 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.1
  have c98341 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.1
  have c98342 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1607_4 := arithEq_of_rows h (row := 1607) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_4
  simp only [← c98337, ← c98338, ← c98339] at e_1607_4
  have e_1607_5 := arithEq_of_rows h (row := 1607) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_5
  simp only [← c98340, ← c98341, k2, ← c98342] at e_1607_5
  have hr := e_1607_5
  simp only [e_1607_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1607 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15911)) := by
  have c98343 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.1
  have c98344 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.1
  have c98345 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.1
  have c98346 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98347 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98348 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1607_6 := arithEq_of_rows h (row := 1607) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_6
  simp only [← c98343, ← c98344, ← c98345] at e_1607_6
  have e_1607_7 := arithEq_of_rows h (row := 1607) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_7
  simp only [← c98346, ← c98347, k2, ← c98348] at e_1607_7
  have hr := e_1607_7
  simp only [e_1607_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1607 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15912)) := by
  have c98349 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98350 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98351 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98352 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98353 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98354 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1607_8 := arithEq_of_rows h (row := 1607) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_8
  simp only [← c98349, ← c98350, ← c98351] at e_1607_8
  have e_1607_9 := arithEq_of_rows h (row := 1607) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_9
  simp only [← c98352, ← c98353, k2, ← c98354] at e_1607_9
  have hr := e_1607_9
  simp only [e_1607_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1607 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15913)) := by
  have c98355 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98356 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98357 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98358 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98359 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98360 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1607_10 := arithEq_of_rows h (row := 1607) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_10
  simp only [← c98355, ← c98356, ← c98357] at e_1607_10
  have e_1607_11 := arithEq_of_rows h (row := 1607) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_11
  simp only [← c98358, ← c98359, k2, ← c98360] at e_1607_11
  have hr := e_1607_11
  simp only [e_1607_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1607 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15914)) := by
  have c98361 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98362 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98363 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98364 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98365 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98366 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1607_12 := arithEq_of_rows h (row := 1607) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_12
  simp only [← c98361, ← c98362, ← c98363] at e_1607_12
  have e_1607_13 := arithEq_of_rows h (row := 1607) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_13
  simp only [← c98364, ← c98365, k2, ← c98366] at e_1607_13
  have hr := e_1607_13
  simp only [e_1607_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1607 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15915)) := by
  have c98367 := (publicBatchWrapper86_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98368 := (publicBatchWrapper86_copies3074 a h).1
  have c98369 := (publicBatchWrapper86_copies3074 a h).2.1
  have c98370 := (publicBatchWrapper86_copies3074 a h).2.2.1
  have c98371 := (publicBatchWrapper86_copies3074 a h).2.2.2.1
  have c98372 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1607_14 := arithEq_of_rows h (row := 1607) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_14
  simp only [← c98367, ← c98368, ← c98369] at e_1607_14
  have e_1607_15 := arithEq_of_rows h (row := 1607) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_15
  simp only [← c98370, ← c98371, k2, ← c98372] at e_1607_15
  have hr := e_1607_15
  simp only [e_1607_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1607 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15916)) := by
  have c98373 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.1
  have c98374 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.1
  have c98375 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.1
  have c98376 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.1
  have c98377 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.1
  have c98378 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1607_16 := arithEq_of_rows h (row := 1607) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_16
  simp only [← c98373, ← c98374, ← c98375] at e_1607_16
  have e_1607_17 := arithEq_of_rows h (row := 1607) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_17
  simp only [← c98376, ← c98377, k2, ← c98378] at e_1607_17
  have hr := e_1607_17
  simp only [e_1607_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1607 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15917)) := by
  have c98379 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98380 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98381 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98382 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98383 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98384 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1607_18 := arithEq_of_rows h (row := 1607) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_18
  simp only [← c98379, ← c98380, ← c98381] at e_1607_18
  have e_1607_19 := arithEq_of_rows h (row := 1607) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_19
  simp only [← c98382, ← c98383, k2, ← c98384] at e_1607_19
  have hr := e_1607_19
  simp only [e_1607_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1608 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15918)) := by
  have c98385 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98386 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98387 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98388 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98389 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98390 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1608_0 := arithEq_of_rows h (row := 1608) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_0
  simp only [← c98385, ← c98386, ← c98387] at e_1608_0
  have e_1608_1 := arithEq_of_rows h (row := 1608) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_1
  simp only [← c98388, ← c98389, k2, ← c98390] at e_1608_1
  have hr := e_1608_1
  simp only [e_1608_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1608 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15919)) := by
  have c98391 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98392 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98393 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98394 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98395 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98396 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1608_2 := arithEq_of_rows h (row := 1608) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_2
  simp only [← c98391, ← c98392, ← c98393] at e_1608_2
  have e_1608_3 := arithEq_of_rows h (row := 1608) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_3
  simp only [← c98394, ← c98395, k2, ← c98396] at e_1608_3
  have hr := e_1608_3
  simp only [e_1608_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1608 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15920)) := by
  have c98397 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98398 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98399 := (publicBatchWrapper86_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98400 := (publicBatchWrapper86_copies3075 a h).1
  have c98401 := (publicBatchWrapper86_copies3075 a h).2.1
  have c98402 := (publicBatchWrapper86_copies3075 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1608_4 := arithEq_of_rows h (row := 1608) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_4
  simp only [← c98397, ← c98398, ← c98399] at e_1608_4
  have e_1608_5 := arithEq_of_rows h (row := 1608) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_5
  simp only [← c98400, ← c98401, k2, ← c98402] at e_1608_5
  have hr := e_1608_5
  simp only [e_1608_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1608 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15921)) := by
  have c98403 := (publicBatchWrapper86_copies3075 a h).2.2.2.1
  have c98404 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.1
  have c98405 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.1
  have c98406 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.1
  have c98407 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.1
  have c98408 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1608_6 := arithEq_of_rows h (row := 1608) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_6
  simp only [← c98403, ← c98404, ← c98405] at e_1608_6
  have e_1608_7 := arithEq_of_rows h (row := 1608) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_7
  simp only [← c98406, ← c98407, k2, ← c98408] at e_1608_7
  have hr := e_1608_7
  simp only [e_1608_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1608 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15922)) := by
  have c98409 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.1
  have c98410 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98411 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98412 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98413 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98414 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1608_8 := arithEq_of_rows h (row := 1608) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_8
  simp only [← c98409, ← c98410, ← c98411] at e_1608_8
  have e_1608_9 := arithEq_of_rows h (row := 1608) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_9
  simp only [← c98412, ← c98413, k2, ← c98414] at e_1608_9
  have hr := e_1608_9
  simp only [e_1608_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1608 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15923)) := by
  have c98415 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98416 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98417 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98418 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98419 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98420 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1608_10 := arithEq_of_rows h (row := 1608) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_10
  simp only [← c98415, ← c98416, ← c98417] at e_1608_10
  have e_1608_11 := arithEq_of_rows h (row := 1608) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_11
  simp only [← c98418, ← c98419, k2, ← c98420] at e_1608_11
  have hr := e_1608_11
  simp only [e_1608_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1608 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15924)) := by
  have c98421 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98422 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98423 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98424 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98425 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98426 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1608_12 := arithEq_of_rows h (row := 1608) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_12
  simp only [← c98421, ← c98422, ← c98423] at e_1608_12
  have e_1608_13 := arithEq_of_rows h (row := 1608) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_13
  simp only [← c98424, ← c98425, k2, ← c98426] at e_1608_13
  have hr := e_1608_13
  simp only [e_1608_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1608 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15925)) := by
  have c98427 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98428 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98429 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98430 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98431 := (publicBatchWrapper86_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98432 := (publicBatchWrapper86_copies3076 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1608_14 := arithEq_of_rows h (row := 1608) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_14
  simp only [← c98427, ← c98428, ← c98429] at e_1608_14
  have e_1608_15 := arithEq_of_rows h (row := 1608) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_15
  simp only [← c98430, ← c98431, k2, ← c98432] at e_1608_15
  have hr := e_1608_15
  simp only [e_1608_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1608 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15926)) := by
  have c98433 := (publicBatchWrapper86_copies3076 a h).2.1
  have c98434 := (publicBatchWrapper86_copies3076 a h).2.2.1
  have c98435 := (publicBatchWrapper86_copies3076 a h).2.2.2.1
  have c98436 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.1
  have c98437 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.1
  have c98438 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1608_16 := arithEq_of_rows h (row := 1608) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_16
  simp only [← c98433, ← c98434, ← c98435] at e_1608_16
  have e_1608_17 := arithEq_of_rows h (row := 1608) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_17
  simp only [← c98436, ← c98437, k2, ← c98438] at e_1608_17
  have hr := e_1608_17
  simp only [e_1608_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1608 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15927)) := by
  have c98439 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.1
  have c98440 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.1
  have c98441 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.1
  have c98442 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98443 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98444 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1608_18 := arithEq_of_rows h (row := 1608) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_18
  simp only [← c98439, ← c98440, ← c98441] at e_1608_18
  have e_1608_19 := arithEq_of_rows h (row := 1608) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_19
  simp only [← c98442, ← c98443, k2, ← c98444] at e_1608_19
  have hr := e_1608_19
  simp only [e_1608_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1609 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15928)) := by
  have c98445 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98446 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98447 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98448 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98449 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98450 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1609_0 := arithEq_of_rows h (row := 1609) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_0
  simp only [← c98445, ← c98446, ← c98447] at e_1609_0
  have e_1609_1 := arithEq_of_rows h (row := 1609) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_1
  simp only [← c98448, ← c98449, k2, ← c98450] at e_1609_1
  have hr := e_1609_1
  simp only [e_1609_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1609 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15929)) := by
  have c98451 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98452 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98453 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98454 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98455 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98456 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1609_2 := arithEq_of_rows h (row := 1609) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_2
  simp only [← c98451, ← c98452, ← c98453] at e_1609_2
  have e_1609_3 := arithEq_of_rows h (row := 1609) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_3
  simp only [← c98454, ← c98455, k2, ← c98456] at e_1609_3
  have hr := e_1609_3
  simp only [e_1609_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1609 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15930)) := by
  have c98457 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98458 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98459 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98460 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98461 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98462 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1609_4 := arithEq_of_rows h (row := 1609) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_4
  simp only [← c98457, ← c98458, ← c98459] at e_1609_4
  have e_1609_5 := arithEq_of_rows h (row := 1609) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_5
  simp only [← c98460, ← c98461, k2, ← c98462] at e_1609_5
  have hr := e_1609_5
  simp only [e_1609_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1609 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15931)) := by
  have c98463 := (publicBatchWrapper86_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98464 := (publicBatchWrapper86_copies3077 a h).1
  have c98465 := (publicBatchWrapper86_copies3077 a h).2.1
  have c98466 := (publicBatchWrapper86_copies3077 a h).2.2.1
  have c98467 := (publicBatchWrapper86_copies3077 a h).2.2.2.1
  have c98468 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1609_6 := arithEq_of_rows h (row := 1609) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_6
  simp only [← c98463, ← c98464, ← c98465] at e_1609_6
  have e_1609_7 := arithEq_of_rows h (row := 1609) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_7
  simp only [← c98466, ← c98467, k2, ← c98468] at e_1609_7
  have hr := e_1609_7
  simp only [e_1609_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1609 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15932)) := by
  have c98469 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.1
  have c98470 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.1
  have c98471 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.1
  have c98472 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.1
  have c98473 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.1
  have c98474 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1609_8 := arithEq_of_rows h (row := 1609) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_8
  simp only [← c98469, ← c98470, ← c98471] at e_1609_8
  have e_1609_9 := arithEq_of_rows h (row := 1609) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_9
  simp only [← c98472, ← c98473, k2, ← c98474] at e_1609_9
  have hr := e_1609_9
  simp only [e_1609_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1609 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15933)) := by
  have c98475 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98476 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98477 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98478 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98479 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98480 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1609_10 := arithEq_of_rows h (row := 1609) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_10
  simp only [← c98475, ← c98476, ← c98477] at e_1609_10
  have e_1609_11 := arithEq_of_rows h (row := 1609) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_11
  simp only [← c98478, ← c98479, k2, ← c98480] at e_1609_11
  have hr := e_1609_11
  simp only [e_1609_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1609 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15934)) := by
  have c98481 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98482 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98483 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98484 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98485 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98486 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1609_12 := arithEq_of_rows h (row := 1609) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_12
  simp only [← c98481, ← c98482, ← c98483] at e_1609_12
  have e_1609_13 := arithEq_of_rows h (row := 1609) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_13
  simp only [← c98484, ← c98485, k2, ← c98486] at e_1609_13
  have hr := e_1609_13
  simp only [e_1609_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1609 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15935)) := by
  have c98487 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98488 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98489 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98490 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98491 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98492 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1609_14 := arithEq_of_rows h (row := 1609) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_14
  simp only [← c98487, ← c98488, ← c98489] at e_1609_14
  have e_1609_15 := arithEq_of_rows h (row := 1609) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_15
  simp only [← c98490, ← c98491, k2, ← c98492] at e_1609_15
  have hr := e_1609_15
  simp only [e_1609_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1609 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15936)) := by
  have c98493 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98494 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98495 := (publicBatchWrapper86_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98496 := (publicBatchWrapper86_copies3078 a h).1
  have c98497 := (publicBatchWrapper86_copies3078 a h).2.1
  have c98498 := (publicBatchWrapper86_copies3078 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1609_16 := arithEq_of_rows h (row := 1609) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_16
  simp only [← c98493, ← c98494, ← c98495] at e_1609_16
  have e_1609_17 := arithEq_of_rows h (row := 1609) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_17
  simp only [← c98496, ← c98497, k2, ← c98498] at e_1609_17
  have hr := e_1609_17
  simp only [e_1609_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1609 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15937)) := by
  have c98499 := (publicBatchWrapper86_copies3078 a h).2.2.2.1
  have c98500 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.1
  have c98501 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.1
  have c98502 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.1
  have c98503 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.1
  have c98504 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1609_18 := arithEq_of_rows h (row := 1609) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_18
  simp only [← c98499, ← c98500, ← c98501] at e_1609_18
  have e_1609_19 := arithEq_of_rows h (row := 1609) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_19
  simp only [← c98502, ← c98503, k2, ← c98504] at e_1609_19
  have hr := e_1609_19
  simp only [e_1609_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1610 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15938)) := by
  have c98505 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.1
  have c98506 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98507 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98508 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98509 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98510 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1610_0 := arithEq_of_rows h (row := 1610) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_0
  simp only [← c98505, ← c98506, ← c98507] at e_1610_0
  have e_1610_1 := arithEq_of_rows h (row := 1610) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_1
  simp only [← c98508, ← c98509, k2, ← c98510] at e_1610_1
  have hr := e_1610_1
  simp only [e_1610_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1610 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15939)) := by
  have c98511 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98512 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98513 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98514 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98515 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98516 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1610_2 := arithEq_of_rows h (row := 1610) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_2
  simp only [← c98511, ← c98512, ← c98513] at e_1610_2
  have e_1610_3 := arithEq_of_rows h (row := 1610) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_3
  simp only [← c98514, ← c98515, k2, ← c98516] at e_1610_3
  have hr := e_1610_3
  simp only [e_1610_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1610 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15940)) := by
  have c98517 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98518 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98519 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98520 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98521 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98522 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1610_4 := arithEq_of_rows h (row := 1610) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_4
  simp only [← c98517, ← c98518, ← c98519] at e_1610_4
  have e_1610_5 := arithEq_of_rows h (row := 1610) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_5
  simp only [← c98520, ← c98521, k2, ← c98522] at e_1610_5
  have hr := e_1610_5
  simp only [e_1610_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1610 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15941)) := by
  have c98523 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98524 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98525 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98526 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98527 := (publicBatchWrapper86_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98528 := (publicBatchWrapper86_copies3079 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1610_6 := arithEq_of_rows h (row := 1610) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_6
  simp only [← c98523, ← c98524, ← c98525] at e_1610_6
  have e_1610_7 := arithEq_of_rows h (row := 1610) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_7
  simp only [← c98526, ← c98527, k2, ← c98528] at e_1610_7
  have hr := e_1610_7
  simp only [e_1610_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1610 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15942)) := by
  have c98529 := (publicBatchWrapper86_copies3079 a h).2.1
  have c98530 := (publicBatchWrapper86_copies3079 a h).2.2.1
  have c98531 := (publicBatchWrapper86_copies3079 a h).2.2.2.1
  have c98532 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.1
  have c98533 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.1
  have c98534 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1610_8 := arithEq_of_rows h (row := 1610) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_8
  simp only [← c98529, ← c98530, ← c98531] at e_1610_8
  have e_1610_9 := arithEq_of_rows h (row := 1610) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_9
  simp only [← c98532, ← c98533, k2, ← c98534] at e_1610_9
  have hr := e_1610_9
  simp only [e_1610_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1610 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15943)) := by
  have c98535 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.1
  have c98536 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.1
  have c98537 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.1
  have c98538 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98539 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98540 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1610_10 := arithEq_of_rows h (row := 1610) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_10
  simp only [← c98535, ← c98536, ← c98537] at e_1610_10
  have e_1610_11 := arithEq_of_rows h (row := 1610) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_11
  simp only [← c98538, ← c98539, k2, ← c98540] at e_1610_11
  have hr := e_1610_11
  simp only [e_1610_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1610 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15944)) := by
  have c98541 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98542 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98543 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98544 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98545 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98546 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1610_12 := arithEq_of_rows h (row := 1610) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_12
  simp only [← c98541, ← c98542, ← c98543] at e_1610_12
  have e_1610_13 := arithEq_of_rows h (row := 1610) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_13
  simp only [← c98544, ← c98545, k2, ← c98546] at e_1610_13
  have hr := e_1610_13
  simp only [e_1610_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1610 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15945)) := by
  have c98547 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98548 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98549 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98550 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98551 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98552 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1610_14 := arithEq_of_rows h (row := 1610) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_14
  simp only [← c98547, ← c98548, ← c98549] at e_1610_14
  have e_1610_15 := arithEq_of_rows h (row := 1610) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_15
  simp only [← c98550, ← c98551, k2, ← c98552] at e_1610_15
  have hr := e_1610_15
  simp only [e_1610_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1610 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15946)) := by
  have c98553 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98554 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98555 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98556 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98557 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98558 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1610_16 := arithEq_of_rows h (row := 1610) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_16
  simp only [← c98553, ← c98554, ← c98555] at e_1610_16
  have e_1610_17 := arithEq_of_rows h (row := 1610) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_17
  simp only [← c98556, ← c98557, k2, ← c98558] at e_1610_17
  have hr := e_1610_17
  simp only [e_1610_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1610 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15947)) := by
  have c98559 := (publicBatchWrapper86_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98560 := (publicBatchWrapper86_copies3080 a h).1
  have c98561 := (publicBatchWrapper86_copies3080 a h).2.1
  have c98562 := (publicBatchWrapper86_copies3080 a h).2.2.1
  have c98563 := (publicBatchWrapper86_copies3080 a h).2.2.2.1
  have c98564 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1610_18 := arithEq_of_rows h (row := 1610) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_18
  simp only [← c98559, ← c98560, ← c98561] at e_1610_18
  have e_1610_19 := arithEq_of_rows h (row := 1610) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_19
  simp only [← c98562, ← c98563, k2, ← c98564] at e_1610_19
  have hr := e_1610_19
  simp only [e_1610_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1611 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15948)) := by
  have c98565 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.1
  have c98566 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.1
  have c98567 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.1
  have c98568 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.1
  have c98569 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.1
  have c98570 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1611_0 := arithEq_of_rows h (row := 1611) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_0
  simp only [← c98565, ← c98566, ← c98567] at e_1611_0
  have e_1611_1 := arithEq_of_rows h (row := 1611) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_1
  simp only [← c98568, ← c98569, k2, ← c98570] at e_1611_1
  have hr := e_1611_1
  simp only [e_1611_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1611 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15949)) := by
  have c98571 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98572 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98573 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98574 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98575 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98576 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1611_2 := arithEq_of_rows h (row := 1611) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_2
  simp only [← c98571, ← c98572, ← c98573] at e_1611_2
  have e_1611_3 := arithEq_of_rows h (row := 1611) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_3
  simp only [← c98574, ← c98575, k2, ← c98576] at e_1611_3
  have hr := e_1611_3
  simp only [e_1611_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1611 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15950)) := by
  have c98577 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98578 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98579 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98580 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98581 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98582 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1611_4 := arithEq_of_rows h (row := 1611) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_4
  simp only [← c98577, ← c98578, ← c98579] at e_1611_4
  have e_1611_5 := arithEq_of_rows h (row := 1611) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_5
  simp only [← c98580, ← c98581, k2, ← c98582] at e_1611_5
  have hr := e_1611_5
  simp only [e_1611_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1611 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15951)) := by
  have c98583 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98584 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98585 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98586 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98587 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98588 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1611_6 := arithEq_of_rows h (row := 1611) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_6
  simp only [← c98583, ← c98584, ← c98585] at e_1611_6
  have e_1611_7 := arithEq_of_rows h (row := 1611) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_7
  simp only [← c98586, ← c98587, k2, ← c98588] at e_1611_7
  have hr := e_1611_7
  simp only [e_1611_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1611 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15952)) := by
  have c98589 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98590 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98591 := (publicBatchWrapper86_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98592 := (publicBatchWrapper86_copies3081 a h).1
  have c98593 := (publicBatchWrapper86_copies3081 a h).2.1
  have c98594 := (publicBatchWrapper86_copies3081 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1611_8 := arithEq_of_rows h (row := 1611) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_8
  simp only [← c98589, ← c98590, ← c98591] at e_1611_8
  have e_1611_9 := arithEq_of_rows h (row := 1611) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_9
  simp only [← c98592, ← c98593, k2, ← c98594] at e_1611_9
  have hr := e_1611_9
  simp only [e_1611_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1611 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15953)) := by
  have c98595 := (publicBatchWrapper86_copies3081 a h).2.2.2.1
  have c98596 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.1
  have c98597 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.1
  have c98598 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.1
  have c98599 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.1
  have c98600 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1611_10 := arithEq_of_rows h (row := 1611) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_10
  simp only [← c98595, ← c98596, ← c98597] at e_1611_10
  have e_1611_11 := arithEq_of_rows h (row := 1611) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_11
  simp only [← c98598, ← c98599, k2, ← c98600] at e_1611_11
  have hr := e_1611_11
  simp only [e_1611_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1611 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15954)) := by
  have c98601 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.1
  have c98602 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98603 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98604 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98605 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98606 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1611_12 := arithEq_of_rows h (row := 1611) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_12
  simp only [← c98601, ← c98602, ← c98603] at e_1611_12
  have e_1611_13 := arithEq_of_rows h (row := 1611) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_13
  simp only [← c98604, ← c98605, k2, ← c98606] at e_1611_13
  have hr := e_1611_13
  simp only [e_1611_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1611 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15955)) := by
  have c98607 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98608 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98609 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98610 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98611 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98612 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1611_14 := arithEq_of_rows h (row := 1611) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_14
  simp only [← c98607, ← c98608, ← c98609] at e_1611_14
  have e_1611_15 := arithEq_of_rows h (row := 1611) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_15
  simp only [← c98610, ← c98611, k2, ← c98612] at e_1611_15
  have hr := e_1611_15
  simp only [e_1611_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1611 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15956)) := by
  have c98613 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98614 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98615 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98616 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98617 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98618 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1611_16 := arithEq_of_rows h (row := 1611) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_16
  simp only [← c98613, ← c98614, ← c98615] at e_1611_16
  have e_1611_17 := arithEq_of_rows h (row := 1611) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_17
  simp only [← c98616, ← c98617, k2, ← c98618] at e_1611_17
  have hr := e_1611_17
  simp only [e_1611_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1611 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15957)) := by
  have c98619 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98620 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98621 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98622 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98623 := (publicBatchWrapper86_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98624 := (publicBatchWrapper86_copies3082 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1611_18 := arithEq_of_rows h (row := 1611) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_18
  simp only [← c98619, ← c98620, ← c98621] at e_1611_18
  have e_1611_19 := arithEq_of_rows h (row := 1611) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_19
  simp only [← c98622, ← c98623, k2, ← c98624] at e_1611_19
  have hr := e_1611_19
  simp only [e_1611_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1612 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15958)) := by
  have c98625 := (publicBatchWrapper86_copies3082 a h).2.1
  have c98626 := (publicBatchWrapper86_copies3082 a h).2.2.1
  have c98627 := (publicBatchWrapper86_copies3082 a h).2.2.2.1
  have c98628 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.1
  have c98629 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.1
  have c98630 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1612_0 := arithEq_of_rows h (row := 1612) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_0
  simp only [← c98625, ← c98626, ← c98627] at e_1612_0
  have e_1612_1 := arithEq_of_rows h (row := 1612) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_1
  simp only [← c98628, ← c98629, k2, ← c98630] at e_1612_1
  have hr := e_1612_1
  simp only [e_1612_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1612 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15959)) := by
  have c98631 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.1
  have c98632 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.1
  have c98633 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.1
  have c98634 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98635 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98636 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1612_2 := arithEq_of_rows h (row := 1612) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_2
  simp only [← c98631, ← c98632, ← c98633] at e_1612_2
  have e_1612_3 := arithEq_of_rows h (row := 1612) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_3
  simp only [← c98634, ← c98635, k2, ← c98636] at e_1612_3
  have hr := e_1612_3
  simp only [e_1612_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1612 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15960)) := by
  have c98637 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98638 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98639 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98640 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98641 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98642 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1612_4 := arithEq_of_rows h (row := 1612) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_4
  simp only [← c98637, ← c98638, ← c98639] at e_1612_4
  have e_1612_5 := arithEq_of_rows h (row := 1612) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_5
  simp only [← c98640, ← c98641, k2, ← c98642] at e_1612_5
  have hr := e_1612_5
  simp only [e_1612_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1612 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15961)) := by
  have c98643 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98644 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98645 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98646 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98647 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98648 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1612_6 := arithEq_of_rows h (row := 1612) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_6
  simp only [← c98643, ← c98644, ← c98645] at e_1612_6
  have e_1612_7 := arithEq_of_rows h (row := 1612) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_7
  simp only [← c98646, ← c98647, k2, ← c98648] at e_1612_7
  have hr := e_1612_7
  simp only [e_1612_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1612 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15962)) := by
  have c98649 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98650 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98651 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98652 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98653 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98654 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1612_8 := arithEq_of_rows h (row := 1612) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_8
  simp only [← c98649, ← c98650, ← c98651] at e_1612_8
  have e_1612_9 := arithEq_of_rows h (row := 1612) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_9
  simp only [← c98652, ← c98653, k2, ← c98654] at e_1612_9
  have hr := e_1612_9
  simp only [e_1612_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1612 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15963)) := by
  have c98655 := (publicBatchWrapper86_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98656 := (publicBatchWrapper86_copies3083 a h).1
  have c98657 := (publicBatchWrapper86_copies3083 a h).2.1
  have c98658 := (publicBatchWrapper86_copies3083 a h).2.2.1
  have c98659 := (publicBatchWrapper86_copies3083 a h).2.2.2.1
  have c98660 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1612_10 := arithEq_of_rows h (row := 1612) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_10
  simp only [← c98655, ← c98656, ← c98657] at e_1612_10
  have e_1612_11 := arithEq_of_rows h (row := 1612) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_11
  simp only [← c98658, ← c98659, k2, ← c98660] at e_1612_11
  have hr := e_1612_11
  simp only [e_1612_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1612 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15964)) := by
  have c98661 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.1
  have c98662 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.1
  have c98663 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.1
  have c98664 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.1
  have c98665 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.1
  have c98666 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1612_12 := arithEq_of_rows h (row := 1612) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_12
  simp only [← c98661, ← c98662, ← c98663] at e_1612_12
  have e_1612_13 := arithEq_of_rows h (row := 1612) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_13
  simp only [← c98664, ← c98665, k2, ← c98666] at e_1612_13
  have hr := e_1612_13
  simp only [e_1612_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1612 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15965)) := by
  have c98667 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98668 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98669 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98670 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98671 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98672 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1612_14 := arithEq_of_rows h (row := 1612) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_14
  simp only [← c98667, ← c98668, ← c98669] at e_1612_14
  have e_1612_15 := arithEq_of_rows h (row := 1612) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_15
  simp only [← c98670, ← c98671, k2, ← c98672] at e_1612_15
  have hr := e_1612_15
  simp only [e_1612_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1612 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15966)) := by
  have c98673 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98674 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98675 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98676 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98677 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98678 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1612_16 := arithEq_of_rows h (row := 1612) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_16
  simp only [← c98673, ← c98674, ← c98675] at e_1612_16
  have e_1612_17 := arithEq_of_rows h (row := 1612) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_17
  simp only [← c98676, ← c98677, k2, ← c98678] at e_1612_17
  have hr := e_1612_17
  simp only [e_1612_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1612 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15967)) := by
  have c98679 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98680 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98681 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98682 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98683 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98684 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1612_18 := arithEq_of_rows h (row := 1612) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_18
  simp only [← c98679, ← c98680, ← c98681] at e_1612_18
  have e_1612_19 := arithEq_of_rows h (row := 1612) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_19
  simp only [← c98682, ← c98683, k2, ← c98684] at e_1612_19
  have hr := e_1612_19
  simp only [e_1612_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1613 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15968)) := by
  have c98685 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98686 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98687 := (publicBatchWrapper86_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98688 := (publicBatchWrapper86_copies3084 a h).1
  have c98689 := (publicBatchWrapper86_copies3084 a h).2.1
  have c98690 := (publicBatchWrapper86_copies3084 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1613_0 := arithEq_of_rows h (row := 1613) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_0
  simp only [← c98685, ← c98686, ← c98687] at e_1613_0
  have e_1613_1 := arithEq_of_rows h (row := 1613) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_1
  simp only [← c98688, ← c98689, k2, ← c98690] at e_1613_1
  have hr := e_1613_1
  simp only [e_1613_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1613 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15969)) := by
  have c98691 := (publicBatchWrapper86_copies3084 a h).2.2.2.1
  have c98692 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.1
  have c98693 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.1
  have c98694 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.1
  have c98695 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.1
  have c98696 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1613_2 := arithEq_of_rows h (row := 1613) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_2
  simp only [← c98691, ← c98692, ← c98693] at e_1613_2
  have e_1613_3 := arithEq_of_rows h (row := 1613) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_3
  simp only [← c98694, ← c98695, k2, ← c98696] at e_1613_3
  have hr := e_1613_3
  simp only [e_1613_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1613 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16010)) := by
  have c98697 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.1
  have c98698 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98699 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98700 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98701 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98702 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1613_4 := arithEq_of_rows h (row := 1613) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_4
  simp only [← c98697, ← c98698, ← c98699] at e_1613_4
  have e_1613_5 := arithEq_of_rows h (row := 1613) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_5
  simp only [← c98700, ← c98701, k2, ← c98702] at e_1613_5
  have hr := e_1613_5
  simp only [e_1613_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1613 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16011)) := by
  have c98703 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98704 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98705 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98706 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98707 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98708 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1613_6 := arithEq_of_rows h (row := 1613) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_6
  simp only [← c98703, ← c98704, ← c98705] at e_1613_6
  have e_1613_7 := arithEq_of_rows h (row := 1613) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_7
  simp only [← c98706, ← c98707, k2, ← c98708] at e_1613_7
  have hr := e_1613_7
  simp only [e_1613_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1613 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16012)) := by
  have c98709 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98710 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98711 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98712 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98713 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98714 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1613_8 := arithEq_of_rows h (row := 1613) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_8
  simp only [← c98709, ← c98710, ← c98711] at e_1613_8
  have e_1613_9 := arithEq_of_rows h (row := 1613) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_9
  simp only [← c98712, ← c98713, k2, ← c98714] at e_1613_9
  have hr := e_1613_9
  simp only [e_1613_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1613 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16013)) := by
  have c98715 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98716 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98717 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98718 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98719 := (publicBatchWrapper86_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98720 := (publicBatchWrapper86_copies3085 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1613_10 := arithEq_of_rows h (row := 1613) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_10
  simp only [← c98715, ← c98716, ← c98717] at e_1613_10
  have e_1613_11 := arithEq_of_rows h (row := 1613) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_11
  simp only [← c98718, ← c98719, k2, ← c98720] at e_1613_11
  have hr := e_1613_11
  simp only [e_1613_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1613 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16014)) := by
  have c98721 := (publicBatchWrapper86_copies3085 a h).2.1
  have c98722 := (publicBatchWrapper86_copies3085 a h).2.2.1
  have c98723 := (publicBatchWrapper86_copies3085 a h).2.2.2.1
  have c98724 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.1
  have c98725 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.1
  have c98726 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1613_12 := arithEq_of_rows h (row := 1613) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_12
  simp only [← c98721, ← c98722, ← c98723] at e_1613_12
  have e_1613_13 := arithEq_of_rows h (row := 1613) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_13
  simp only [← c98724, ← c98725, k2, ← c98726] at e_1613_13
  have hr := e_1613_13
  simp only [e_1613_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1613 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16015)) := by
  have c98727 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.1
  have c98728 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.1
  have c98729 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.1
  have c98730 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98731 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98732 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1613_14 := arithEq_of_rows h (row := 1613) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_14
  simp only [← c98727, ← c98728, ← c98729] at e_1613_14
  have e_1613_15 := arithEq_of_rows h (row := 1613) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_15
  simp only [← c98730, ← c98731, k2, ← c98732] at e_1613_15
  have hr := e_1613_15
  simp only [e_1613_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1613 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16016)) := by
  have c98733 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98734 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98735 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98736 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98737 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98738 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1613_16 := arithEq_of_rows h (row := 1613) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_16
  simp only [← c98733, ← c98734, ← c98735] at e_1613_16
  have e_1613_17 := arithEq_of_rows h (row := 1613) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_17
  simp only [← c98736, ← c98737, k2, ← c98738] at e_1613_17
  have hr := e_1613_17
  simp only [e_1613_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1613 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16017)) := by
  have c98739 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98740 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98741 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98742 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98743 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98744 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1613_18 := arithEq_of_rows h (row := 1613) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_18
  simp only [← c98739, ← c98740, ← c98741] at e_1613_18
  have e_1613_19 := arithEq_of_rows h (row := 1613) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_19
  simp only [← c98742, ← c98743, k2, ← c98744] at e_1613_19
  have hr := e_1613_19
  simp only [e_1613_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1614 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16018)) := by
  have c98745 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98746 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98747 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98748 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98749 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98750 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1614_0 := arithEq_of_rows h (row := 1614) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_0
  simp only [← c98745, ← c98746, ← c98747] at e_1614_0
  have e_1614_1 := arithEq_of_rows h (row := 1614) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_1
  simp only [← c98748, ← c98749, k2, ← c98750] at e_1614_1
  have hr := e_1614_1
  simp only [e_1614_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1614 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16019)) := by
  have c98751 := (publicBatchWrapper86_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98752 := (publicBatchWrapper86_copies3086 a h).1
  have c98753 := (publicBatchWrapper86_copies3086 a h).2.1
  have c98754 := (publicBatchWrapper86_copies3086 a h).2.2.1
  have c98755 := (publicBatchWrapper86_copies3086 a h).2.2.2.1
  have c98756 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1614_2 := arithEq_of_rows h (row := 1614) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_2
  simp only [← c98751, ← c98752, ← c98753] at e_1614_2
  have e_1614_3 := arithEq_of_rows h (row := 1614) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_3
  simp only [← c98754, ← c98755, k2, ← c98756] at e_1614_3
  have hr := e_1614_3
  simp only [e_1614_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1614 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16020)) := by
  have c98757 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.1
  have c98758 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.1
  have c98759 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.1
  have c98760 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.1
  have c98761 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.1
  have c98762 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1614_4 := arithEq_of_rows h (row := 1614) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_4
  simp only [← c98757, ← c98758, ← c98759] at e_1614_4
  have e_1614_5 := arithEq_of_rows h (row := 1614) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_5
  simp only [← c98760, ← c98761, k2, ← c98762] at e_1614_5
  have hr := e_1614_5
  simp only [e_1614_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1614 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16021)) := by
  have c98763 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98764 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98765 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98766 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98767 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98768 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1614_6 := arithEq_of_rows h (row := 1614) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_6
  simp only [← c98763, ← c98764, ← c98765] at e_1614_6
  have e_1614_7 := arithEq_of_rows h (row := 1614) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_7
  simp only [← c98766, ← c98767, k2, ← c98768] at e_1614_7
  have hr := e_1614_7
  simp only [e_1614_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1614 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16022)) := by
  have c98769 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98770 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98771 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98772 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98773 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98774 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1614_8 := arithEq_of_rows h (row := 1614) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_8
  simp only [← c98769, ← c98770, ← c98771] at e_1614_8
  have e_1614_9 := arithEq_of_rows h (row := 1614) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_9
  simp only [← c98772, ← c98773, k2, ← c98774] at e_1614_9
  have hr := e_1614_9
  simp only [e_1614_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1614 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16023)) := by
  have c98775 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98776 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98777 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98778 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98779 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98780 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1614_10 := arithEq_of_rows h (row := 1614) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_10
  simp only [← c98775, ← c98776, ← c98777] at e_1614_10
  have e_1614_11 := arithEq_of_rows h (row := 1614) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_11
  simp only [← c98778, ← c98779, k2, ← c98780] at e_1614_11
  have hr := e_1614_11
  simp only [e_1614_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1614 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16024)) := by
  have c98781 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98782 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98783 := (publicBatchWrapper86_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98784 := (publicBatchWrapper86_copies3087 a h).1
  have c98785 := (publicBatchWrapper86_copies3087 a h).2.1
  have c98786 := (publicBatchWrapper86_copies3087 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1614_12 := arithEq_of_rows h (row := 1614) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_12
  simp only [← c98781, ← c98782, ← c98783] at e_1614_12
  have e_1614_13 := arithEq_of_rows h (row := 1614) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_13
  simp only [← c98784, ← c98785, k2, ← c98786] at e_1614_13
  have hr := e_1614_13
  simp only [e_1614_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1614 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16025)) := by
  have c98787 := (publicBatchWrapper86_copies3087 a h).2.2.2.1
  have c98788 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.1
  have c98789 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.1
  have c98790 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.1
  have c98791 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.1
  have c98792 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1614_14 := arithEq_of_rows h (row := 1614) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_14
  simp only [← c98787, ← c98788, ← c98789] at e_1614_14
  have e_1614_15 := arithEq_of_rows h (row := 1614) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_15
  simp only [← c98790, ← c98791, k2, ← c98792] at e_1614_15
  have hr := e_1614_15
  simp only [e_1614_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1614 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16026)) := by
  have c98793 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.1
  have c98794 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98795 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98796 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98797 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98798 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1614_16 := arithEq_of_rows h (row := 1614) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_16
  simp only [← c98793, ← c98794, ← c98795] at e_1614_16
  have e_1614_17 := arithEq_of_rows h (row := 1614) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_17
  simp only [← c98796, ← c98797, k2, ← c98798] at e_1614_17
  have hr := e_1614_17
  simp only [e_1614_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1614 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16027)) := by
  have c98799 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98800 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98801 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98802 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98803 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98804 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1614_18 := arithEq_of_rows h (row := 1614) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_18
  simp only [← c98799, ← c98800, ← c98801] at e_1614_18
  have e_1614_19 := arithEq_of_rows h (row := 1614) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_19
  simp only [← c98802, ← c98803, k2, ← c98804] at e_1614_19
  have hr := e_1614_19
  simp only [e_1614_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1615 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16028)) := by
  have c98805 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98806 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98807 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98808 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98809 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98810 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1615_0 := arithEq_of_rows h (row := 1615) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_0
  simp only [← c98805, ← c98806, ← c98807] at e_1615_0
  have e_1615_1 := arithEq_of_rows h (row := 1615) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_1
  simp only [← c98808, ← c98809, k2, ← c98810] at e_1615_1
  have hr := e_1615_1
  simp only [e_1615_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1615 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16029)) := by
  have c98811 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98812 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98813 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98814 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98815 := (publicBatchWrapper86_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98816 := (publicBatchWrapper86_copies3088 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1615_2 := arithEq_of_rows h (row := 1615) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_2
  simp only [← c98811, ← c98812, ← c98813] at e_1615_2
  have e_1615_3 := arithEq_of_rows h (row := 1615) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_3
  simp only [← c98814, ← c98815, k2, ← c98816] at e_1615_3
  have hr := e_1615_3
  simp only [e_1615_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1615 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16030)) := by
  have c98817 := (publicBatchWrapper86_copies3088 a h).2.1
  have c98818 := (publicBatchWrapper86_copies3088 a h).2.2.1
  have c98819 := (publicBatchWrapper86_copies3088 a h).2.2.2.1
  have c98820 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.1
  have c98821 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.1
  have c98822 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1615_4 := arithEq_of_rows h (row := 1615) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_4
  simp only [← c98817, ← c98818, ← c98819] at e_1615_4
  have e_1615_5 := arithEq_of_rows h (row := 1615) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_5
  simp only [← c98820, ← c98821, k2, ← c98822] at e_1615_5
  have hr := e_1615_5
  simp only [e_1615_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1615 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16031)) := by
  have c98823 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.1
  have c98824 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.1
  have c98825 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.1
  have c98826 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98827 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98828 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1615_6 := arithEq_of_rows h (row := 1615) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_6
  simp only [← c98823, ← c98824, ← c98825] at e_1615_6
  have e_1615_7 := arithEq_of_rows h (row := 1615) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_7
  simp only [← c98826, ← c98827, k2, ← c98828] at e_1615_7
  have hr := e_1615_7
  simp only [e_1615_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1615 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16032)) := by
  have c98829 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98830 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98831 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98832 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98833 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98834 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1615_8 := arithEq_of_rows h (row := 1615) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_8
  simp only [← c98829, ← c98830, ← c98831] at e_1615_8
  have e_1615_9 := arithEq_of_rows h (row := 1615) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_9
  simp only [← c98832, ← c98833, k2, ← c98834] at e_1615_9
  have hr := e_1615_9
  simp only [e_1615_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1615 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16033)) := by
  have c98835 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98836 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98837 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98838 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98839 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98840 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1615_10 := arithEq_of_rows h (row := 1615) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_10
  simp only [← c98835, ← c98836, ← c98837] at e_1615_10
  have e_1615_11 := arithEq_of_rows h (row := 1615) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_11
  simp only [← c98838, ← c98839, k2, ← c98840] at e_1615_11
  have hr := e_1615_11
  simp only [e_1615_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1615 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16034)) := by
  have c98841 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98842 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98843 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98844 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98845 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98846 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1615_12 := arithEq_of_rows h (row := 1615) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_12
  simp only [← c98841, ← c98842, ← c98843] at e_1615_12
  have e_1615_13 := arithEq_of_rows h (row := 1615) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_13
  simp only [← c98844, ← c98845, k2, ← c98846] at e_1615_13
  have hr := e_1615_13
  simp only [e_1615_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1615 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16035)) := by
  have c98847 := (publicBatchWrapper86_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98848 := (publicBatchWrapper86_copies3089 a h).1
  have c98849 := (publicBatchWrapper86_copies3089 a h).2.1
  have c98850 := (publicBatchWrapper86_copies3089 a h).2.2.1
  have c98851 := (publicBatchWrapper86_copies3089 a h).2.2.2.1
  have c98852 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1615_14 := arithEq_of_rows h (row := 1615) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_14
  simp only [← c98847, ← c98848, ← c98849] at e_1615_14
  have e_1615_15 := arithEq_of_rows h (row := 1615) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_15
  simp only [← c98850, ← c98851, k2, ← c98852] at e_1615_15
  have hr := e_1615_15
  simp only [e_1615_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1615 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16036)) := by
  have c98853 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.1
  have c98854 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.1
  have c98855 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.1
  have c98856 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.1
  have c98857 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.1
  have c98858 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1615_16 := arithEq_of_rows h (row := 1615) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_16
  simp only [← c98853, ← c98854, ← c98855] at e_1615_16
  have e_1615_17 := arithEq_of_rows h (row := 1615) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_17
  simp only [← c98856, ← c98857, k2, ← c98858] at e_1615_17
  have hr := e_1615_17
  simp only [e_1615_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1615 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16037)) := by
  have c98859 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98860 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98861 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98862 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98863 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98864 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1615_18 := arithEq_of_rows h (row := 1615) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_18
  simp only [← c98859, ← c98860, ← c98861] at e_1615_18
  have e_1615_19 := arithEq_of_rows h (row := 1615) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_19
  simp only [← c98862, ← c98863, k2, ← c98864] at e_1615_19
  have hr := e_1615_19
  simp only [e_1615_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1616 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16038)) := by
  have c98865 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98866 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98867 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98868 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98869 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98870 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1616_0 := arithEq_of_rows h (row := 1616) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_0
  simp only [← c98865, ← c98866, ← c98867] at e_1616_0
  have e_1616_1 := arithEq_of_rows h (row := 1616) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_1
  simp only [← c98868, ← c98869, k2, ← c98870] at e_1616_1
  have hr := e_1616_1
  simp only [e_1616_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1616 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16039)) := by
  have c98871 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98872 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98873 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98874 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98875 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98876 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1616_2 := arithEq_of_rows h (row := 1616) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_2
  simp only [← c98871, ← c98872, ← c98873] at e_1616_2
  have e_1616_3 := arithEq_of_rows h (row := 1616) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_3
  simp only [← c98874, ← c98875, k2, ← c98876] at e_1616_3
  have hr := e_1616_3
  simp only [e_1616_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1616 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16040)) := by
  have c98877 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98878 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98879 := (publicBatchWrapper86_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98880 := (publicBatchWrapper86_copies3090 a h).1
  have c98881 := (publicBatchWrapper86_copies3090 a h).2.1
  have c98882 := (publicBatchWrapper86_copies3090 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1616_4 := arithEq_of_rows h (row := 1616) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_4
  simp only [← c98877, ← c98878, ← c98879] at e_1616_4
  have e_1616_5 := arithEq_of_rows h (row := 1616) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_5
  simp only [← c98880, ← c98881, k2, ← c98882] at e_1616_5
  have hr := e_1616_5
  simp only [e_1616_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1616 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16041)) := by
  have c98883 := (publicBatchWrapper86_copies3090 a h).2.2.2.1
  have c98884 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.1
  have c98885 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.1
  have c98886 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.1
  have c98887 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.1
  have c98888 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1616_6 := arithEq_of_rows h (row := 1616) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_6
  simp only [← c98883, ← c98884, ← c98885] at e_1616_6
  have e_1616_7 := arithEq_of_rows h (row := 1616) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_7
  simp only [← c98886, ← c98887, k2, ← c98888] at e_1616_7
  have hr := e_1616_7
  simp only [e_1616_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1616 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16042)) := by
  have c98889 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.1
  have c98890 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98891 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98892 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98893 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98894 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1616_8 := arithEq_of_rows h (row := 1616) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_8
  simp only [← c98889, ← c98890, ← c98891] at e_1616_8
  have e_1616_9 := arithEq_of_rows h (row := 1616) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_9
  simp only [← c98892, ← c98893, k2, ← c98894] at e_1616_9
  have hr := e_1616_9
  simp only [e_1616_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1616 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16043)) := by
  have c98895 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98896 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98897 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98898 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98899 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98900 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1616_10 := arithEq_of_rows h (row := 1616) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_10
  simp only [← c98895, ← c98896, ← c98897] at e_1616_10
  have e_1616_11 := arithEq_of_rows h (row := 1616) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_11
  simp only [← c98898, ← c98899, k2, ← c98900] at e_1616_11
  have hr := e_1616_11
  simp only [e_1616_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1616 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16044)) := by
  have c98901 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98902 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98903 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98904 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98905 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98906 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1616_12 := arithEq_of_rows h (row := 1616) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_12
  simp only [← c98901, ← c98902, ← c98903] at e_1616_12
  have e_1616_13 := arithEq_of_rows h (row := 1616) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_13
  simp only [← c98904, ← c98905, k2, ← c98906] at e_1616_13
  have hr := e_1616_13
  simp only [e_1616_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1616 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16045)) := by
  have c98907 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98908 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98909 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98910 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98911 := (publicBatchWrapper86_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98912 := (publicBatchWrapper86_copies3091 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1616_14 := arithEq_of_rows h (row := 1616) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_14
  simp only [← c98907, ← c98908, ← c98909] at e_1616_14
  have e_1616_15 := arithEq_of_rows h (row := 1616) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_15
  simp only [← c98910, ← c98911, k2, ← c98912] at e_1616_15
  have hr := e_1616_15
  simp only [e_1616_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1616 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16046)) := by
  have c98913 := (publicBatchWrapper86_copies3091 a h).2.1
  have c98914 := (publicBatchWrapper86_copies3091 a h).2.2.1
  have c98915 := (publicBatchWrapper86_copies3091 a h).2.2.2.1
  have c98916 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.1
  have c98917 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.1
  have c98918 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1616_16 := arithEq_of_rows h (row := 1616) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_16
  simp only [← c98913, ← c98914, ← c98915] at e_1616_16
  have e_1616_17 := arithEq_of_rows h (row := 1616) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_17
  simp only [← c98916, ← c98917, k2, ← c98918] at e_1616_17
  have hr := e_1616_17
  simp only [e_1616_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1616 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16047)) := by
  have c98919 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.1
  have c98920 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.1
  have c98921 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.1
  have c98922 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98923 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98924 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1616_18 := arithEq_of_rows h (row := 1616) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_18
  simp only [← c98919, ← c98920, ← c98921] at e_1616_18
  have e_1616_19 := arithEq_of_rows h (row := 1616) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_19
  simp only [← c98922, ← c98923, k2, ← c98924] at e_1616_19
  have hr := e_1616_19
  simp only [e_1616_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1617 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16048)) := by
  have c98925 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98926 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98927 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98928 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98929 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98930 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1617_0 := arithEq_of_rows h (row := 1617) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_0
  simp only [← c98925, ← c98926, ← c98927] at e_1617_0
  have e_1617_1 := arithEq_of_rows h (row := 1617) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_1
  simp only [← c98928, ← c98929, k2, ← c98930] at e_1617_1
  have hr := e_1617_1
  simp only [e_1617_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1617 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16049)) := by
  have c98931 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98932 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98933 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98934 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98935 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98936 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1617_2 := arithEq_of_rows h (row := 1617) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_2
  simp only [← c98931, ← c98932, ← c98933] at e_1617_2
  have e_1617_3 := arithEq_of_rows h (row := 1617) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_3
  simp only [← c98934, ← c98935, k2, ← c98936] at e_1617_3
  have hr := e_1617_3
  simp only [e_1617_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1617 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16050)) := by
  have c98937 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98938 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98939 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98940 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98941 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98942 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1617_4 := arithEq_of_rows h (row := 1617) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_4
  simp only [← c98937, ← c98938, ← c98939] at e_1617_4
  have e_1617_5 := arithEq_of_rows h (row := 1617) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_5
  simp only [← c98940, ← c98941, k2, ← c98942] at e_1617_5
  have hr := e_1617_5
  simp only [e_1617_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1617 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16051)) := by
  have c98943 := (publicBatchWrapper86_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98944 := (publicBatchWrapper86_copies3092 a h).1
  have c98945 := (publicBatchWrapper86_copies3092 a h).2.1
  have c98946 := (publicBatchWrapper86_copies3092 a h).2.2.1
  have c98947 := (publicBatchWrapper86_copies3092 a h).2.2.2.1
  have c98948 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1617_6 := arithEq_of_rows h (row := 1617) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_6
  simp only [← c98943, ← c98944, ← c98945] at e_1617_6
  have e_1617_7 := arithEq_of_rows h (row := 1617) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_7
  simp only [← c98946, ← c98947, k2, ← c98948] at e_1617_7
  have hr := e_1617_7
  simp only [e_1617_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1617 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16052)) := by
  have c98949 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.1
  have c98950 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.1
  have c98951 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.1
  have c98952 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.1
  have c98953 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.1
  have c98954 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1617_8 := arithEq_of_rows h (row := 1617) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_8
  simp only [← c98949, ← c98950, ← c98951] at e_1617_8
  have e_1617_9 := arithEq_of_rows h (row := 1617) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_9
  simp only [← c98952, ← c98953, k2, ← c98954] at e_1617_9
  have hr := e_1617_9
  simp only [e_1617_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1617 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16053)) := by
  have c98955 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98956 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98957 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98958 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98959 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98960 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1617_10 := arithEq_of_rows h (row := 1617) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_10
  simp only [← c98955, ← c98956, ← c98957] at e_1617_10
  have e_1617_11 := arithEq_of_rows h (row := 1617) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_11
  simp only [← c98958, ← c98959, k2, ← c98960] at e_1617_11
  have hr := e_1617_11
  simp only [e_1617_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1617 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16054)) := by
  have c98961 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98962 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98963 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98964 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98965 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98966 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1617_12 := arithEq_of_rows h (row := 1617) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_12
  simp only [← c98961, ← c98962, ← c98963] at e_1617_12
  have e_1617_13 := arithEq_of_rows h (row := 1617) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_13
  simp only [← c98964, ← c98965, k2, ← c98966] at e_1617_13
  have hr := e_1617_13
  simp only [e_1617_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1617 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16055)) := by
  have c98967 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98968 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98969 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98970 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98971 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98972 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1617_14 := arithEq_of_rows h (row := 1617) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_14
  simp only [← c98967, ← c98968, ← c98969] at e_1617_14
  have e_1617_15 := arithEq_of_rows h (row := 1617) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_15
  simp only [← c98970, ← c98971, k2, ← c98972] at e_1617_15
  have hr := e_1617_15
  simp only [e_1617_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1617 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16056)) := by
  have c98973 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98974 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98975 := (publicBatchWrapper86_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98976 := (publicBatchWrapper86_copies3093 a h).1
  have c98977 := (publicBatchWrapper86_copies3093 a h).2.1
  have c98978 := (publicBatchWrapper86_copies3093 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1617_16 := arithEq_of_rows h (row := 1617) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_16
  simp only [← c98973, ← c98974, ← c98975] at e_1617_16
  have e_1617_17 := arithEq_of_rows h (row := 1617) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_17
  simp only [← c98976, ← c98977, k2, ← c98978] at e_1617_17
  have hr := e_1617_17
  simp only [e_1617_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1617 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16057)) := by
  have c98979 := (publicBatchWrapper86_copies3093 a h).2.2.2.1
  have c98980 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.1
  have c98981 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.1
  have c98982 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.1
  have c98983 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.1
  have c98984 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1617_18 := arithEq_of_rows h (row := 1617) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_18
  simp only [← c98979, ← c98980, ← c98981] at e_1617_18
  have e_1617_19 := arithEq_of_rows h (row := 1617) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_19
  simp only [← c98982, ← c98983, k2, ← c98984] at e_1617_19
  have hr := e_1617_19
  simp only [e_1617_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1618 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16058)) := by
  have c98985 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.1
  have c98986 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98987 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98988 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98989 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98990 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1618_0 := arithEq_of_rows h (row := 1618) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_0
  simp only [← c98985, ← c98986, ← c98987] at e_1618_0
  have e_1618_1 := arithEq_of_rows h (row := 1618) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_1
  simp only [← c98988, ← c98989, k2, ← c98990] at e_1618_1
  have hr := e_1618_1
  simp only [e_1618_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1618 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16059)) := by
  have c98991 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98992 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98993 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98994 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98995 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98996 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1618_2 := arithEq_of_rows h (row := 1618) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_2
  simp only [← c98991, ← c98992, ← c98993] at e_1618_2
  have e_1618_3 := arithEq_of_rows h (row := 1618) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_3
  simp only [← c98994, ← c98995, k2, ← c98996] at e_1618_3
  have hr := e_1618_3
  simp only [e_1618_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1618 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16060)) := by
  have c98997 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98998 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98999 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99000 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99001 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99002 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1618_4 := arithEq_of_rows h (row := 1618) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_4
  simp only [← c98997, ← c98998, ← c98999] at e_1618_4
  have e_1618_5 := arithEq_of_rows h (row := 1618) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_5
  simp only [← c99000, ← c99001, k2, ← c99002] at e_1618_5
  have hr := e_1618_5
  simp only [e_1618_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1618 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16061)) := by
  have c99003 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99004 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99005 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99006 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99007 := (publicBatchWrapper86_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99008 := (publicBatchWrapper86_copies3094 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1618_6 := arithEq_of_rows h (row := 1618) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_6
  simp only [← c99003, ← c99004, ← c99005] at e_1618_6
  have e_1618_7 := arithEq_of_rows h (row := 1618) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_7
  simp only [← c99006, ← c99007, k2, ← c99008] at e_1618_7
  have hr := e_1618_7
  simp only [e_1618_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1618 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16062)) := by
  have c99009 := (publicBatchWrapper86_copies3094 a h).2.1
  have c99010 := (publicBatchWrapper86_copies3094 a h).2.2.1
  have c99011 := (publicBatchWrapper86_copies3094 a h).2.2.2.1
  have c99012 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.1
  have c99013 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.1
  have c99014 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1618_8 := arithEq_of_rows h (row := 1618) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_8
  simp only [← c99009, ← c99010, ← c99011] at e_1618_8
  have e_1618_9 := arithEq_of_rows h (row := 1618) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_9
  simp only [← c99012, ← c99013, k2, ← c99014] at e_1618_9
  have hr := e_1618_9
  simp only [e_1618_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1618 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16063)) := by
  have c99015 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.1
  have c99016 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.1
  have c99017 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.1
  have c99018 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99019 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99020 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1618_10 := arithEq_of_rows h (row := 1618) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_10
  simp only [← c99015, ← c99016, ← c99017] at e_1618_10
  have e_1618_11 := arithEq_of_rows h (row := 1618) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_11
  simp only [← c99018, ← c99019, k2, ← c99020] at e_1618_11
  have hr := e_1618_11
  simp only [e_1618_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1618 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16064)) := by
  have c99021 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99022 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99023 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99024 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99025 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99026 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1618_12 := arithEq_of_rows h (row := 1618) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_12
  simp only [← c99021, ← c99022, ← c99023] at e_1618_12
  have e_1618_13 := arithEq_of_rows h (row := 1618) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_13
  simp only [← c99024, ← c99025, k2, ← c99026] at e_1618_13
  have hr := e_1618_13
  simp only [e_1618_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1618 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16065)) := by
  have c99027 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99028 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99029 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99030 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99031 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99032 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1618_14 := arithEq_of_rows h (row := 1618) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_14
  simp only [← c99027, ← c99028, ← c99029] at e_1618_14
  have e_1618_15 := arithEq_of_rows h (row := 1618) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_15
  simp only [← c99030, ← c99031, k2, ← c99032] at e_1618_15
  have hr := e_1618_15
  simp only [e_1618_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1618 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16066)) := by
  have c99033 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99034 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99035 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99036 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99037 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99038 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1618_16 := arithEq_of_rows h (row := 1618) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_16
  simp only [← c99033, ← c99034, ← c99035] at e_1618_16
  have e_1618_17 := arithEq_of_rows h (row := 1618) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_17
  simp only [← c99036, ← c99037, k2, ← c99038] at e_1618_17
  have hr := e_1618_17
  simp only [e_1618_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15781 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1618 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16067)) := by
  have c99039 := (publicBatchWrapper86_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99040 := (publicBatchWrapper86_copies3095 a h).1
  have c99041 := (publicBatchWrapper86_copies3095 a h).2.1
  have c99042 := (publicBatchWrapper86_copies3095 a h).2.2.1
  have c99043 := (publicBatchWrapper86_copies3095 a h).2.2.2.1
  have c99044 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1618_18 := arithEq_of_rows h (row := 1618) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_18
  simp only [← c99039, ← c99040, ← c99041] at e_1618_18
  have e_1618_19 := arithEq_of_rows h (row := 1618) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_19
  simp only [← c99042, ← c99043, k2, ← c99044] at e_1618_19
  have hr := e_1618_19
  simp only [e_1618_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15782 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1619 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16068)) := by
  have c99045 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.1
  have c99046 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.1
  have c99047 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.1
  have c99048 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.1
  have c99049 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.1
  have c99050 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1619_0 := arithEq_of_rows h (row := 1619) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_0
  simp only [← c99045, ← c99046, ← c99047] at e_1619_0
  have e_1619_1 := arithEq_of_rows h (row := 1619) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_1
  simp only [← c99048, ← c99049, k2, ← c99050] at e_1619_1
  have hr := e_1619_1
  simp only [e_1619_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15783 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1619 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16069)) := by
  have c99051 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99052 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99053 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99054 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99055 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99056 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1619_2 := arithEq_of_rows h (row := 1619) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_2
  simp only [← c99051, ← c99052, ← c99053] at e_1619_2
  have e_1619_3 := arithEq_of_rows h (row := 1619) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_3
  simp only [← c99054, ← c99055, k2, ← c99056] at e_1619_3
  have hr := e_1619_3
  simp only [e_1619_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15784 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1619 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16070)) := by
  have c99057 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99058 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99059 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99060 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99061 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99062 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1619_4 := arithEq_of_rows h (row := 1619) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_4
  simp only [← c99057, ← c99058, ← c99059] at e_1619_4
  have e_1619_5 := arithEq_of_rows h (row := 1619) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_5
  simp only [← c99060, ← c99061, k2, ← c99062] at e_1619_5
  have hr := e_1619_5
  simp only [e_1619_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15785 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1619 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16071)) := by
  have c99063 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99064 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99065 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99066 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99067 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99068 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1619_6 := arithEq_of_rows h (row := 1619) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_6
  simp only [← c99063, ← c99064, ← c99065] at e_1619_6
  have e_1619_7 := arithEq_of_rows h (row := 1619) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_7
  simp only [← c99066, ← c99067, k2, ← c99068] at e_1619_7
  have hr := e_1619_7
  simp only [e_1619_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15786 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1619 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16072)) := by
  have c99069 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99070 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99071 := (publicBatchWrapper86_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99072 := (publicBatchWrapper86_copies3096 a h).1
  have c99073 := (publicBatchWrapper86_copies3096 a h).2.1
  have c99074 := (publicBatchWrapper86_copies3096 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1619_8 := arithEq_of_rows h (row := 1619) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_8
  simp only [← c99069, ← c99070, ← c99071] at e_1619_8
  have e_1619_9 := arithEq_of_rows h (row := 1619) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_9
  simp only [← c99072, ← c99073, k2, ← c99074] at e_1619_9
  have hr := e_1619_9
  simp only [e_1619_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15787 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1619 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16073)) := by
  have c99075 := (publicBatchWrapper86_copies3096 a h).2.2.2.1
  have c99076 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.1
  have c99077 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.1
  have c99078 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.1
  have c99079 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.1
  have c99080 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1619_10 := arithEq_of_rows h (row := 1619) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_10
  simp only [← c99075, ← c99076, ← c99077] at e_1619_10
  have e_1619_11 := arithEq_of_rows h (row := 1619) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_11
  simp only [← c99078, ← c99079, k2, ← c99080] at e_1619_11
  have hr := e_1619_11
  simp only [e_1619_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15788 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1619 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16074)) := by
  have c99081 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.1
  have c99082 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99083 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99084 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99085 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99086 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1619_12 := arithEq_of_rows h (row := 1619) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_12
  simp only [← c99081, ← c99082, ← c99083] at e_1619_12
  have e_1619_13 := arithEq_of_rows h (row := 1619) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_13
  simp only [← c99084, ← c99085, k2, ← c99086] at e_1619_13
  have hr := e_1619_13
  simp only [e_1619_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15789 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1619 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16075)) := by
  have c99087 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99088 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99089 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99090 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99091 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99092 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1619_14 := arithEq_of_rows h (row := 1619) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_14
  simp only [← c99087, ← c99088, ← c99089] at e_1619_14
  have e_1619_15 := arithEq_of_rows h (row := 1619) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_15
  simp only [← c99090, ← c99091, k2, ← c99092] at e_1619_15
  have hr := e_1619_15
  simp only [e_1619_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15790 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1619 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16076)) := by
  have c99093 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99094 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99095 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99096 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99097 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99098 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1619_16 := arithEq_of_rows h (row := 1619) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_16
  simp only [← c99093, ← c99094, ← c99095] at e_1619_16
  have e_1619_17 := arithEq_of_rows h (row := 1619) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_17
  simp only [← c99096, ← c99097, k2, ← c99098] at e_1619_17
  have hr := e_1619_17
  simp only [e_1619_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15791 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1619 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16077)) := by
  have c99099 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99100 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99101 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99102 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99103 := (publicBatchWrapper86_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99104 := (publicBatchWrapper86_copies3097 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1619_18 := arithEq_of_rows h (row := 1619) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_18
  simp only [← c99099, ← c99100, ← c99101] at e_1619_18
  have e_1619_19 := arithEq_of_rows h (row := 1619) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_19
  simp only [← c99102, ← c99103, k2, ← c99104] at e_1619_19
  have hr := e_1619_19
  simp only [e_1619_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15792 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1620 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16078)) := by
  have c99105 := (publicBatchWrapper86_copies3097 a h).2.1
  have c99106 := (publicBatchWrapper86_copies3097 a h).2.2.1
  have c99107 := (publicBatchWrapper86_copies3097 a h).2.2.2.1
  have c99108 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.1
  have c99109 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.1
  have c99110 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1620_0 := arithEq_of_rows h (row := 1620) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_0
  simp only [← c99105, ← c99106, ← c99107] at e_1620_0
  have e_1620_1 := arithEq_of_rows h (row := 1620) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_1
  simp only [← c99108, ← c99109, k2, ← c99110] at e_1620_1
  have hr := e_1620_1
  simp only [e_1620_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15793 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1620 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16079)) := by
  have c99111 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.1
  have c99112 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.1
  have c99113 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.1
  have c99114 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99115 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99116 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1620_2 := arithEq_of_rows h (row := 1620) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_2
  simp only [← c99111, ← c99112, ← c99113] at e_1620_2
  have e_1620_3 := arithEq_of_rows h (row := 1620) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_3
  simp only [← c99114, ← c99115, k2, ← c99116] at e_1620_3
  have hr := e_1620_3
  simp only [e_1620_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15794 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1620 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16080)) := by
  have c99117 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99118 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99119 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99120 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99121 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99122 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1620_4 := arithEq_of_rows h (row := 1620) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_4
  simp only [← c99117, ← c99118, ← c99119] at e_1620_4
  have e_1620_5 := arithEq_of_rows h (row := 1620) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_5
  simp only [← c99120, ← c99121, k2, ← c99122] at e_1620_5
  have hr := e_1620_5
  simp only [e_1620_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15795 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1620 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16081)) := by
  have c99123 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99124 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99125 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99126 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99127 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99128 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1620_6 := arithEq_of_rows h (row := 1620) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_6
  simp only [← c99123, ← c99124, ← c99125] at e_1620_6
  have e_1620_7 := arithEq_of_rows h (row := 1620) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_7
  simp only [← c99126, ← c99127, k2, ← c99128] at e_1620_7
  have hr := e_1620_7
  simp only [e_1620_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15796 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1620 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16082)) := by
  have c99129 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99130 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99131 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99132 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99133 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99134 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1620_8 := arithEq_of_rows h (row := 1620) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_8
  simp only [← c99129, ← c99130, ← c99131] at e_1620_8
  have e_1620_9 := arithEq_of_rows h (row := 1620) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_9
  simp only [← c99132, ← c99133, k2, ← c99134] at e_1620_9
  have hr := e_1620_9
  simp only [e_1620_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15797 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1620 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16083)) := by
  have c99135 := (publicBatchWrapper86_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99136 := (publicBatchWrapper86_copies3098 a h).1
  have c99137 := (publicBatchWrapper86_copies3098 a h).2.1
  have c99138 := (publicBatchWrapper86_copies3098 a h).2.2.1
  have c99139 := (publicBatchWrapper86_copies3098 a h).2.2.2.1
  have c99140 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1620_10 := arithEq_of_rows h (row := 1620) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_10
  simp only [← c99135, ← c99136, ← c99137] at e_1620_10
  have e_1620_11 := arithEq_of_rows h (row := 1620) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_11
  simp only [← c99138, ← c99139, k2, ← c99140] at e_1620_11
  have hr := e_1620_11
  simp only [e_1620_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15798 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1620 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16084)) := by
  have c99141 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.1
  have c99142 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.1
  have c99143 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.1
  have c99144 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.1
  have c99145 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.1
  have c99146 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1620_12 := arithEq_of_rows h (row := 1620) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_12
  simp only [← c99141, ← c99142, ← c99143] at e_1620_12
  have e_1620_13 := arithEq_of_rows h (row := 1620) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_13
  simp only [← c99144, ← c99145, k2, ← c99146] at e_1620_13
  have hr := e_1620_13
  simp only [e_1620_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15799 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1620 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16085)) := by
  have c99147 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99148 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99149 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99150 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99151 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99152 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1620_14 := arithEq_of_rows h (row := 1620) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_14
  simp only [← c99147, ← c99148, ← c99149] at e_1620_14
  have e_1620_15 := arithEq_of_rows h (row := 1620) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_15
  simp only [← c99150, ← c99151, k2, ← c99152] at e_1620_15
  have hr := e_1620_15
  simp only [e_1620_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1620 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16086)) := by
  have c99153 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99154 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99155 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99156 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99157 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99158 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1620_16 := arithEq_of_rows h (row := 1620) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_16
  simp only [← c99153, ← c99154, ← c99155] at e_1620_16
  have e_1620_17 := arithEq_of_rows h (row := 1620) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_17
  simp only [← c99156, ← c99157, k2, ← c99158] at e_1620_17
  have hr := e_1620_17
  simp only [e_1620_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1620 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16087)) := by
  have c99159 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99160 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99161 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99162 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99163 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99164 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1620_18 := arithEq_of_rows h (row := 1620) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_18
  simp only [← c99159, ← c99160, ← c99161] at e_1620_18
  have e_1620_19 := arithEq_of_rows h (row := 1620) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_19
  simp only [← c99162, ← c99163, k2, ← c99164] at e_1620_19
  have hr := e_1620_19
  simp only [e_1620_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1621 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16088)) := by
  have c99165 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99166 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99167 := (publicBatchWrapper86_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99168 := (publicBatchWrapper86_copies3099 a h).1
  have c99169 := (publicBatchWrapper86_copies3099 a h).2.1
  have c99170 := (publicBatchWrapper86_copies3099 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1621_0 := arithEq_of_rows h (row := 1621) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_0
  simp only [← c99165, ← c99166, ← c99167] at e_1621_0
  have e_1621_1 := arithEq_of_rows h (row := 1621) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_1
  simp only [← c99168, ← c99169, k2, ← c99170] at e_1621_1
  have hr := e_1621_1
  simp only [e_1621_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1621 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16089)) := by
  have c99171 := (publicBatchWrapper86_copies3099 a h).2.2.2.1
  have c99172 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.1
  have c99173 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.1
  have c99174 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.1
  have c99175 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.1
  have c99176 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1621_2 := arithEq_of_rows h (row := 1621) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_2
  simp only [← c99171, ← c99172, ← c99173] at e_1621_2
  have e_1621_3 := arithEq_of_rows h (row := 1621) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_3
  simp only [← c99174, ← c99175, k2, ← c99176] at e_1621_3
  have hr := e_1621_3
  simp only [e_1621_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1621 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16090)) := by
  have c99177 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.1
  have c99178 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99179 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99180 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99181 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99182 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1621_4 := arithEq_of_rows h (row := 1621) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_4
  simp only [← c99177, ← c99178, ← c99179] at e_1621_4
  have e_1621_5 := arithEq_of_rows h (row := 1621) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_5
  simp only [← c99180, ← c99181, k2, ← c99182] at e_1621_5
  have hr := e_1621_5
  simp only [e_1621_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1621 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16091)) := by
  have c99183 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99184 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99185 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99186 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99187 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99188 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1621_6 := arithEq_of_rows h (row := 1621) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_6
  simp only [← c99183, ← c99184, ← c99185] at e_1621_6
  have e_1621_7 := arithEq_of_rows h (row := 1621) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_7
  simp only [← c99186, ← c99187, k2, ← c99188] at e_1621_7
  have hr := e_1621_7
  simp only [e_1621_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1621 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16092)) := by
  have c99189 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99190 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99191 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99192 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99193 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99194 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1621_8 := arithEq_of_rows h (row := 1621) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_8
  simp only [← c99189, ← c99190, ← c99191] at e_1621_8
  have e_1621_9 := arithEq_of_rows h (row := 1621) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_9
  simp only [← c99192, ← c99193, k2, ← c99194] at e_1621_9
  have hr := e_1621_9
  simp only [e_1621_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1621 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16093)) := by
  have c99195 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99196 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99197 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99198 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99199 := (publicBatchWrapper86_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99200 := (publicBatchWrapper86_copies3100 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1621_10 := arithEq_of_rows h (row := 1621) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_10
  simp only [← c99195, ← c99196, ← c99197] at e_1621_10
  have e_1621_11 := arithEq_of_rows h (row := 1621) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_11
  simp only [← c99198, ← c99199, k2, ← c99200] at e_1621_11
  have hr := e_1621_11
  simp only [e_1621_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1621 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16094)) := by
  have c99201 := (publicBatchWrapper86_copies3100 a h).2.1
  have c99202 := (publicBatchWrapper86_copies3100 a h).2.2.1
  have c99203 := (publicBatchWrapper86_copies3100 a h).2.2.2.1
  have c99204 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.1
  have c99205 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.1
  have c99206 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1621_12 := arithEq_of_rows h (row := 1621) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_12
  simp only [← c99201, ← c99202, ← c99203] at e_1621_12
  have e_1621_13 := arithEq_of_rows h (row := 1621) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_13
  simp only [← c99204, ← c99205, k2, ← c99206] at e_1621_13
  have hr := e_1621_13
  simp only [e_1621_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1621 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16095)) := by
  have c99207 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.1
  have c99208 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.1
  have c99209 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.1
  have c99210 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99211 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99212 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1621_14 := arithEq_of_rows h (row := 1621) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_14
  simp only [← c99207, ← c99208, ← c99209] at e_1621_14
  have e_1621_15 := arithEq_of_rows h (row := 1621) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_15
  simp only [← c99210, ← c99211, k2, ← c99212] at e_1621_15
  have hr := e_1621_15
  simp only [e_1621_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1621 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16096)) := by
  have c99213 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99214 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99215 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99216 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99217 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99218 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1621_16 := arithEq_of_rows h (row := 1621) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_16
  simp only [← c99213, ← c99214, ← c99215] at e_1621_16
  have e_1621_17 := arithEq_of_rows h (row := 1621) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_17
  simp only [← c99216, ← c99217, k2, ← c99218] at e_1621_17
  have hr := e_1621_17
  simp only [e_1621_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1621 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16097)) := by
  have c99219 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99220 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99221 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99222 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99223 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99224 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1621_18 := arithEq_of_rows h (row := 1621) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_18
  simp only [← c99219, ← c99220, ← c99221] at e_1621_18
  have e_1621_19 := arithEq_of_rows h (row := 1621) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_19
  simp only [← c99222, ← c99223, k2, ← c99224] at e_1621_19
  have hr := e_1621_19
  simp only [e_1621_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1622 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16098)) := by
  have c99225 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99226 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99227 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99228 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99229 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99230 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1622_0 := arithEq_of_rows h (row := 1622) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_0
  simp only [← c99225, ← c99226, ← c99227] at e_1622_0
  have e_1622_1 := arithEq_of_rows h (row := 1622) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_1
  simp only [← c99228, ← c99229, k2, ← c99230] at e_1622_1
  have hr := e_1622_1
  simp only [e_1622_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1622 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16099)) := by
  have c99231 := (publicBatchWrapper86_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99232 := (publicBatchWrapper86_copies3101 a h).1
  have c99233 := (publicBatchWrapper86_copies3101 a h).2.1
  have c99234 := (publicBatchWrapper86_copies3101 a h).2.2.1
  have c99235 := (publicBatchWrapper86_copies3101 a h).2.2.2.1
  have c99236 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1622_2 := arithEq_of_rows h (row := 1622) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_2
  simp only [← c99231, ← c99232, ← c99233] at e_1622_2
  have e_1622_3 := arithEq_of_rows h (row := 1622) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_3
  simp only [← c99234, ← c99235, k2, ← c99236] at e_1622_3
  have hr := e_1622_3
  simp only [e_1622_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1622 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16100)) := by
  have c99237 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.1
  have c99238 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.1
  have c99239 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.1
  have c99240 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.1
  have c99241 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.1
  have c99242 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1622_4 := arithEq_of_rows h (row := 1622) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_4
  simp only [← c99237, ← c99238, ← c99239] at e_1622_4
  have e_1622_5 := arithEq_of_rows h (row := 1622) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_5
  simp only [← c99240, ← c99241, k2, ← c99242] at e_1622_5
  have hr := e_1622_5
  simp only [e_1622_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1622 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16101)) := by
  have c99243 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99244 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99245 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99246 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99247 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99248 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1622_6 := arithEq_of_rows h (row := 1622) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_6
  simp only [← c99243, ← c99244, ← c99245] at e_1622_6
  have e_1622_7 := arithEq_of_rows h (row := 1622) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_7
  simp only [← c99246, ← c99247, k2, ← c99248] at e_1622_7
  have hr := e_1622_7
  simp only [e_1622_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1622 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16102)) := by
  have c99249 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99250 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99251 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99252 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99253 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99254 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1622_8 := arithEq_of_rows h (row := 1622) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_8
  simp only [← c99249, ← c99250, ← c99251] at e_1622_8
  have e_1622_9 := arithEq_of_rows h (row := 1622) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_9
  simp only [← c99252, ← c99253, k2, ← c99254] at e_1622_9
  have hr := e_1622_9
  simp only [e_1622_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1622 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16103)) := by
  have c99255 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99256 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99257 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99258 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99259 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99260 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1622_10 := arithEq_of_rows h (row := 1622) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_10
  simp only [← c99255, ← c99256, ← c99257] at e_1622_10
  have e_1622_11 := arithEq_of_rows h (row := 1622) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_11
  simp only [← c99258, ← c99259, k2, ← c99260] at e_1622_11
  have hr := e_1622_11
  simp only [e_1622_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1622 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16104)) := by
  have c99261 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99262 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99263 := (publicBatchWrapper86_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99264 := (publicBatchWrapper86_copies3102 a h).1
  have c99265 := (publicBatchWrapper86_copies3102 a h).2.1
  have c99266 := (publicBatchWrapper86_copies3102 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1622_12 := arithEq_of_rows h (row := 1622) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_12
  simp only [← c99261, ← c99262, ← c99263] at e_1622_12
  have e_1622_13 := arithEq_of_rows h (row := 1622) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_13
  simp only [← c99264, ← c99265, k2, ← c99266] at e_1622_13
  have hr := e_1622_13
  simp only [e_1622_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1622 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16105)) := by
  have c99267 := (publicBatchWrapper86_copies3102 a h).2.2.2.1
  have c99268 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.1
  have c99269 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.1
  have c99270 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.1
  have c99271 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.1
  have c99272 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1622_14 := arithEq_of_rows h (row := 1622) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_14
  simp only [← c99267, ← c99268, ← c99269] at e_1622_14
  have e_1622_15 := arithEq_of_rows h (row := 1622) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_15
  simp only [← c99270, ← c99271, k2, ← c99272] at e_1622_15
  have hr := e_1622_15
  simp only [e_1622_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1622 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16106)) := by
  have c99273 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.1
  have c99274 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99275 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99276 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99277 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99278 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1622_16 := arithEq_of_rows h (row := 1622) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_16
  simp only [← c99273, ← c99274, ← c99275] at e_1622_16
  have e_1622_17 := arithEq_of_rows h (row := 1622) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_17
  simp only [← c99276, ← c99277, k2, ← c99278] at e_1622_17
  have hr := e_1622_17
  simp only [e_1622_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1622 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16107)) := by
  have c99279 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99280 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99281 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99282 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99283 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99284 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1622_18 := arithEq_of_rows h (row := 1622) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_18
  simp only [← c99279, ← c99280, ← c99281] at e_1622_18
  have e_1622_19 := arithEq_of_rows h (row := 1622) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_19
  simp only [← c99282, ← c99283, k2, ← c99284] at e_1622_19
  have hr := e_1622_19
  simp only [e_1622_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1623 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16108)) := by
  have c99285 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99286 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99287 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99288 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99289 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99290 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1623_0 := arithEq_of_rows h (row := 1623) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_0
  simp only [← c99285, ← c99286, ← c99287] at e_1623_0
  have e_1623_1 := arithEq_of_rows h (row := 1623) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_1
  simp only [← c99288, ← c99289, k2, ← c99290] at e_1623_1
  have hr := e_1623_1
  simp only [e_1623_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1623 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16109)) := by
  have c99291 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99292 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99293 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99294 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99295 := (publicBatchWrapper86_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99296 := (publicBatchWrapper86_copies3103 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1623_2 := arithEq_of_rows h (row := 1623) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_2
  simp only [← c99291, ← c99292, ← c99293] at e_1623_2
  have e_1623_3 := arithEq_of_rows h (row := 1623) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_3
  simp only [← c99294, ← c99295, k2, ← c99296] at e_1623_3
  have hr := e_1623_3
  simp only [e_1623_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1623 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16110)) := by
  have c99297 := (publicBatchWrapper86_copies3103 a h).2.1
  have c99298 := (publicBatchWrapper86_copies3103 a h).2.2.1
  have c99299 := (publicBatchWrapper86_copies3103 a h).2.2.2.1
  have c99300 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.1
  have c99301 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.1
  have c99302 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1623_4 := arithEq_of_rows h (row := 1623) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_4
  simp only [← c99297, ← c99298, ← c99299] at e_1623_4
  have e_1623_5 := arithEq_of_rows h (row := 1623) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_5
  simp only [← c99300, ← c99301, k2, ← c99302] at e_1623_5
  have hr := e_1623_5
  simp only [e_1623_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1623 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16111)) := by
  have c99303 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.1
  have c99304 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.1
  have c99305 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.1
  have c99306 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99307 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99308 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1623_6 := arithEq_of_rows h (row := 1623) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_6
  simp only [← c99303, ← c99304, ← c99305] at e_1623_6
  have e_1623_7 := arithEq_of_rows h (row := 1623) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_7
  simp only [← c99306, ← c99307, k2, ← c99308] at e_1623_7
  have hr := e_1623_7
  simp only [e_1623_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15826 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1623 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16112)) := by
  have c99309 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99310 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99311 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99312 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99313 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99314 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1623_8 := arithEq_of_rows h (row := 1623) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_8
  simp only [← c99309, ← c99310, ← c99311] at e_1623_8
  have e_1623_9 := arithEq_of_rows h (row := 1623) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_9
  simp only [← c99312, ← c99313, k2, ← c99314] at e_1623_9
  have hr := e_1623_9
  simp only [e_1623_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15827 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1623 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16113)) := by
  have c99315 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99316 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99317 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99318 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99319 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99320 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1623_10 := arithEq_of_rows h (row := 1623) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_10
  simp only [← c99315, ← c99316, ← c99317] at e_1623_10
  have e_1623_11 := arithEq_of_rows h (row := 1623) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_11
  simp only [← c99318, ← c99319, k2, ← c99320] at e_1623_11
  have hr := e_1623_11
  simp only [e_1623_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15828 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1623 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16114)) := by
  have c99321 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99322 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99323 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99324 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99325 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99326 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1623_12 := arithEq_of_rows h (row := 1623) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_12
  simp only [← c99321, ← c99322, ← c99323] at e_1623_12
  have e_1623_13 := arithEq_of_rows h (row := 1623) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_13
  simp only [← c99324, ← c99325, k2, ← c99326] at e_1623_13
  have hr := e_1623_13
  simp only [e_1623_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15829 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1623 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16115)) := by
  have c99327 := (publicBatchWrapper86_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99328 := (publicBatchWrapper86_copies3104 a h).1
  have c99329 := (publicBatchWrapper86_copies3104 a h).2.1
  have c99330 := (publicBatchWrapper86_copies3104 a h).2.2.1
  have c99331 := (publicBatchWrapper86_copies3104 a h).2.2.2.1
  have c99332 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1623_14 := arithEq_of_rows h (row := 1623) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_14
  simp only [← c99327, ← c99328, ← c99329] at e_1623_14
  have e_1623_15 := arithEq_of_rows h (row := 1623) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_15
  simp only [← c99330, ← c99331, k2, ← c99332] at e_1623_15
  have hr := e_1623_15
  simp only [e_1623_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15830 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1623 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16116)) := by
  have c99333 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.1
  have c99334 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.1
  have c99335 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.1
  have c99336 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.1
  have c99337 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.1
  have c99338 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1623_16 := arithEq_of_rows h (row := 1623) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_16
  simp only [← c99333, ← c99334, ← c99335] at e_1623_16
  have e_1623_17 := arithEq_of_rows h (row := 1623) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_17
  simp only [← c99336, ← c99337, k2, ← c99338] at e_1623_17
  have hr := e_1623_17
  simp only [e_1623_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15831 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1623 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16117)) := by
  have c99339 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99340 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99341 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99342 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99343 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99344 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1623_18 := arithEq_of_rows h (row := 1623) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_18
  simp only [← c99339, ← c99340, ← c99341] at e_1623_18
  have e_1623_19 := arithEq_of_rows h (row := 1623) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_19
  simp only [← c99342, ← c99343, k2, ← c99344] at e_1623_19
  have hr := e_1623_19
  simp only [e_1623_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15832 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1624 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16118)) := by
  have c99345 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99346 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99347 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99348 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99349 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99350 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1624_0 := arithEq_of_rows h (row := 1624) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_0
  simp only [← c99345, ← c99346, ← c99347] at e_1624_0
  have e_1624_1 := arithEq_of_rows h (row := 1624) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_1
  simp only [← c99348, ← c99349, k2, ← c99350] at e_1624_1
  have hr := e_1624_1
  simp only [e_1624_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15833 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1624 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16119)) := by
  have c99351 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99352 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99353 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99354 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99355 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99356 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1624_2 := arithEq_of_rows h (row := 1624) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_2
  simp only [← c99351, ← c99352, ← c99353] at e_1624_2
  have e_1624_3 := arithEq_of_rows h (row := 1624) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_3
  simp only [← c99354, ← c99355, k2, ← c99356] at e_1624_3
  have hr := e_1624_3
  simp only [e_1624_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15834 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1624 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16120)) := by
  have c99357 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99358 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99359 := (publicBatchWrapper86_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99360 := (publicBatchWrapper86_copies3105 a h).1
  have c99361 := (publicBatchWrapper86_copies3105 a h).2.1
  have c99362 := (publicBatchWrapper86_copies3105 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1624_4 := arithEq_of_rows h (row := 1624) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_4
  simp only [← c99357, ← c99358, ← c99359] at e_1624_4
  have e_1624_5 := arithEq_of_rows h (row := 1624) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_5
  simp only [← c99360, ← c99361, k2, ← c99362] at e_1624_5
  have hr := e_1624_5
  simp only [e_1624_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15835 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1624 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16121)) := by
  have c99363 := (publicBatchWrapper86_copies3105 a h).2.2.2.1
  have c99364 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.1
  have c99365 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.1
  have c99366 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.1
  have c99367 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.1
  have c99368 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1624_6 := arithEq_of_rows h (row := 1624) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_6
  simp only [← c99363, ← c99364, ← c99365] at e_1624_6
  have e_1624_7 := arithEq_of_rows h (row := 1624) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_7
  simp only [← c99366, ← c99367, k2, ← c99368] at e_1624_7
  have hr := e_1624_7
  simp only [e_1624_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15836 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1624 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16122)) := by
  have c99369 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.1
  have c99370 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99371 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99372 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99373 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99374 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1624_8 := arithEq_of_rows h (row := 1624) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_8
  simp only [← c99369, ← c99370, ← c99371] at e_1624_8
  have e_1624_9 := arithEq_of_rows h (row := 1624) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_9
  simp only [← c99372, ← c99373, k2, ← c99374] at e_1624_9
  have hr := e_1624_9
  simp only [e_1624_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15837 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1624 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16123)) := by
  have c99375 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99376 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99377 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99378 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99379 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99380 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1624_10 := arithEq_of_rows h (row := 1624) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_10
  simp only [← c99375, ← c99376, ← c99377] at e_1624_10
  have e_1624_11 := arithEq_of_rows h (row := 1624) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_11
  simp only [← c99378, ← c99379, k2, ← c99380] at e_1624_11
  have hr := e_1624_11
  simp only [e_1624_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15838 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1624 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16124)) := by
  have c99381 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99382 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99383 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99384 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99385 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99386 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1624_12 := arithEq_of_rows h (row := 1624) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_12
  simp only [← c99381, ← c99382, ← c99383] at e_1624_12
  have e_1624_13 := arithEq_of_rows h (row := 1624) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_13
  simp only [← c99384, ← c99385, k2, ← c99386] at e_1624_13
  have hr := e_1624_13
  simp only [e_1624_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15839 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1624 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16125)) := by
  have c99387 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99388 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99389 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99390 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99391 := (publicBatchWrapper86_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99392 := (publicBatchWrapper86_copies3106 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1624_14 := arithEq_of_rows h (row := 1624) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_14
  simp only [← c99387, ← c99388, ← c99389] at e_1624_14
  have e_1624_15 := arithEq_of_rows h (row := 1624) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_15
  simp only [← c99390, ← c99391, k2, ← c99392] at e_1624_15
  have hr := e_1624_15
  simp only [e_1624_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15840 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1624 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16126)) := by
  have c99393 := (publicBatchWrapper86_copies3106 a h).2.1
  have c99394 := (publicBatchWrapper86_copies3106 a h).2.2.1
  have c99395 := (publicBatchWrapper86_copies3106 a h).2.2.2.1
  have c99396 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.1
  have c99397 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.1
  have c99398 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1624_16 := arithEq_of_rows h (row := 1624) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_16
  simp only [← c99393, ← c99394, ← c99395] at e_1624_16
  have e_1624_17 := arithEq_of_rows h (row := 1624) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_17
  simp only [← c99396, ← c99397, k2, ← c99398] at e_1624_17
  have hr := e_1624_17
  simp only [e_1624_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15841 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1624 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16127)) := by
  have c99399 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.1
  have c99400 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.1
  have c99401 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.1
  have c99402 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99403 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99404 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1624_18 := arithEq_of_rows h (row := 1624) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_18
  simp only [← c99399, ← c99400, ← c99401] at e_1624_18
  have e_1624_19 := arithEq_of_rows h (row := 1624) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_19
  simp only [← c99402, ← c99403, k2, ← c99404] at e_1624_19
  have hr := e_1624_19
  simp only [e_1624_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15842 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1625 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16128)) := by
  have c99405 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99406 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99407 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99408 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99409 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99410 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1625_0 := arithEq_of_rows h (row := 1625) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_0
  simp only [← c99405, ← c99406, ← c99407] at e_1625_0
  have e_1625_1 := arithEq_of_rows h (row := 1625) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_1
  simp only [← c99408, ← c99409, k2, ← c99410] at e_1625_1
  have hr := e_1625_1
  simp only [e_1625_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15843 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1625 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16129)) := by
  have c99411 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99412 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99413 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99414 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99415 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99416 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1625_2 := arithEq_of_rows h (row := 1625) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_2
  simp only [← c99411, ← c99412, ← c99413] at e_1625_2
  have e_1625_3 := arithEq_of_rows h (row := 1625) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_3
  simp only [← c99414, ← c99415, k2, ← c99416] at e_1625_3
  have hr := e_1625_3
  simp only [e_1625_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15844 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1625 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16130)) := by
  have c99417 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99418 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99419 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99420 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99421 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99422 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1625_4 := arithEq_of_rows h (row := 1625) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_4
  simp only [← c99417, ← c99418, ← c99419] at e_1625_4
  have e_1625_5 := arithEq_of_rows h (row := 1625) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_5
  simp only [← c99420, ← c99421, k2, ← c99422] at e_1625_5
  have hr := e_1625_5
  simp only [e_1625_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15845 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1625 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16131)) := by
  have c99423 := (publicBatchWrapper86_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99424 := (publicBatchWrapper86_copies3107 a h).1
  have c99425 := (publicBatchWrapper86_copies3107 a h).2.1
  have c99426 := (publicBatchWrapper86_copies3107 a h).2.2.1
  have c99427 := (publicBatchWrapper86_copies3107 a h).2.2.2.1
  have c99428 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1625_6 := arithEq_of_rows h (row := 1625) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_6
  simp only [← c99423, ← c99424, ← c99425] at e_1625_6
  have e_1625_7 := arithEq_of_rows h (row := 1625) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_7
  simp only [← c99426, ← c99427, k2, ← c99428] at e_1625_7
  have hr := e_1625_7
  simp only [e_1625_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15846 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1625 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16132)) := by
  have c99429 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.1
  have c99430 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.1
  have c99431 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.1
  have c99432 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.1
  have c99433 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.1
  have c99434 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1625_8 := arithEq_of_rows h (row := 1625) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_8
  simp only [← c99429, ← c99430, ← c99431] at e_1625_8
  have e_1625_9 := arithEq_of_rows h (row := 1625) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_9
  simp only [← c99432, ← c99433, k2, ← c99434] at e_1625_9
  have hr := e_1625_9
  simp only [e_1625_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15847 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1625 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16133)) := by
  have c99435 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99436 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99437 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99438 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99439 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99440 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1625_10 := arithEq_of_rows h (row := 1625) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_10
  simp only [← c99435, ← c99436, ← c99437] at e_1625_10
  have e_1625_11 := arithEq_of_rows h (row := 1625) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_11
  simp only [← c99438, ← c99439, k2, ← c99440] at e_1625_11
  have hr := e_1625_11
  simp only [e_1625_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15848 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1625 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16134)) := by
  have c99441 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99442 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99443 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99444 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99445 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99446 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1625_12 := arithEq_of_rows h (row := 1625) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_12
  simp only [← c99441, ← c99442, ← c99443] at e_1625_12
  have e_1625_13 := arithEq_of_rows h (row := 1625) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_13
  simp only [← c99444, ← c99445, k2, ← c99446] at e_1625_13
  have hr := e_1625_13
  simp only [e_1625_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15849 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1625 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16135)) := by
  have c99447 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99448 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99449 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99450 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99451 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99452 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1625_14 := arithEq_of_rows h (row := 1625) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_14
  simp only [← c99447, ← c99448, ← c99449] at e_1625_14
  have e_1625_15 := arithEq_of_rows h (row := 1625) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_15
  simp only [← c99450, ← c99451, k2, ← c99452] at e_1625_15
  have hr := e_1625_15
  simp only [e_1625_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15850 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1625 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16136)) := by
  have c99453 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99454 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99455 := (publicBatchWrapper86_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99456 := (publicBatchWrapper86_copies3108 a h).1
  have c99457 := (publicBatchWrapper86_copies3108 a h).2.1
  have c99458 := (publicBatchWrapper86_copies3108 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1625_16 := arithEq_of_rows h (row := 1625) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_16
  simp only [← c99453, ← c99454, ← c99455] at e_1625_16
  have e_1625_17 := arithEq_of_rows h (row := 1625) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_17
  simp only [← c99456, ← c99457, k2, ← c99458] at e_1625_17
  have hr := e_1625_17
  simp only [e_1625_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15851 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1625 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16137)) := by
  have c99459 := (publicBatchWrapper86_copies3108 a h).2.2.2.1
  have c99460 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.1
  have c99461 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.1
  have c99462 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.1
  have c99463 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.1
  have c99464 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1625_18 := arithEq_of_rows h (row := 1625) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_18
  simp only [← c99459, ← c99460, ← c99461] at e_1625_18
  have e_1625_19 := arithEq_of_rows h (row := 1625) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_19
  simp only [← c99462, ← c99463, k2, ← c99464] at e_1625_19
  have hr := e_1625_19
  simp only [e_1625_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15852 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1626 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16138)) := by
  have c99465 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.1
  have c99466 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99467 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99468 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99469 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99470 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1626_0 := arithEq_of_rows h (row := 1626) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_0
  simp only [← c99465, ← c99466, ← c99467] at e_1626_0
  have e_1626_1 := arithEq_of_rows h (row := 1626) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_1
  simp only [← c99468, ← c99469, k2, ← c99470] at e_1626_1
  have hr := e_1626_1
  simp only [e_1626_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15853 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1626 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16139)) := by
  have c99471 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99472 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99473 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99474 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99475 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99476 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1626_2 := arithEq_of_rows h (row := 1626) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_2
  simp only [← c99471, ← c99472, ← c99473] at e_1626_2
  have e_1626_3 := arithEq_of_rows h (row := 1626) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_3
  simp only [← c99474, ← c99475, k2, ← c99476] at e_1626_3
  have hr := e_1626_3
  simp only [e_1626_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15854 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1626 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16140)) := by
  have c99477 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99478 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99479 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99480 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99481 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99482 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1626_4 := arithEq_of_rows h (row := 1626) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_4
  simp only [← c99477, ← c99478, ← c99479] at e_1626_4
  have e_1626_5 := arithEq_of_rows h (row := 1626) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_5
  simp only [← c99480, ← c99481, k2, ← c99482] at e_1626_5
  have hr := e_1626_5
  simp only [e_1626_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15855 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1626 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16141)) := by
  have c99483 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99484 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99485 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99486 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99487 := (publicBatchWrapper86_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99488 := (publicBatchWrapper86_copies3109 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1626_6 := arithEq_of_rows h (row := 1626) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_6
  simp only [← c99483, ← c99484, ← c99485] at e_1626_6
  have e_1626_7 := arithEq_of_rows h (row := 1626) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_7
  simp only [← c99486, ← c99487, k2, ← c99488] at e_1626_7
  have hr := e_1626_7
  simp only [e_1626_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15856 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1626 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16142)) := by
  have c99489 := (publicBatchWrapper86_copies3109 a h).2.1
  have c99490 := (publicBatchWrapper86_copies3109 a h).2.2.1
  have c99491 := (publicBatchWrapper86_copies3109 a h).2.2.2.1
  have c99492 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.1
  have c99493 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.1
  have c99494 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1626_8 := arithEq_of_rows h (row := 1626) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_8
  simp only [← c99489, ← c99490, ← c99491] at e_1626_8
  have e_1626_9 := arithEq_of_rows h (row := 1626) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_9
  simp only [← c99492, ← c99493, k2, ← c99494] at e_1626_9
  have hr := e_1626_9
  simp only [e_1626_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15857 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1626 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16143)) := by
  have c99495 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.1
  have c99496 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.1
  have c99497 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.1
  have c99498 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99499 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99500 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1626_10 := arithEq_of_rows h (row := 1626) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_10
  simp only [← c99495, ← c99496, ← c99497] at e_1626_10
  have e_1626_11 := arithEq_of_rows h (row := 1626) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_11
  simp only [← c99498, ← c99499, k2, ← c99500] at e_1626_11
  have hr := e_1626_11
  simp only [e_1626_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15858 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1626 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16144)) := by
  have c99501 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99502 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99503 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99504 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99505 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99506 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1626_12 := arithEq_of_rows h (row := 1626) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_12
  simp only [← c99501, ← c99502, ← c99503] at e_1626_12
  have e_1626_13 := arithEq_of_rows h (row := 1626) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_13
  simp only [← c99504, ← c99505, k2, ← c99506] at e_1626_13
  have hr := e_1626_13
  simp only [e_1626_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15859 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1626 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16145)) := by
  have c99507 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99508 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99509 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99510 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99511 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99512 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1626_14 := arithEq_of_rows h (row := 1626) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_14
  simp only [← c99507, ← c99508, ← c99509] at e_1626_14
  have e_1626_15 := arithEq_of_rows h (row := 1626) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_15
  simp only [← c99510, ← c99511, k2, ← c99512] at e_1626_15
  have hr := e_1626_15
  simp only [e_1626_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15860 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1626 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16146)) := by
  have c99513 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99514 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99515 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99516 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99517 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99518 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1626_16 := arithEq_of_rows h (row := 1626) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_16
  simp only [← c99513, ← c99514, ← c99515] at e_1626_16
  have e_1626_17 := arithEq_of_rows h (row := 1626) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_17
  simp only [← c99516, ← c99517, k2, ← c99518] at e_1626_17
  have hr := e_1626_17
  simp only [e_1626_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15861 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1626 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16147)) := by
  have c99519 := (publicBatchWrapper86_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99520 := (publicBatchWrapper86_copies3110 a h).1
  have c99521 := (publicBatchWrapper86_copies3110 a h).2.1
  have c99522 := (publicBatchWrapper86_copies3110 a h).2.2.1
  have c99523 := (publicBatchWrapper86_copies3110 a h).2.2.2.1
  have c99524 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1626_18 := arithEq_of_rows h (row := 1626) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_18
  simp only [← c99519, ← c99520, ← c99521] at e_1626_18
  have e_1626_19 := arithEq_of_rows h (row := 1626) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_19
  simp only [← c99522, ← c99523, k2, ← c99524] at e_1626_19
  have hr := e_1626_19
  simp only [e_1626_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15862 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1627 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16148)) := by
  have c99525 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.1
  have c99526 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.1
  have c99527 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.1
  have c99528 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.1
  have c99529 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.1
  have c99530 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1627_0 := arithEq_of_rows h (row := 1627) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_0
  simp only [← c99525, ← c99526, ← c99527] at e_1627_0
  have e_1627_1 := arithEq_of_rows h (row := 1627) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_1
  simp only [← c99528, ← c99529, k2, ← c99530] at e_1627_1
  have hr := e_1627_1
  simp only [e_1627_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15863 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1627 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16149)) := by
  have c99531 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99532 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99533 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99534 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99535 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99536 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1627_2 := arithEq_of_rows h (row := 1627) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_2
  simp only [← c99531, ← c99532, ← c99533] at e_1627_2
  have e_1627_3 := arithEq_of_rows h (row := 1627) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_3
  simp only [← c99534, ← c99535, k2, ← c99536] at e_1627_3
  have hr := e_1627_3
  simp only [e_1627_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15864 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1627 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16150)) := by
  have c99537 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99538 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99539 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99540 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99541 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99542 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1627_4 := arithEq_of_rows h (row := 1627) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_4
  simp only [← c99537, ← c99538, ← c99539] at e_1627_4
  have e_1627_5 := arithEq_of_rows h (row := 1627) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_5
  simp only [← c99540, ← c99541, k2, ← c99542] at e_1627_5
  have hr := e_1627_5
  simp only [e_1627_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15865 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1627 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16151)) := by
  have c99543 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99544 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99545 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99546 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99547 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99548 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1627_6 := arithEq_of_rows h (row := 1627) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_6
  simp only [← c99543, ← c99544, ← c99545] at e_1627_6
  have e_1627_7 := arithEq_of_rows h (row := 1627) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_7
  simp only [← c99546, ← c99547, k2, ← c99548] at e_1627_7
  have hr := e_1627_7
  simp only [e_1627_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15866 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1627 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16152)) := by
  have c99549 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99550 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99551 := (publicBatchWrapper86_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99552 := (publicBatchWrapper86_copies3111 a h).1
  have c99553 := (publicBatchWrapper86_copies3111 a h).2.1
  have c99554 := (publicBatchWrapper86_copies3111 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1627_8 := arithEq_of_rows h (row := 1627) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_8
  simp only [← c99549, ← c99550, ← c99551] at e_1627_8
  have e_1627_9 := arithEq_of_rows h (row := 1627) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_9
  simp only [← c99552, ← c99553, k2, ← c99554] at e_1627_9
  have hr := e_1627_9
  simp only [e_1627_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15867 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1627 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16153)) := by
  have c99555 := (publicBatchWrapper86_copies3111 a h).2.2.2.1
  have c99556 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.1
  have c99557 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.1
  have c99558 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.1
  have c99559 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.1
  have c99560 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1627_10 := arithEq_of_rows h (row := 1627) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_10
  simp only [← c99555, ← c99556, ← c99557] at e_1627_10
  have e_1627_11 := arithEq_of_rows h (row := 1627) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_11
  simp only [← c99558, ← c99559, k2, ← c99560] at e_1627_11
  have hr := e_1627_11
  simp only [e_1627_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15868 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1627 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16154)) := by
  have c99561 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.1
  have c99562 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99563 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99564 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99565 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99566 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1627_12 := arithEq_of_rows h (row := 1627) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_12
  simp only [← c99561, ← c99562, ← c99563] at e_1627_12
  have e_1627_13 := arithEq_of_rows h (row := 1627) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_13
  simp only [← c99564, ← c99565, k2, ← c99566] at e_1627_13
  have hr := e_1627_13
  simp only [e_1627_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15869 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1627 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16155)) := by
  have c99567 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99568 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99569 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99570 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99571 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99572 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1627_14 := arithEq_of_rows h (row := 1627) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_14
  simp only [← c99567, ← c99568, ← c99569] at e_1627_14
  have e_1627_15 := arithEq_of_rows h (row := 1627) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_15
  simp only [← c99570, ← c99571, k2, ← c99572] at e_1627_15
  have hr := e_1627_15
  simp only [e_1627_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1627 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16156)) := by
  have c99573 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99574 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99575 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99576 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99577 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99578 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1627_16 := arithEq_of_rows h (row := 1627) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_16
  simp only [← c99573, ← c99574, ← c99575] at e_1627_16
  have e_1627_17 := arithEq_of_rows h (row := 1627) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_17
  simp only [← c99576, ← c99577, k2, ← c99578] at e_1627_17
  have hr := e_1627_17
  simp only [e_1627_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1627 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16157)) := by
  have c99579 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99580 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99581 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99582 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99583 := (publicBatchWrapper86_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99584 := (publicBatchWrapper86_copies3112 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1627_18 := arithEq_of_rows h (row := 1627) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_18
  simp only [← c99579, ← c99580, ← c99581] at e_1627_18
  have e_1627_19 := arithEq_of_rows h (row := 1627) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_19
  simp only [← c99582, ← c99583, k2, ← c99584] at e_1627_19
  have hr := e_1627_19
  simp only [e_1627_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1628 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16158)) := by
  have c99585 := (publicBatchWrapper86_copies3112 a h).2.1
  have c99586 := (publicBatchWrapper86_copies3112 a h).2.2.1
  have c99587 := (publicBatchWrapper86_copies3112 a h).2.2.2.1
  have c99588 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.1
  have c99589 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.1
  have c99590 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1628_0 := arithEq_of_rows h (row := 1628) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_0
  simp only [← c99585, ← c99586, ← c99587] at e_1628_0
  have e_1628_1 := arithEq_of_rows h (row := 1628) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_1
  simp only [← c99588, ← c99589, k2, ← c99590] at e_1628_1
  have hr := e_1628_1
  simp only [e_1628_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1628 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16159)) := by
  have c99591 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.1
  have c99592 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.1
  have c99593 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.1
  have c99594 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99595 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99596 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1628_2 := arithEq_of_rows h (row := 1628) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_2
  simp only [← c99591, ← c99592, ← c99593] at e_1628_2
  have e_1628_3 := arithEq_of_rows h (row := 1628) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_3
  simp only [← c99594, ← c99595, k2, ← c99596] at e_1628_3
  have hr := e_1628_3
  simp only [e_1628_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1628 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16160)) := by
  have c99597 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99598 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99599 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99600 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99601 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99602 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1628_4 := arithEq_of_rows h (row := 1628) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_4
  simp only [← c99597, ← c99598, ← c99599] at e_1628_4
  have e_1628_5 := arithEq_of_rows h (row := 1628) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_5
  simp only [← c99600, ← c99601, k2, ← c99602] at e_1628_5
  have hr := e_1628_5
  simp only [e_1628_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1628 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16161)) := by
  have c99603 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99604 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99605 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99606 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99607 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99608 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1628_6 := arithEq_of_rows h (row := 1628) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_6
  simp only [← c99603, ← c99604, ← c99605] at e_1628_6
  have e_1628_7 := arithEq_of_rows h (row := 1628) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_7
  simp only [← c99606, ← c99607, k2, ← c99608] at e_1628_7
  have hr := e_1628_7
  simp only [e_1628_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1628 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16162)) := by
  have c99609 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99610 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99611 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99612 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99613 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99614 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1628_8 := arithEq_of_rows h (row := 1628) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_8
  simp only [← c99609, ← c99610, ← c99611] at e_1628_8
  have e_1628_9 := arithEq_of_rows h (row := 1628) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_9
  simp only [← c99612, ← c99613, k2, ← c99614] at e_1628_9
  have hr := e_1628_9
  simp only [e_1628_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15877 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1628 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16163)) := by
  have c99615 := (publicBatchWrapper86_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99616 := (publicBatchWrapper86_copies3113 a h).1
  have c99617 := (publicBatchWrapper86_copies3113 a h).2.1
  have c99618 := (publicBatchWrapper86_copies3113 a h).2.2.1
  have c99619 := (publicBatchWrapper86_copies3113 a h).2.2.2.1
  have c99620 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1628_10 := arithEq_of_rows h (row := 1628) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_10
  simp only [← c99615, ← c99616, ← c99617] at e_1628_10
  have e_1628_11 := arithEq_of_rows h (row := 1628) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_11
  simp only [← c99618, ← c99619, k2, ← c99620] at e_1628_11
  have hr := e_1628_11
  simp only [e_1628_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15878 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1628 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16164)) := by
  have c99621 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.1
  have c99622 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.1
  have c99623 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.1
  have c99624 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.1
  have c99625 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.1
  have c99626 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1628_12 := arithEq_of_rows h (row := 1628) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_12
  simp only [← c99621, ← c99622, ← c99623] at e_1628_12
  have e_1628_13 := arithEq_of_rows h (row := 1628) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_13
  simp only [← c99624, ← c99625, k2, ← c99626] at e_1628_13
  have hr := e_1628_13
  simp only [e_1628_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15879 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1628 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16165)) := by
  have c99627 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99628 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99629 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99630 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99631 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99632 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1628_14 := arithEq_of_rows h (row := 1628) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_14
  simp only [← c99627, ← c99628, ← c99629] at e_1628_14
  have e_1628_15 := arithEq_of_rows h (row := 1628) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_15
  simp only [← c99630, ← c99631, k2, ← c99632] at e_1628_15
  have hr := e_1628_15
  simp only [e_1628_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15880 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1628 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16166)) := by
  have c99633 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99634 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99635 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99636 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99637 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99638 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1628_16 := arithEq_of_rows h (row := 1628) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_16
  simp only [← c99633, ← c99634, ← c99635] at e_1628_16
  have e_1628_17 := arithEq_of_rows h (row := 1628) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_17
  simp only [← c99636, ← c99637, k2, ← c99638] at e_1628_17
  have hr := e_1628_17
  simp only [e_1628_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15881 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1628 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16167)) := by
  have c99639 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99640 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99641 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99642 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99643 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99644 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1628_18 := arithEq_of_rows h (row := 1628) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_18
  simp only [← c99639, ← c99640, ← c99641] at e_1628_18
  have e_1628_19 := arithEq_of_rows h (row := 1628) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_19
  simp only [← c99642, ← c99643, k2, ← c99644] at e_1628_19
  have hr := e_1628_19
  simp only [e_1628_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15882 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1629 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16168)) := by
  have c99645 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99646 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99647 := (publicBatchWrapper86_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99648 := (publicBatchWrapper86_copies3114 a h).1
  have c99649 := (publicBatchWrapper86_copies3114 a h).2.1
  have c99650 := (publicBatchWrapper86_copies3114 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1629_0 := arithEq_of_rows h (row := 1629) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_0
  simp only [← c99645, ← c99646, ← c99647] at e_1629_0
  have e_1629_1 := arithEq_of_rows h (row := 1629) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_1
  simp only [← c99648, ← c99649, k2, ← c99650] at e_1629_1
  have hr := e_1629_1
  simp only [e_1629_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15883 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1629 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16169)) := by
  have c99651 := (publicBatchWrapper86_copies3114 a h).2.2.2.1
  have c99652 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.1
  have c99653 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.1
  have c99654 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.1
  have c99655 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.1
  have c99656 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1629_2 := arithEq_of_rows h (row := 1629) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_2
  simp only [← c99651, ← c99652, ← c99653] at e_1629_2
  have e_1629_3 := arithEq_of_rows h (row := 1629) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_3
  simp only [← c99654, ← c99655, k2, ← c99656] at e_1629_3
  have hr := e_1629_3
  simp only [e_1629_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15884 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1629 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16210)) := by
  have c99657 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.1
  have c99658 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99659 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99660 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99661 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99662 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1629_4 := arithEq_of_rows h (row := 1629) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_4
  simp only [← c99657, ← c99658, ← c99659] at e_1629_4
  have e_1629_5 := arithEq_of_rows h (row := 1629) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_5
  simp only [← c99660, ← c99661, k2, ← c99662] at e_1629_5
  have hr := e_1629_5
  simp only [e_1629_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15885 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1629 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16211)) := by
  have c99663 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99664 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99665 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99666 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99667 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99668 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1629_6 := arithEq_of_rows h (row := 1629) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_6
  simp only [← c99663, ← c99664, ← c99665] at e_1629_6
  have e_1629_7 := arithEq_of_rows h (row := 1629) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_7
  simp only [← c99666, ← c99667, k2, ← c99668] at e_1629_7
  have hr := e_1629_7
  simp only [e_1629_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15886 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1629 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16212)) := by
  have c99669 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99670 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99671 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99672 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99673 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99674 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1629_8 := arithEq_of_rows h (row := 1629) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_8
  simp only [← c99669, ← c99670, ← c99671] at e_1629_8
  have e_1629_9 := arithEq_of_rows h (row := 1629) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_9
  simp only [← c99672, ← c99673, k2, ← c99674] at e_1629_9
  have hr := e_1629_9
  simp only [e_1629_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15887 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1629 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16213)) := by
  have c99675 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99676 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99677 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99678 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99679 := (publicBatchWrapper86_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99680 := (publicBatchWrapper86_copies3115 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1629_10 := arithEq_of_rows h (row := 1629) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_10
  simp only [← c99675, ← c99676, ← c99677] at e_1629_10
  have e_1629_11 := arithEq_of_rows h (row := 1629) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_11
  simp only [← c99678, ← c99679, k2, ← c99680] at e_1629_11
  have hr := e_1629_11
  simp only [e_1629_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15888 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1629 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16214)) := by
  have c99681 := (publicBatchWrapper86_copies3115 a h).2.1
  have c99682 := (publicBatchWrapper86_copies3115 a h).2.2.1
  have c99683 := (publicBatchWrapper86_copies3115 a h).2.2.2.1
  have c99684 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.1
  have c99685 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.1
  have c99686 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1629_12 := arithEq_of_rows h (row := 1629) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_12
  simp only [← c99681, ← c99682, ← c99683] at e_1629_12
  have e_1629_13 := arithEq_of_rows h (row := 1629) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_13
  simp only [← c99684, ← c99685, k2, ← c99686] at e_1629_13
  have hr := e_1629_13
  simp only [e_1629_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15889 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1629 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16215)) := by
  have c99687 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.1
  have c99688 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.1
  have c99689 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.1
  have c99690 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99691 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99692 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1629_14 := arithEq_of_rows h (row := 1629) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_14
  simp only [← c99687, ← c99688, ← c99689] at e_1629_14
  have e_1629_15 := arithEq_of_rows h (row := 1629) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_15
  simp only [← c99690, ← c99691, k2, ← c99692] at e_1629_15
  have hr := e_1629_15
  simp only [e_1629_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15890 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1629 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16216)) := by
  have c99693 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99694 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99695 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99696 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99697 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99698 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1629_16 := arithEq_of_rows h (row := 1629) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_16
  simp only [← c99693, ← c99694, ← c99695] at e_1629_16
  have e_1629_17 := arithEq_of_rows h (row := 1629) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_17
  simp only [← c99696, ← c99697, k2, ← c99698] at e_1629_17
  have hr := e_1629_17
  simp only [e_1629_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15891 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1629 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16217)) := by
  have c99699 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99700 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99701 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99702 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99703 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99704 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1629_18 := arithEq_of_rows h (row := 1629) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_18
  simp only [← c99699, ← c99700, ← c99701] at e_1629_18
  have e_1629_19 := arithEq_of_rows h (row := 1629) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_19
  simp only [← c99702, ← c99703, k2, ← c99704] at e_1629_19
  have hr := e_1629_19
  simp only [e_1629_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15892 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1630 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16218)) := by
  have c99705 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99706 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99707 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99708 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99709 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99710 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1630_0 := arithEq_of_rows h (row := 1630) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_0
  simp only [← c99705, ← c99706, ← c99707] at e_1630_0
  have e_1630_1 := arithEq_of_rows h (row := 1630) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_1
  simp only [← c99708, ← c99709, k2, ← c99710] at e_1630_1
  have hr := e_1630_1
  simp only [e_1630_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15893 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1630 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16219)) := by
  have c99711 := (publicBatchWrapper86_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99712 := (publicBatchWrapper86_copies3116 a h).1
  have c99713 := (publicBatchWrapper86_copies3116 a h).2.1
  have c99714 := (publicBatchWrapper86_copies3116 a h).2.2.1
  have c99715 := (publicBatchWrapper86_copies3116 a h).2.2.2.1
  have c99716 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1630_2 := arithEq_of_rows h (row := 1630) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_2
  simp only [← c99711, ← c99712, ← c99713] at e_1630_2
  have e_1630_3 := arithEq_of_rows h (row := 1630) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_3
  simp only [← c99714, ← c99715, k2, ← c99716] at e_1630_3
  have hr := e_1630_3
  simp only [e_1630_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15894 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1630 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16220)) := by
  have c99717 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.1
  have c99718 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.1
  have c99719 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.1
  have c99720 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.1
  have c99721 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.1
  have c99722 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1630_4 := arithEq_of_rows h (row := 1630) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_4
  simp only [← c99717, ← c99718, ← c99719] at e_1630_4
  have e_1630_5 := arithEq_of_rows h (row := 1630) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_5
  simp only [← c99720, ← c99721, k2, ← c99722] at e_1630_5
  have hr := e_1630_5
  simp only [e_1630_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15895 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1630 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16221)) := by
  have c99723 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99724 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99725 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99726 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99727 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99728 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1630_6 := arithEq_of_rows h (row := 1630) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_6
  simp only [← c99723, ← c99724, ← c99725] at e_1630_6
  have e_1630_7 := arithEq_of_rows h (row := 1630) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_7
  simp only [← c99726, ← c99727, k2, ← c99728] at e_1630_7
  have hr := e_1630_7
  simp only [e_1630_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15896 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1630 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16222)) := by
  have c99729 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99730 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99731 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99732 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99733 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99734 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1630_8 := arithEq_of_rows h (row := 1630) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_8
  simp only [← c99729, ← c99730, ← c99731] at e_1630_8
  have e_1630_9 := arithEq_of_rows h (row := 1630) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_9
  simp only [← c99732, ← c99733, k2, ← c99734] at e_1630_9
  have hr := e_1630_9
  simp only [e_1630_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15897 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1630 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16223)) := by
  have c99735 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99736 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99737 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99738 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99739 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99740 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1630_10 := arithEq_of_rows h (row := 1630) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_10
  simp only [← c99735, ← c99736, ← c99737] at e_1630_10
  have e_1630_11 := arithEq_of_rows h (row := 1630) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_11
  simp only [← c99738, ← c99739, k2, ← c99740] at e_1630_11
  have hr := e_1630_11
  simp only [e_1630_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15898 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1630 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16224)) := by
  have c99741 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99742 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99743 := (publicBatchWrapper86_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99744 := (publicBatchWrapper86_copies3117 a h).1
  have c99745 := (publicBatchWrapper86_copies3117 a h).2.1
  have c99746 := (publicBatchWrapper86_copies3117 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1630_12 := arithEq_of_rows h (row := 1630) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_12
  simp only [← c99741, ← c99742, ← c99743] at e_1630_12
  have e_1630_13 := arithEq_of_rows h (row := 1630) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_13
  simp only [← c99744, ← c99745, k2, ← c99746] at e_1630_13
  have hr := e_1630_13
  simp only [e_1630_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15899 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1630 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16225)) := by
  have c99747 := (publicBatchWrapper86_copies3117 a h).2.2.2.1
  have c99748 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.1
  have c99749 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.1
  have c99750 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.1
  have c99751 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.1
  have c99752 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1630_14 := arithEq_of_rows h (row := 1630) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_14
  simp only [← c99747, ← c99748, ← c99749] at e_1630_14
  have e_1630_15 := arithEq_of_rows h (row := 1630) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_15
  simp only [← c99750, ← c99751, k2, ← c99752] at e_1630_15
  have hr := e_1630_15
  simp only [e_1630_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15900 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1630 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16226)) := by
  have c99753 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.1
  have c99754 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99755 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99756 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99757 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99758 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1630_16 := arithEq_of_rows h (row := 1630) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_16
  simp only [← c99753, ← c99754, ← c99755] at e_1630_16
  have e_1630_17 := arithEq_of_rows h (row := 1630) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_17
  simp only [← c99756, ← c99757, k2, ← c99758] at e_1630_17
  have hr := e_1630_17
  simp only [e_1630_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15901 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1630 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16227)) := by
  have c99759 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99760 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99761 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99762 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99763 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99764 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1630_18 := arithEq_of_rows h (row := 1630) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_18
  simp only [← c99759, ← c99760, ← c99761] at e_1630_18
  have e_1630_19 := arithEq_of_rows h (row := 1630) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_19
  simp only [← c99762, ← c99763, k2, ← c99764] at e_1630_19
  have hr := e_1630_19
  simp only [e_1630_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15902 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1631 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16228)) := by
  have c99765 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99766 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99767 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99768 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99769 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99770 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1631_0 := arithEq_of_rows h (row := 1631) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_0
  simp only [← c99765, ← c99766, ← c99767] at e_1631_0
  have e_1631_1 := arithEq_of_rows h (row := 1631) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_1
  simp only [← c99768, ← c99769, k2, ← c99770] at e_1631_1
  have hr := e_1631_1
  simp only [e_1631_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15903 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1631 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16229)) := by
  have c99771 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99772 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99773 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99774 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99775 := (publicBatchWrapper86_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99776 := (publicBatchWrapper86_copies3118 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1631_2 := arithEq_of_rows h (row := 1631) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_2
  simp only [← c99771, ← c99772, ← c99773] at e_1631_2
  have e_1631_3 := arithEq_of_rows h (row := 1631) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_3
  simp only [← c99774, ← c99775, k2, ← c99776] at e_1631_3
  have hr := e_1631_3
  simp only [e_1631_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15904 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1631 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16230)) := by
  have c99777 := (publicBatchWrapper86_copies3118 a h).2.1
  have c99778 := (publicBatchWrapper86_copies3118 a h).2.2.1
  have c99779 := (publicBatchWrapper86_copies3118 a h).2.2.2.1
  have c99780 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.1
  have c99781 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.1
  have c99782 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1631_4 := arithEq_of_rows h (row := 1631) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_4
  simp only [← c99777, ← c99778, ← c99779] at e_1631_4
  have e_1631_5 := arithEq_of_rows h (row := 1631) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_5
  simp only [← c99780, ← c99781, k2, ← c99782] at e_1631_5
  have hr := e_1631_5
  simp only [e_1631_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15905 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1631 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16231)) := by
  have c99783 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.1
  have c99784 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.1
  have c99785 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.1
  have c99786 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99787 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99788 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1631_6 := arithEq_of_rows h (row := 1631) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_6
  simp only [← c99783, ← c99784, ← c99785] at e_1631_6
  have e_1631_7 := arithEq_of_rows h (row := 1631) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_7
  simp only [← c99786, ← c99787, k2, ← c99788] at e_1631_7
  have hr := e_1631_7
  simp only [e_1631_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15906 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1631 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16232)) := by
  have c99789 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99790 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99791 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99792 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99793 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99794 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1631_8 := arithEq_of_rows h (row := 1631) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_8
  simp only [← c99789, ← c99790, ← c99791] at e_1631_8
  have e_1631_9 := arithEq_of_rows h (row := 1631) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_9
  simp only [← c99792, ← c99793, k2, ← c99794] at e_1631_9
  have hr := e_1631_9
  simp only [e_1631_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15907 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1631 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16233)) := by
  have c99795 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99796 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99797 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99798 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99799 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99800 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1631_10 := arithEq_of_rows h (row := 1631) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_10
  simp only [← c99795, ← c99796, ← c99797] at e_1631_10
  have e_1631_11 := arithEq_of_rows h (row := 1631) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_11
  simp only [← c99798, ← c99799, k2, ← c99800] at e_1631_11
  have hr := e_1631_11
  simp only [e_1631_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15908 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1631 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16234)) := by
  have c99801 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99802 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99803 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99804 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99805 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99806 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1631_12 := arithEq_of_rows h (row := 1631) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_12
  simp only [← c99801, ← c99802, ← c99803] at e_1631_12
  have e_1631_13 := arithEq_of_rows h (row := 1631) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_13
  simp only [← c99804, ← c99805, k2, ← c99806] at e_1631_13
  have hr := e_1631_13
  simp only [e_1631_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15909 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1631 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16235)) := by
  have c99807 := (publicBatchWrapper86_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99808 := (publicBatchWrapper86_copies3119 a h).1
  have c99809 := (publicBatchWrapper86_copies3119 a h).2.1
  have c99810 := (publicBatchWrapper86_copies3119 a h).2.2.1
  have c99811 := (publicBatchWrapper86_copies3119 a h).2.2.2.1
  have c99812 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1631_14 := arithEq_of_rows h (row := 1631) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_14
  simp only [← c99807, ← c99808, ← c99809] at e_1631_14
  have e_1631_15 := arithEq_of_rows h (row := 1631) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_15
  simp only [← c99810, ← c99811, k2, ← c99812] at e_1631_15
  have hr := e_1631_15
  simp only [e_1631_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15910 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1631 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16236)) := by
  have c99813 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.1
  have c99814 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.1
  have c99815 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.1
  have c99816 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.1
  have c99817 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.1
  have c99818 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1631_16 := arithEq_of_rows h (row := 1631) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_16
  simp only [← c99813, ← c99814, ← c99815] at e_1631_16
  have e_1631_17 := arithEq_of_rows h (row := 1631) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_17
  simp only [← c99816, ← c99817, k2, ← c99818] at e_1631_17
  have hr := e_1631_17
  simp only [e_1631_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15911 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1631 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16237)) := by
  have c99819 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99820 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99821 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99822 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99823 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99824 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1631_18 := arithEq_of_rows h (row := 1631) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_18
  simp only [← c99819, ← c99820, ← c99821] at e_1631_18
  have e_1631_19 := arithEq_of_rows h (row := 1631) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_19
  simp only [← c99822, ← c99823, k2, ← c99824] at e_1631_19
  have hr := e_1631_19
  simp only [e_1631_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15912 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1632 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16238)) := by
  have c99825 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99826 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99827 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99828 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99829 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99830 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1632_0 := arithEq_of_rows h (row := 1632) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_0
  simp only [← c99825, ← c99826, ← c99827] at e_1632_0
  have e_1632_1 := arithEq_of_rows h (row := 1632) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_1
  simp only [← c99828, ← c99829, k2, ← c99830] at e_1632_1
  have hr := e_1632_1
  simp only [e_1632_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15913 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1632 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16239)) := by
  have c99831 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99832 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99833 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99834 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99835 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99836 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1632_2 := arithEq_of_rows h (row := 1632) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_2
  simp only [← c99831, ← c99832, ← c99833] at e_1632_2
  have e_1632_3 := arithEq_of_rows h (row := 1632) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_3
  simp only [← c99834, ← c99835, k2, ← c99836] at e_1632_3
  have hr := e_1632_3
  simp only [e_1632_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15914 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1632 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16240)) := by
  have c99837 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99838 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99839 := (publicBatchWrapper86_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99840 := (publicBatchWrapper86_copies3120 a h).1
  have c99841 := (publicBatchWrapper86_copies3120 a h).2.1
  have c99842 := (publicBatchWrapper86_copies3120 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1632_4 := arithEq_of_rows h (row := 1632) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_4
  simp only [← c99837, ← c99838, ← c99839] at e_1632_4
  have e_1632_5 := arithEq_of_rows h (row := 1632) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_5
  simp only [← c99840, ← c99841, k2, ← c99842] at e_1632_5
  have hr := e_1632_5
  simp only [e_1632_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15915 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1632 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16241)) := by
  have c99843 := (publicBatchWrapper86_copies3120 a h).2.2.2.1
  have c99844 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.1
  have c99845 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.1
  have c99846 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.1
  have c99847 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.1
  have c99848 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1632_6 := arithEq_of_rows h (row := 1632) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_6
  simp only [← c99843, ← c99844, ← c99845] at e_1632_6
  have e_1632_7 := arithEq_of_rows h (row := 1632) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_7
  simp only [← c99846, ← c99847, k2, ← c99848] at e_1632_7
  have hr := e_1632_7
  simp only [e_1632_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15916 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1632 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16242)) := by
  have c99849 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.1
  have c99850 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99851 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99852 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99853 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99854 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1632_8 := arithEq_of_rows h (row := 1632) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_8
  simp only [← c99849, ← c99850, ← c99851] at e_1632_8
  have e_1632_9 := arithEq_of_rows h (row := 1632) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_9
  simp only [← c99852, ← c99853, k2, ← c99854] at e_1632_9
  have hr := e_1632_9
  simp only [e_1632_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15917 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1632 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16243)) := by
  have c99855 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99856 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99857 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99858 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99859 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99860 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1632_10 := arithEq_of_rows h (row := 1632) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_10
  simp only [← c99855, ← c99856, ← c99857] at e_1632_10
  have e_1632_11 := arithEq_of_rows h (row := 1632) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_11
  simp only [← c99858, ← c99859, k2, ← c99860] at e_1632_11
  have hr := e_1632_11
  simp only [e_1632_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15918 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1632 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16244)) := by
  have c99861 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99862 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99863 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99864 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99865 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99866 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1632_12 := arithEq_of_rows h (row := 1632) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_12
  simp only [← c99861, ← c99862, ← c99863] at e_1632_12
  have e_1632_13 := arithEq_of_rows h (row := 1632) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_13
  simp only [← c99864, ← c99865, k2, ← c99866] at e_1632_13
  have hr := e_1632_13
  simp only [e_1632_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15919 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1632 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16245)) := by
  have c99867 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99868 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99869 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99870 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99871 := (publicBatchWrapper86_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99872 := (publicBatchWrapper86_copies3121 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1632_14 := arithEq_of_rows h (row := 1632) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_14
  simp only [← c99867, ← c99868, ← c99869] at e_1632_14
  have e_1632_15 := arithEq_of_rows h (row := 1632) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_15
  simp only [← c99870, ← c99871, k2, ← c99872] at e_1632_15
  have hr := e_1632_15
  simp only [e_1632_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15920 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1632 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16246)) := by
  have c99873 := (publicBatchWrapper86_copies3121 a h).2.1
  have c99874 := (publicBatchWrapper86_copies3121 a h).2.2.1
  have c99875 := (publicBatchWrapper86_copies3121 a h).2.2.2.1
  have c99876 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.1
  have c99877 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.1
  have c99878 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1632_16 := arithEq_of_rows h (row := 1632) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_16
  simp only [← c99873, ← c99874, ← c99875] at e_1632_16
  have e_1632_17 := arithEq_of_rows h (row := 1632) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_17
  simp only [← c99876, ← c99877, k2, ← c99878] at e_1632_17
  have hr := e_1632_17
  simp only [e_1632_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15921 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1632 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16247)) := by
  have c99879 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.1
  have c99880 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.1
  have c99881 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.1
  have c99882 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99883 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99884 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1632_18 := arithEq_of_rows h (row := 1632) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_18
  simp only [← c99879, ← c99880, ← c99881] at e_1632_18
  have e_1632_19 := arithEq_of_rows h (row := 1632) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_19
  simp only [← c99882, ← c99883, k2, ← c99884] at e_1632_19
  have hr := e_1632_19
  simp only [e_1632_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15922 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1633 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16248)) := by
  have c99885 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99886 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99887 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99888 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99889 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99890 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1633_0 := arithEq_of_rows h (row := 1633) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_0
  simp only [← c99885, ← c99886, ← c99887] at e_1633_0
  have e_1633_1 := arithEq_of_rows h (row := 1633) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_1
  simp only [← c99888, ← c99889, k2, ← c99890] at e_1633_1
  have hr := e_1633_1
  simp only [e_1633_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15923 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1633 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16249)) := by
  have c99891 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99892 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99893 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99894 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99895 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99896 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1633_2 := arithEq_of_rows h (row := 1633) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_2
  simp only [← c99891, ← c99892, ← c99893] at e_1633_2
  have e_1633_3 := arithEq_of_rows h (row := 1633) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_3
  simp only [← c99894, ← c99895, k2, ← c99896] at e_1633_3
  have hr := e_1633_3
  simp only [e_1633_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15924 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1633 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16250)) := by
  have c99897 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99898 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99899 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99900 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99901 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99902 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1633_4 := arithEq_of_rows h (row := 1633) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_4
  simp only [← c99897, ← c99898, ← c99899] at e_1633_4
  have e_1633_5 := arithEq_of_rows h (row := 1633) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_5
  simp only [← c99900, ← c99901, k2, ← c99902] at e_1633_5
  have hr := e_1633_5
  simp only [e_1633_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15925 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1633 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16251)) := by
  have c99903 := (publicBatchWrapper86_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99904 := (publicBatchWrapper86_copies3122 a h).1
  have c99905 := (publicBatchWrapper86_copies3122 a h).2.1
  have c99906 := (publicBatchWrapper86_copies3122 a h).2.2.1
  have c99907 := (publicBatchWrapper86_copies3122 a h).2.2.2.1
  have c99908 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1633_6 := arithEq_of_rows h (row := 1633) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_6
  simp only [← c99903, ← c99904, ← c99905] at e_1633_6
  have e_1633_7 := arithEq_of_rows h (row := 1633) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_7
  simp only [← c99906, ← c99907, k2, ← c99908] at e_1633_7
  have hr := e_1633_7
  simp only [e_1633_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15926 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1633 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16252)) := by
  have c99909 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.1
  have c99910 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.1
  have c99911 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.1
  have c99912 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.1
  have c99913 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.1
  have c99914 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1633_8 := arithEq_of_rows h (row := 1633) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_8
  simp only [← c99909, ← c99910, ← c99911] at e_1633_8
  have e_1633_9 := arithEq_of_rows h (row := 1633) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_9
  simp only [← c99912, ← c99913, k2, ← c99914] at e_1633_9
  have hr := e_1633_9
  simp only [e_1633_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15927 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1633 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16253)) := by
  have c99915 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99916 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99917 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99918 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99919 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99920 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1633_10 := arithEq_of_rows h (row := 1633) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_10
  simp only [← c99915, ← c99916, ← c99917] at e_1633_10
  have e_1633_11 := arithEq_of_rows h (row := 1633) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_11
  simp only [← c99918, ← c99919, k2, ← c99920] at e_1633_11
  have hr := e_1633_11
  simp only [e_1633_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15928 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1633 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16254)) := by
  have c99921 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99922 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99923 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99924 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99925 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99926 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1633_12 := arithEq_of_rows h (row := 1633) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_12
  simp only [← c99921, ← c99922, ← c99923] at e_1633_12
  have e_1633_13 := arithEq_of_rows h (row := 1633) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_13
  simp only [← c99924, ← c99925, k2, ← c99926] at e_1633_13
  have hr := e_1633_13
  simp only [e_1633_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15929 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1633 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16255)) := by
  have c99927 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99928 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99929 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99930 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99931 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99932 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1633_14 := arithEq_of_rows h (row := 1633) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_14
  simp only [← c99927, ← c99928, ← c99929] at e_1633_14
  have e_1633_15 := arithEq_of_rows h (row := 1633) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_15
  simp only [← c99930, ← c99931, k2, ← c99932] at e_1633_15
  have hr := e_1633_15
  simp only [e_1633_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15930 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1633 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16256)) := by
  have c99933 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99934 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99935 := (publicBatchWrapper86_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99936 := (publicBatchWrapper86_copies3123 a h).1
  have c99937 := (publicBatchWrapper86_copies3123 a h).2.1
  have c99938 := (publicBatchWrapper86_copies3123 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1633_16 := arithEq_of_rows h (row := 1633) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_16
  simp only [← c99933, ← c99934, ← c99935] at e_1633_16
  have e_1633_17 := arithEq_of_rows h (row := 1633) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_17
  simp only [← c99936, ← c99937, k2, ← c99938] at e_1633_17
  have hr := e_1633_17
  simp only [e_1633_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15931 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1633 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16257)) := by
  have c99939 := (publicBatchWrapper86_copies3123 a h).2.2.2.1
  have c99940 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.1
  have c99941 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.1
  have c99942 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.1
  have c99943 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.1
  have c99944 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1633_18 := arithEq_of_rows h (row := 1633) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_18
  simp only [← c99939, ← c99940, ← c99941] at e_1633_18
  have e_1633_19 := arithEq_of_rows h (row := 1633) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_19
  simp only [← c99942, ← c99943, k2, ← c99944] at e_1633_19
  have hr := e_1633_19
  simp only [e_1633_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15932 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1634 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16258)) := by
  have c99945 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.1
  have c99946 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99947 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99948 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99949 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99950 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1634_0 := arithEq_of_rows h (row := 1634) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_0
  simp only [← c99945, ← c99946, ← c99947] at e_1634_0
  have e_1634_1 := arithEq_of_rows h (row := 1634) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_1
  simp only [← c99948, ← c99949, k2, ← c99950] at e_1634_1
  have hr := e_1634_1
  simp only [e_1634_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15933 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1634 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16259)) := by
  have c99951 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99952 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99953 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99954 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99955 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99956 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1634_2 := arithEq_of_rows h (row := 1634) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_2
  simp only [← c99951, ← c99952, ← c99953] at e_1634_2
  have e_1634_3 := arithEq_of_rows h (row := 1634) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_3
  simp only [← c99954, ← c99955, k2, ← c99956] at e_1634_3
  have hr := e_1634_3
  simp only [e_1634_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15934 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1634 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16260)) := by
  have c99957 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99958 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99959 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99960 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99961 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99962 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1634_4 := arithEq_of_rows h (row := 1634) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_4
  simp only [← c99957, ← c99958, ← c99959] at e_1634_4
  have e_1634_5 := arithEq_of_rows h (row := 1634) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_5
  simp only [← c99960, ← c99961, k2, ← c99962] at e_1634_5
  have hr := e_1634_5
  simp only [e_1634_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15935 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1634 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16261)) := by
  have c99963 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99964 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99965 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99966 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99967 := (publicBatchWrapper86_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99968 := (publicBatchWrapper86_copies3124 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1634_6 := arithEq_of_rows h (row := 1634) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_6
  simp only [← c99963, ← c99964, ← c99965] at e_1634_6
  have e_1634_7 := arithEq_of_rows h (row := 1634) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_7
  simp only [← c99966, ← c99967, k2, ← c99968] at e_1634_7
  have hr := e_1634_7
  simp only [e_1634_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15936 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1634 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16262)) := by
  have c99969 := (publicBatchWrapper86_copies3124 a h).2.1
  have c99970 := (publicBatchWrapper86_copies3124 a h).2.2.1
  have c99971 := (publicBatchWrapper86_copies3124 a h).2.2.2.1
  have c99972 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.1
  have c99973 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.1
  have c99974 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1634_8 := arithEq_of_rows h (row := 1634) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_8
  simp only [← c99969, ← c99970, ← c99971] at e_1634_8
  have e_1634_9 := arithEq_of_rows h (row := 1634) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_9
  simp only [← c99972, ← c99973, k2, ← c99974] at e_1634_9
  have hr := e_1634_9
  simp only [e_1634_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15937 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1634 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16263)) := by
  have c99975 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.1
  have c99976 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.1
  have c99977 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.1
  have c99978 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99979 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99980 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1634_10 := arithEq_of_rows h (row := 1634) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_10
  simp only [← c99975, ← c99976, ← c99977] at e_1634_10
  have e_1634_11 := arithEq_of_rows h (row := 1634) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_11
  simp only [← c99978, ← c99979, k2, ← c99980] at e_1634_11
  have hr := e_1634_11
  simp only [e_1634_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15938 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1634 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16264)) := by
  have c99981 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99982 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99983 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99984 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99985 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99986 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1634_12 := arithEq_of_rows h (row := 1634) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_12
  simp only [← c99981, ← c99982, ← c99983] at e_1634_12
  have e_1634_13 := arithEq_of_rows h (row := 1634) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_13
  simp only [← c99984, ← c99985, k2, ← c99986] at e_1634_13
  have hr := e_1634_13
  simp only [e_1634_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15939 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1634 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16265)) := by
  have c99987 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99988 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99989 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99990 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99991 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99992 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1634_14 := arithEq_of_rows h (row := 1634) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_14
  simp only [← c99987, ← c99988, ← c99989] at e_1634_14
  have e_1634_15 := arithEq_of_rows h (row := 1634) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_15
  simp only [← c99990, ← c99991, k2, ← c99992] at e_1634_15
  have hr := e_1634_15
  simp only [e_1634_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15940 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1634 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16266)) := by
  have c99993 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99994 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99995 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99996 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99997 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99998 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1634_16 := arithEq_of_rows h (row := 1634) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_16
  simp only [← c99993, ← c99994, ← c99995] at e_1634_16
  have e_1634_17 := arithEq_of_rows h (row := 1634) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_17
  simp only [← c99996, ← c99997, k2, ← c99998] at e_1634_17
  have hr := e_1634_17
  simp only [e_1634_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15941 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1634 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16267)) := by
  have c99999 := (publicBatchWrapper86_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100000 := (publicBatchWrapper86_copies3125 a h).1
  have c100001 := (publicBatchWrapper86_copies3125 a h).2.1
  have c100002 := (publicBatchWrapper86_copies3125 a h).2.2.1
  have c100003 := (publicBatchWrapper86_copies3125 a h).2.2.2.1
  have c100004 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1634_18 := arithEq_of_rows h (row := 1634) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_18
  simp only [← c99999, ← c100000, ← c100001] at e_1634_18
  have e_1634_19 := arithEq_of_rows h (row := 1634) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_19
  simp only [← c100002, ← c100003, k2, ← c100004] at e_1634_19
  have hr := e_1634_19
  simp only [e_1634_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15942 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1635 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16268)) := by
  have c100005 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.1
  have c100006 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.1
  have c100007 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.1
  have c100008 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.1
  have c100009 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.1
  have c100010 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1635_0 := arithEq_of_rows h (row := 1635) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_0
  simp only [← c100005, ← c100006, ← c100007] at e_1635_0
  have e_1635_1 := arithEq_of_rows h (row := 1635) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_1
  simp only [← c100008, ← c100009, k2, ← c100010] at e_1635_1
  have hr := e_1635_1
  simp only [e_1635_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15943 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1635 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16269)) := by
  have c100011 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100012 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100013 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100014 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100015 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100016 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1635_2 := arithEq_of_rows h (row := 1635) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_2
  simp only [← c100011, ← c100012, ← c100013] at e_1635_2
  have e_1635_3 := arithEq_of_rows h (row := 1635) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_3
  simp only [← c100014, ← c100015, k2, ← c100016] at e_1635_3
  have hr := e_1635_3
  simp only [e_1635_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15944 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1635 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16270)) := by
  have c100017 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100018 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100019 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100020 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100021 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100022 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1635_4 := arithEq_of_rows h (row := 1635) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_4
  simp only [← c100017, ← c100018, ← c100019] at e_1635_4
  have e_1635_5 := arithEq_of_rows h (row := 1635) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_5
  simp only [← c100020, ← c100021, k2, ← c100022] at e_1635_5
  have hr := e_1635_5
  simp only [e_1635_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15945 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1635 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16271)) := by
  have c100023 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100024 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100025 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100026 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100027 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100028 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1635_6 := arithEq_of_rows h (row := 1635) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_6
  simp only [← c100023, ← c100024, ← c100025] at e_1635_6
  have e_1635_7 := arithEq_of_rows h (row := 1635) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_7
  simp only [← c100026, ← c100027, k2, ← c100028] at e_1635_7
  have hr := e_1635_7
  simp only [e_1635_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15946 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1635 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16272)) := by
  have c100029 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100030 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100031 := (publicBatchWrapper86_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100032 := (publicBatchWrapper86_copies3126 a h).1
  have c100033 := (publicBatchWrapper86_copies3126 a h).2.1
  have c100034 := (publicBatchWrapper86_copies3126 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1635_8 := arithEq_of_rows h (row := 1635) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_8
  simp only [← c100029, ← c100030, ← c100031] at e_1635_8
  have e_1635_9 := arithEq_of_rows h (row := 1635) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_9
  simp only [← c100032, ← c100033, k2, ← c100034] at e_1635_9
  have hr := e_1635_9
  simp only [e_1635_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15947 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1635 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16273)) := by
  have c100035 := (publicBatchWrapper86_copies3126 a h).2.2.2.1
  have c100036 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.1
  have c100037 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.1
  have c100038 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.1
  have c100039 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.1
  have c100040 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1635_10 := arithEq_of_rows h (row := 1635) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_10
  simp only [← c100035, ← c100036, ← c100037] at e_1635_10
  have e_1635_11 := arithEq_of_rows h (row := 1635) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_11
  simp only [← c100038, ← c100039, k2, ← c100040] at e_1635_11
  have hr := e_1635_11
  simp only [e_1635_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15948 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1635 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16274)) := by
  have c100041 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.1
  have c100042 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100043 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100044 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100045 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100046 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1635_12 := arithEq_of_rows h (row := 1635) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_12
  simp only [← c100041, ← c100042, ← c100043] at e_1635_12
  have e_1635_13 := arithEq_of_rows h (row := 1635) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_13
  simp only [← c100044, ← c100045, k2, ← c100046] at e_1635_13
  have hr := e_1635_13
  simp only [e_1635_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15949 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1635 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16275)) := by
  have c100047 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100048 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100049 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100050 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100051 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100052 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1635_14 := arithEq_of_rows h (row := 1635) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_14
  simp only [← c100047, ← c100048, ← c100049] at e_1635_14
  have e_1635_15 := arithEq_of_rows h (row := 1635) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_15
  simp only [← c100050, ← c100051, k2, ← c100052] at e_1635_15
  have hr := e_1635_15
  simp only [e_1635_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15950 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1635 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16276)) := by
  have c100053 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100054 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100055 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100056 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100057 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100058 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1635_16 := arithEq_of_rows h (row := 1635) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_16
  simp only [← c100053, ← c100054, ← c100055] at e_1635_16
  have e_1635_17 := arithEq_of_rows h (row := 1635) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_17
  simp only [← c100056, ← c100057, k2, ← c100058] at e_1635_17
  have hr := e_1635_17
  simp only [e_1635_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15951 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1635 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16277)) := by
  have c100059 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100060 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100061 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100062 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100063 := (publicBatchWrapper86_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100064 := (publicBatchWrapper86_copies3127 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1635_18 := arithEq_of_rows h (row := 1635) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_18
  simp only [← c100059, ← c100060, ← c100061] at e_1635_18
  have e_1635_19 := arithEq_of_rows h (row := 1635) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_19
  simp only [← c100062, ← c100063, k2, ← c100064] at e_1635_19
  have hr := e_1635_19
  simp only [e_1635_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15952 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1636 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16278)) := by
  have c100065 := (publicBatchWrapper86_copies3127 a h).2.1
  have c100066 := (publicBatchWrapper86_copies3127 a h).2.2.1
  have c100067 := (publicBatchWrapper86_copies3127 a h).2.2.2.1
  have c100068 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.1
  have c100069 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.1
  have c100070 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1636_0 := arithEq_of_rows h (row := 1636) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_0
  simp only [← c100065, ← c100066, ← c100067] at e_1636_0
  have e_1636_1 := arithEq_of_rows h (row := 1636) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_1
  simp only [← c100068, ← c100069, k2, ← c100070] at e_1636_1
  have hr := e_1636_1
  simp only [e_1636_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1636 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16279)) := by
  have c100071 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.1
  have c100072 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.1
  have c100073 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.1
  have c100074 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100075 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100076 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1636_2 := arithEq_of_rows h (row := 1636) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_2
  simp only [← c100071, ← c100072, ← c100073] at e_1636_2
  have e_1636_3 := arithEq_of_rows h (row := 1636) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_3
  simp only [← c100074, ← c100075, k2, ← c100076] at e_1636_3
  have hr := e_1636_3
  simp only [e_1636_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15954 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1636 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16280)) := by
  have c100077 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100078 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100079 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100080 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100081 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100082 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1636_4 := arithEq_of_rows h (row := 1636) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_4
  simp only [← c100077, ← c100078, ← c100079] at e_1636_4
  have e_1636_5 := arithEq_of_rows h (row := 1636) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_5
  simp only [← c100080, ← c100081, k2, ← c100082] at e_1636_5
  have hr := e_1636_5
  simp only [e_1636_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15955 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1636 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16281)) := by
  have c100083 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100084 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100085 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100086 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100087 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100088 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1636_6 := arithEq_of_rows h (row := 1636) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_6
  simp only [← c100083, ← c100084, ← c100085] at e_1636_6
  have e_1636_7 := arithEq_of_rows h (row := 1636) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_7
  simp only [← c100086, ← c100087, k2, ← c100088] at e_1636_7
  have hr := e_1636_7
  simp only [e_1636_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15956 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1636 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16282)) := by
  have c100089 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100090 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100091 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100092 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100093 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100094 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1636_8 := arithEq_of_rows h (row := 1636) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_8
  simp only [← c100089, ← c100090, ← c100091] at e_1636_8
  have e_1636_9 := arithEq_of_rows h (row := 1636) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_9
  simp only [← c100092, ← c100093, k2, ← c100094] at e_1636_9
  have hr := e_1636_9
  simp only [e_1636_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15957 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1636 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16283)) := by
  have c100095 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100096 := (publicBatchWrapper86_copies3128 a h).1
  have c100097 := (publicBatchWrapper86_copies3128 a h).2.1
  have c100098 := (publicBatchWrapper86_copies3128 a h).2.2.1
  have c100099 := (publicBatchWrapper86_copies3128 a h).2.2.2.1
  have c100100 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1636_10 := arithEq_of_rows h (row := 1636) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_10
  simp only [← c100095, ← c100096, ← c100097] at e_1636_10
  have e_1636_11 := arithEq_of_rows h (row := 1636) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_11
  simp only [← c100098, ← c100099, k2, ← c100100] at e_1636_11
  have hr := e_1636_11
  simp only [e_1636_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15958 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1636 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16284)) := by
  have c100101 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.1
  have c100102 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.1
  have c100103 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.1
  have c100104 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.1
  have c100105 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.1
  have c100106 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1636_12 := arithEq_of_rows h (row := 1636) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_12
  simp only [← c100101, ← c100102, ← c100103] at e_1636_12
  have e_1636_13 := arithEq_of_rows h (row := 1636) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_13
  simp only [← c100104, ← c100105, k2, ← c100106] at e_1636_13
  have hr := e_1636_13
  simp only [e_1636_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15959 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1636 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16285)) := by
  have c100107 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100108 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100109 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100110 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100111 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100112 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1636_14 := arithEq_of_rows h (row := 1636) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_14
  simp only [← c100107, ← c100108, ← c100109] at e_1636_14
  have e_1636_15 := arithEq_of_rows h (row := 1636) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_15
  simp only [← c100110, ← c100111, k2, ← c100112] at e_1636_15
  have hr := e_1636_15
  simp only [e_1636_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15960 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1636 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16286)) := by
  have c100113 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100114 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100115 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100116 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100117 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100118 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1636_16 := arithEq_of_rows h (row := 1636) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_16
  simp only [← c100113, ← c100114, ← c100115] at e_1636_16
  have e_1636_17 := arithEq_of_rows h (row := 1636) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_17
  simp only [← c100116, ← c100117, k2, ← c100118] at e_1636_17
  have hr := e_1636_17
  simp only [e_1636_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15961 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1636 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16287)) := by
  have c100119 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100120 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100121 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100122 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100123 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100124 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1636_18 := arithEq_of_rows h (row := 1636) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_18
  simp only [← c100119, ← c100120, ← c100121] at e_1636_18
  have e_1636_19 := arithEq_of_rows h (row := 1636) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_19
  simp only [← c100122, ← c100123, k2, ← c100124] at e_1636_19
  have hr := e_1636_19
  simp only [e_1636_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15962 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1637 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16288)) := by
  have c100125 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100126 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100127 := (publicBatchWrapper86_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100128 := (publicBatchWrapper86_copies3129 a h).1
  have c100129 := (publicBatchWrapper86_copies3129 a h).2.1
  have c100130 := (publicBatchWrapper86_copies3129 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1637_0 := arithEq_of_rows h (row := 1637) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_0
  simp only [← c100125, ← c100126, ← c100127] at e_1637_0
  have e_1637_1 := arithEq_of_rows h (row := 1637) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_1
  simp only [← c100128, ← c100129, k2, ← c100130] at e_1637_1
  have hr := e_1637_1
  simp only [e_1637_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15963 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1637 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16289)) := by
  have c100131 := (publicBatchWrapper86_copies3129 a h).2.2.2.1
  have c100132 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.1
  have c100133 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.1
  have c100134 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.1
  have c100135 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.1
  have c100136 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1637_2 := arithEq_of_rows h (row := 1637) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_2
  simp only [← c100131, ← c100132, ← c100133] at e_1637_2
  have e_1637_3 := arithEq_of_rows h (row := 1637) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_3
  simp only [← c100134, ← c100135, k2, ← c100136] at e_1637_3
  have hr := e_1637_3
  simp only [e_1637_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15964 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1637 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16290)) := by
  have c100137 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.1
  have c100138 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100139 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100140 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100141 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100142 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1637_4 := arithEq_of_rows h (row := 1637) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_4
  simp only [← c100137, ← c100138, ← c100139] at e_1637_4
  have e_1637_5 := arithEq_of_rows h (row := 1637) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_5
  simp only [← c100140, ← c100141, k2, ← c100142] at e_1637_5
  have hr := e_1637_5
  simp only [e_1637_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15965 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1637 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16291)) := by
  have c100143 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100144 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100145 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100146 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100147 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100148 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1637_6 := arithEq_of_rows h (row := 1637) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_6
  simp only [← c100143, ← c100144, ← c100145] at e_1637_6
  have e_1637_7 := arithEq_of_rows h (row := 1637) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_7
  simp only [← c100146, ← c100147, k2, ← c100148] at e_1637_7
  have hr := e_1637_7
  simp only [e_1637_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15966 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1637 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16292)) := by
  have c100149 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100150 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100151 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100152 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100153 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100154 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1637_8 := arithEq_of_rows h (row := 1637) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_8
  simp only [← c100149, ← c100150, ← c100151] at e_1637_8
  have e_1637_9 := arithEq_of_rows h (row := 1637) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_9
  simp only [← c100152, ← c100153, k2, ← c100154] at e_1637_9
  have hr := e_1637_9
  simp only [e_1637_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15967 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1637 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16293)) := by
  have c100155 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100156 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100157 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100158 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100159 := (publicBatchWrapper86_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100160 := (publicBatchWrapper86_copies3130 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1637_10 := arithEq_of_rows h (row := 1637) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_10
  simp only [← c100155, ← c100156, ← c100157] at e_1637_10
  have e_1637_11 := arithEq_of_rows h (row := 1637) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_11
  simp only [← c100158, ← c100159, k2, ← c100160] at e_1637_11
  have hr := e_1637_11
  simp only [e_1637_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15968 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1637 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16294)) := by
  have c100161 := (publicBatchWrapper86_copies3130 a h).2.1
  have c100162 := (publicBatchWrapper86_copies3130 a h).2.2.1
  have c100163 := (publicBatchWrapper86_copies3130 a h).2.2.2.1
  have c100164 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.1
  have c100165 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.1
  have c100166 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1637_12 := arithEq_of_rows h (row := 1637) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_12
  simp only [← c100161, ← c100162, ← c100163] at e_1637_12
  have e_1637_13 := arithEq_of_rows h (row := 1637) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_13
  simp only [← c100164, ← c100165, k2, ← c100166] at e_1637_13
  have hr := e_1637_13
  simp only [e_1637_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15969 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1637 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16295)) := by
  have c100167 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.1
  have c100168 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.1
  have c100169 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.1
  have c100170 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100171 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100172 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1637_14 := arithEq_of_rows h (row := 1637) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_14
  simp only [← c100167, ← c100168, ← c100169] at e_1637_14
  have e_1637_15 := arithEq_of_rows h (row := 1637) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_15
  simp only [← c100170, ← c100171, k2, ← c100172] at e_1637_15
  have hr := e_1637_15
  simp only [e_1637_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15970 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1637 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16296)) := by
  have c100173 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100174 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100175 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100176 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100177 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100178 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1637_16 := arithEq_of_rows h (row := 1637) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_16
  simp only [← c100173, ← c100174, ← c100175] at e_1637_16
  have e_1637_17 := arithEq_of_rows h (row := 1637) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_17
  simp only [← c100176, ← c100177, k2, ← c100178] at e_1637_17
  have hr := e_1637_17
  simp only [e_1637_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15971 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1637 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16297)) := by
  have c100179 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100180 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100181 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100182 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100183 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100184 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1637_18 := arithEq_of_rows h (row := 1637) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_18
  simp only [← c100179, ← c100180, ← c100181] at e_1637_18
  have e_1637_19 := arithEq_of_rows h (row := 1637) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_19
  simp only [← c100182, ← c100183, k2, ← c100184] at e_1637_19
  have hr := e_1637_19
  simp only [e_1637_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15972 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1638 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16298)) := by
  have c100185 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100186 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100187 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100188 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100189 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100190 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1638_0 := arithEq_of_rows h (row := 1638) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_0
  simp only [← c100185, ← c100186, ← c100187] at e_1638_0
  have e_1638_1 := arithEq_of_rows h (row := 1638) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_1
  simp only [← c100188, ← c100189, k2, ← c100190] at e_1638_1
  have hr := e_1638_1
  simp only [e_1638_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15973 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1638 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16299)) := by
  have c100191 := (publicBatchWrapper86_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100192 := (publicBatchWrapper86_copies3131 a h).1
  have c100193 := (publicBatchWrapper86_copies3131 a h).2.1
  have c100194 := (publicBatchWrapper86_copies3131 a h).2.2.1
  have c100195 := (publicBatchWrapper86_copies3131 a h).2.2.2.1
  have c100196 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1638_2 := arithEq_of_rows h (row := 1638) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_2
  simp only [← c100191, ← c100192, ← c100193] at e_1638_2
  have e_1638_3 := arithEq_of_rows h (row := 1638) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_3
  simp only [← c100194, ← c100195, k2, ← c100196] at e_1638_3
  have hr := e_1638_3
  simp only [e_1638_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1638 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16300)) := by
  have c100197 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.1
  have c100198 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.1
  have c100199 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.1
  have c100200 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.1
  have c100201 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.1
  have c100202 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1638_4 := arithEq_of_rows h (row := 1638) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_4
  simp only [← c100197, ← c100198, ← c100199] at e_1638_4
  have e_1638_5 := arithEq_of_rows h (row := 1638) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_5
  simp only [← c100200, ← c100201, k2, ← c100202] at e_1638_5
  have hr := e_1638_5
  simp only [e_1638_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1638 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16301)) := by
  have c100203 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100204 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100205 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100206 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100207 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100208 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1638_6 := arithEq_of_rows h (row := 1638) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_6
  simp only [← c100203, ← c100204, ← c100205] at e_1638_6
  have e_1638_7 := arithEq_of_rows h (row := 1638) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_7
  simp only [← c100206, ← c100207, k2, ← c100208] at e_1638_7
  have hr := e_1638_7
  simp only [e_1638_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1638 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16302)) := by
  have c100209 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100210 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100211 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100212 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100213 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100214 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1638_8 := arithEq_of_rows h (row := 1638) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_8
  simp only [← c100209, ← c100210, ← c100211] at e_1638_8
  have e_1638_9 := arithEq_of_rows h (row := 1638) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_9
  simp only [← c100212, ← c100213, k2, ← c100214] at e_1638_9
  have hr := e_1638_9
  simp only [e_1638_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1638 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16303)) := by
  have c100215 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100216 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100217 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100218 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100219 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100220 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1638_10 := arithEq_of_rows h (row := 1638) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_10
  simp only [← c100215, ← c100216, ← c100217] at e_1638_10
  have e_1638_11 := arithEq_of_rows h (row := 1638) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_11
  simp only [← c100218, ← c100219, k2, ← c100220] at e_1638_11
  have hr := e_1638_11
  simp only [e_1638_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1638 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16304)) := by
  have c100221 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100222 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100223 := (publicBatchWrapper86_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100224 := (publicBatchWrapper86_copies3132 a h).1
  have c100225 := (publicBatchWrapper86_copies3132 a h).2.1
  have c100226 := (publicBatchWrapper86_copies3132 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1638_12 := arithEq_of_rows h (row := 1638) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_12
  simp only [← c100221, ← c100222, ← c100223] at e_1638_12
  have e_1638_13 := arithEq_of_rows h (row := 1638) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_13
  simp only [← c100224, ← c100225, k2, ← c100226] at e_1638_13
  have hr := e_1638_13
  simp only [e_1638_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1638 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16305)) := by
  have c100227 := (publicBatchWrapper86_copies3132 a h).2.2.2.1
  have c100228 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.1
  have c100229 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.1
  have c100230 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.1
  have c100231 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.1
  have c100232 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1638_14 := arithEq_of_rows h (row := 1638) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_14
  simp only [← c100227, ← c100228, ← c100229] at e_1638_14
  have e_1638_15 := arithEq_of_rows h (row := 1638) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_15
  simp only [← c100230, ← c100231, k2, ← c100232] at e_1638_15
  have hr := e_1638_15
  simp only [e_1638_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1638 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16306)) := by
  have c100233 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.1
  have c100234 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100235 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100236 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100237 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100238 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1638_16 := arithEq_of_rows h (row := 1638) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_16
  simp only [← c100233, ← c100234, ← c100235] at e_1638_16
  have e_1638_17 := arithEq_of_rows h (row := 1638) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_17
  simp only [← c100236, ← c100237, k2, ← c100238] at e_1638_17
  have hr := e_1638_17
  simp only [e_1638_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1638 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16307)) := by
  have c100239 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100240 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100241 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100242 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100243 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100244 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1638_18 := arithEq_of_rows h (row := 1638) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_18
  simp only [← c100239, ← c100240, ← c100241] at e_1638_18
  have e_1638_19 := arithEq_of_rows h (row := 1638) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_19
  simp only [← c100242, ← c100243, k2, ← c100244] at e_1638_19
  have hr := e_1638_19
  simp only [e_1638_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1639 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16308)) := by
  have c100245 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100246 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100247 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100248 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100249 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100250 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1639_0 := arithEq_of_rows h (row := 1639) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_0
  simp only [← c100245, ← c100246, ← c100247] at e_1639_0
  have e_1639_1 := arithEq_of_rows h (row := 1639) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_1
  simp only [← c100248, ← c100249, k2, ← c100250] at e_1639_1
  have hr := e_1639_1
  simp only [e_1639_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1639 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16309)) := by
  have c100251 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100252 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100253 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100254 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100255 := (publicBatchWrapper86_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100256 := (publicBatchWrapper86_copies3133 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1639_2 := arithEq_of_rows h (row := 1639) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_2
  simp only [← c100251, ← c100252, ← c100253] at e_1639_2
  have e_1639_3 := arithEq_of_rows h (row := 1639) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_3
  simp only [← c100254, ← c100255, k2, ← c100256] at e_1639_3
  have hr := e_1639_3
  simp only [e_1639_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1639 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16310)) := by
  have c100257 := (publicBatchWrapper86_copies3133 a h).2.1
  have c100258 := (publicBatchWrapper86_copies3133 a h).2.2.1
  have c100259 := (publicBatchWrapper86_copies3133 a h).2.2.2.1
  have c100260 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.1
  have c100261 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.1
  have c100262 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1639_4 := arithEq_of_rows h (row := 1639) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_4
  simp only [← c100257, ← c100258, ← c100259] at e_1639_4
  have e_1639_5 := arithEq_of_rows h (row := 1639) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_5
  simp only [← c100260, ← c100261, k2, ← c100262] at e_1639_5
  have hr := e_1639_5
  simp only [e_1639_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1639 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16311)) := by
  have c100263 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.1
  have c100264 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.1
  have c100265 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.1
  have c100266 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100267 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100268 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1639_6 := arithEq_of_rows h (row := 1639) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_6
  simp only [← c100263, ← c100264, ← c100265] at e_1639_6
  have e_1639_7 := arithEq_of_rows h (row := 1639) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_7
  simp only [← c100266, ← c100267, k2, ← c100268] at e_1639_7
  have hr := e_1639_7
  simp only [e_1639_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1639 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16312)) := by
  have c100269 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100270 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100271 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100272 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100273 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100274 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1639_8 := arithEq_of_rows h (row := 1639) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_8
  simp only [← c100269, ← c100270, ← c100271] at e_1639_8
  have e_1639_9 := arithEq_of_rows h (row := 1639) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_9
  simp only [← c100272, ← c100273, k2, ← c100274] at e_1639_9
  have hr := e_1639_9
  simp only [e_1639_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1639 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16313)) := by
  have c100275 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100276 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100277 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100278 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100279 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100280 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1639_10 := arithEq_of_rows h (row := 1639) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_10
  simp only [← c100275, ← c100276, ← c100277] at e_1639_10
  have e_1639_11 := arithEq_of_rows h (row := 1639) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_11
  simp only [← c100278, ← c100279, k2, ← c100280] at e_1639_11
  have hr := e_1639_11
  simp only [e_1639_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1639 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16314)) := by
  have c100281 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100282 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100283 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100284 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100285 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100286 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1639_12 := arithEq_of_rows h (row := 1639) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_12
  simp only [← c100281, ← c100282, ← c100283] at e_1639_12
  have e_1639_13 := arithEq_of_rows h (row := 1639) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_13
  simp only [← c100284, ← c100285, k2, ← c100286] at e_1639_13
  have hr := e_1639_13
  simp only [e_1639_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1639 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16315)) := by
  have c100287 := (publicBatchWrapper86_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100288 := (publicBatchWrapper86_copies3134 a h).1
  have c100289 := (publicBatchWrapper86_copies3134 a h).2.1
  have c100290 := (publicBatchWrapper86_copies3134 a h).2.2.1
  have c100291 := (publicBatchWrapper86_copies3134 a h).2.2.2.1
  have c100292 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1639_14 := arithEq_of_rows h (row := 1639) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_14
  simp only [← c100287, ← c100288, ← c100289] at e_1639_14
  have e_1639_15 := arithEq_of_rows h (row := 1639) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_15
  simp only [← c100290, ← c100291, k2, ← c100292] at e_1639_15
  have hr := e_1639_15
  simp only [e_1639_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1639 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16316)) := by
  have c100293 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.1
  have c100294 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.1
  have c100295 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.1
  have c100296 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.1
  have c100297 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.1
  have c100298 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1639_16 := arithEq_of_rows h (row := 1639) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_16
  simp only [← c100293, ← c100294, ← c100295] at e_1639_16
  have e_1639_17 := arithEq_of_rows h (row := 1639) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_17
  simp only [← c100296, ← c100297, k2, ← c100298] at e_1639_17
  have hr := e_1639_17
  simp only [e_1639_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1639 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16317)) := by
  have c100299 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100300 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100301 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100302 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100303 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100304 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1639_18 := arithEq_of_rows h (row := 1639) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_18
  simp only [← c100299, ← c100300, ← c100301] at e_1639_18
  have e_1639_19 := arithEq_of_rows h (row := 1639) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_19
  simp only [← c100302, ← c100303, k2, ← c100304] at e_1639_19
  have hr := e_1639_19
  simp only [e_1639_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1640 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16318)) := by
  have c100305 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100306 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100307 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100308 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100309 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100310 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1640_0 := arithEq_of_rows h (row := 1640) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_0
  simp only [← c100305, ← c100306, ← c100307] at e_1640_0
  have e_1640_1 := arithEq_of_rows h (row := 1640) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_1
  simp only [← c100308, ← c100309, k2, ← c100310] at e_1640_1
  have hr := e_1640_1
  simp only [e_1640_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1640 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16319)) := by
  have c100311 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100312 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100313 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100314 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100315 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100316 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1640_2 := arithEq_of_rows h (row := 1640) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_2
  simp only [← c100311, ← c100312, ← c100313] at e_1640_2
  have e_1640_3 := arithEq_of_rows h (row := 1640) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_3
  simp only [← c100314, ← c100315, k2, ← c100316] at e_1640_3
  have hr := e_1640_3
  simp only [e_1640_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1640 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16320)) := by
  have c100317 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100318 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100319 := (publicBatchWrapper86_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100320 := (publicBatchWrapper86_copies3135 a h).1
  have c100321 := (publicBatchWrapper86_copies3135 a h).2.1
  have c100322 := (publicBatchWrapper86_copies3135 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1640_4 := arithEq_of_rows h (row := 1640) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_4
  simp only [← c100317, ← c100318, ← c100319] at e_1640_4
  have e_1640_5 := arithEq_of_rows h (row := 1640) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_5
  simp only [← c100320, ← c100321, k2, ← c100322] at e_1640_5
  have hr := e_1640_5
  simp only [e_1640_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1640 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16321)) := by
  have c100323 := (publicBatchWrapper86_copies3135 a h).2.2.2.1
  have c100324 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.1
  have c100325 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.1
  have c100326 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.1
  have c100327 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.1
  have c100328 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1640_6 := arithEq_of_rows h (row := 1640) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_6
  simp only [← c100323, ← c100324, ← c100325] at e_1640_6
  have e_1640_7 := arithEq_of_rows h (row := 1640) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_7
  simp only [← c100326, ← c100327, k2, ← c100328] at e_1640_7
  have hr := e_1640_7
  simp only [e_1640_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1640 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16322)) := by
  have c100329 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.1
  have c100330 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100331 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100332 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100333 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100334 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1640_8 := arithEq_of_rows h (row := 1640) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_8
  simp only [← c100329, ← c100330, ← c100331] at e_1640_8
  have e_1640_9 := arithEq_of_rows h (row := 1640) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_9
  simp only [← c100332, ← c100333, k2, ← c100334] at e_1640_9
  have hr := e_1640_9
  simp only [e_1640_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1640 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16323)) := by
  have c100335 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100336 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100337 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100338 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100339 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100340 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1640_10 := arithEq_of_rows h (row := 1640) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_10
  simp only [← c100335, ← c100336, ← c100337] at e_1640_10
  have e_1640_11 := arithEq_of_rows h (row := 1640) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_11
  simp only [← c100338, ← c100339, k2, ← c100340] at e_1640_11
  have hr := e_1640_11
  simp only [e_1640_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1640 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16324)) := by
  have c100341 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100342 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100343 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100344 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100345 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100346 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1640_12 := arithEq_of_rows h (row := 1640) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_12
  simp only [← c100341, ← c100342, ← c100343] at e_1640_12
  have e_1640_13 := arithEq_of_rows h (row := 1640) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_13
  simp only [← c100344, ← c100345, k2, ← c100346] at e_1640_13
  have hr := e_1640_13
  simp only [e_1640_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1640 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16325)) := by
  have c100347 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100348 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100349 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100350 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100351 := (publicBatchWrapper86_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100352 := (publicBatchWrapper86_copies3136 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1640_14 := arithEq_of_rows h (row := 1640) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_14
  simp only [← c100347, ← c100348, ← c100349] at e_1640_14
  have e_1640_15 := arithEq_of_rows h (row := 1640) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_15
  simp only [← c100350, ← c100351, k2, ← c100352] at e_1640_15
  have hr := e_1640_15
  simp only [e_1640_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1640 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16326)) := by
  have c100353 := (publicBatchWrapper86_copies3136 a h).2.1
  have c100354 := (publicBatchWrapper86_copies3136 a h).2.2.1
  have c100355 := (publicBatchWrapper86_copies3136 a h).2.2.2.1
  have c100356 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.1
  have c100357 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.1
  have c100358 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1640_16 := arithEq_of_rows h (row := 1640) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_16
  simp only [← c100353, ← c100354, ← c100355] at e_1640_16
  have e_1640_17 := arithEq_of_rows h (row := 1640) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_17
  simp only [← c100356, ← c100357, k2, ← c100358] at e_1640_17
  have hr := e_1640_17
  simp only [e_1640_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1640 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16327)) := by
  have c100359 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.1
  have c100360 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.1
  have c100361 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.1
  have c100362 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100363 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100364 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1640_18 := arithEq_of_rows h (row := 1640) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_18
  simp only [← c100359, ← c100360, ← c100361] at e_1640_18
  have e_1640_19 := arithEq_of_rows h (row := 1640) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_19
  simp only [← c100362, ← c100363, k2, ← c100364] at e_1640_19
  have hr := e_1640_19
  simp only [e_1640_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1641 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16328)) := by
  have c100365 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100366 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100367 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100368 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100369 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100370 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1641_0 := arithEq_of_rows h (row := 1641) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_0
  simp only [← c100365, ← c100366, ← c100367] at e_1641_0
  have e_1641_1 := arithEq_of_rows h (row := 1641) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_1
  simp only [← c100368, ← c100369, k2, ← c100370] at e_1641_1
  have hr := e_1641_1
  simp only [e_1641_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1641 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16329)) := by
  have c100371 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100372 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100373 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100374 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100375 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100376 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1641_2 := arithEq_of_rows h (row := 1641) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_2
  simp only [← c100371, ← c100372, ← c100373] at e_1641_2
  have e_1641_3 := arithEq_of_rows h (row := 1641) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_3
  simp only [← c100374, ← c100375, k2, ← c100376] at e_1641_3
  have hr := e_1641_3
  simp only [e_1641_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1641 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16330)) := by
  have c100377 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100378 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100379 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100380 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100381 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100382 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1641_4 := arithEq_of_rows h (row := 1641) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_4
  simp only [← c100377, ← c100378, ← c100379] at e_1641_4
  have e_1641_5 := arithEq_of_rows h (row := 1641) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_5
  simp only [← c100380, ← c100381, k2, ← c100382] at e_1641_5
  have hr := e_1641_5
  simp only [e_1641_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1641 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16331)) := by
  have c100383 := (publicBatchWrapper86_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100384 := (publicBatchWrapper86_copies3137 a h).1
  have c100385 := (publicBatchWrapper86_copies3137 a h).2.1
  have c100386 := (publicBatchWrapper86_copies3137 a h).2.2.1
  have c100387 := (publicBatchWrapper86_copies3137 a h).2.2.2.1
  have c100388 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1641_6 := arithEq_of_rows h (row := 1641) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_6
  simp only [← c100383, ← c100384, ← c100385] at e_1641_6
  have e_1641_7 := arithEq_of_rows h (row := 1641) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_7
  simp only [← c100386, ← c100387, k2, ← c100388] at e_1641_7
  have hr := e_1641_7
  simp only [e_1641_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1641 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16332)) := by
  have c100389 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.1
  have c100390 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.1
  have c100391 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.1
  have c100392 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.1
  have c100393 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.1
  have c100394 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1641_8 := arithEq_of_rows h (row := 1641) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_8
  simp only [← c100389, ← c100390, ← c100391] at e_1641_8
  have e_1641_9 := arithEq_of_rows h (row := 1641) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_9
  simp only [← c100392, ← c100393, k2, ← c100394] at e_1641_9
  have hr := e_1641_9
  simp only [e_1641_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1641 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16333)) := by
  have c100395 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100396 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100397 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100398 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100399 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100400 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1641_10 := arithEq_of_rows h (row := 1641) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_10
  simp only [← c100395, ← c100396, ← c100397] at e_1641_10
  have e_1641_11 := arithEq_of_rows h (row := 1641) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_11
  simp only [← c100398, ← c100399, k2, ← c100400] at e_1641_11
  have hr := e_1641_11
  simp only [e_1641_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1641 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16334)) := by
  have c100401 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100402 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100403 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100404 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100405 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100406 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1641_12 := arithEq_of_rows h (row := 1641) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_12
  simp only [← c100401, ← c100402, ← c100403] at e_1641_12
  have e_1641_13 := arithEq_of_rows h (row := 1641) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_13
  simp only [← c100404, ← c100405, k2, ← c100406] at e_1641_13
  have hr := e_1641_13
  simp only [e_1641_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1641 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16335)) := by
  have c100407 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100408 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100409 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100410 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100411 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100412 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1641_14 := arithEq_of_rows h (row := 1641) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_14
  simp only [← c100407, ← c100408, ← c100409] at e_1641_14
  have e_1641_15 := arithEq_of_rows h (row := 1641) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_15
  simp only [← c100410, ← c100411, k2, ← c100412] at e_1641_15
  have hr := e_1641_15
  simp only [e_1641_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1641 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16336)) := by
  have c100413 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100414 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100415 := (publicBatchWrapper86_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100416 := (publicBatchWrapper86_copies3138 a h).1
  have c100417 := (publicBatchWrapper86_copies3138 a h).2.1
  have c100418 := (publicBatchWrapper86_copies3138 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1641_16 := arithEq_of_rows h (row := 1641) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_16
  simp only [← c100413, ← c100414, ← c100415] at e_1641_16
  have e_1641_17 := arithEq_of_rows h (row := 1641) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_17
  simp only [← c100416, ← c100417, k2, ← c100418] at e_1641_17
  have hr := e_1641_17
  simp only [e_1641_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1641 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16337)) := by
  have c100419 := (publicBatchWrapper86_copies3138 a h).2.2.2.1
  have c100420 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.1
  have c100421 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.1
  have c100422 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.1
  have c100423 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.1
  have c100424 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1641_18 := arithEq_of_rows h (row := 1641) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_18
  simp only [← c100419, ← c100420, ← c100421] at e_1641_18
  have e_1641_19 := arithEq_of_rows h (row := 1641) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_19
  simp only [← c100422, ← c100423, k2, ← c100424] at e_1641_19
  have hr := e_1641_19
  simp only [e_1641_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1642 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16338)) := by
  have c100425 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.1
  have c100426 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100427 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100428 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100429 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100430 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1642_0 := arithEq_of_rows h (row := 1642) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_0
  simp only [← c100425, ← c100426, ← c100427] at e_1642_0
  have e_1642_1 := arithEq_of_rows h (row := 1642) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_1
  simp only [← c100428, ← c100429, k2, ← c100430] at e_1642_1
  have hr := e_1642_1
  simp only [e_1642_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1642 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16339)) := by
  have c100431 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100432 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100433 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100434 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100435 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100436 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1642_2 := arithEq_of_rows h (row := 1642) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_2
  simp only [← c100431, ← c100432, ← c100433] at e_1642_2
  have e_1642_3 := arithEq_of_rows h (row := 1642) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_3
  simp only [← c100434, ← c100435, k2, ← c100436] at e_1642_3
  have hr := e_1642_3
  simp only [e_1642_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1642 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16340)) := by
  have c100437 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100438 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100439 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100440 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100441 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100442 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1642_4 := arithEq_of_rows h (row := 1642) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_4
  simp only [← c100437, ← c100438, ← c100439] at e_1642_4
  have e_1642_5 := arithEq_of_rows h (row := 1642) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_5
  simp only [← c100440, ← c100441, k2, ← c100442] at e_1642_5
  have hr := e_1642_5
  simp only [e_1642_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1642 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16341)) := by
  have c100443 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100444 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100445 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100446 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100447 := (publicBatchWrapper86_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100448 := (publicBatchWrapper86_copies3139 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1642_6 := arithEq_of_rows h (row := 1642) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_6
  simp only [← c100443, ← c100444, ← c100445] at e_1642_6
  have e_1642_7 := arithEq_of_rows h (row := 1642) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_7
  simp only [← c100446, ← c100447, k2, ← c100448] at e_1642_7
  have hr := e_1642_7
  simp only [e_1642_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1642 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16342)) := by
  have c100449 := (publicBatchWrapper86_copies3139 a h).2.1
  have c100450 := (publicBatchWrapper86_copies3139 a h).2.2.1
  have c100451 := (publicBatchWrapper86_copies3139 a h).2.2.2.1
  have c100452 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.1
  have c100453 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.1
  have c100454 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1642_8 := arithEq_of_rows h (row := 1642) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_8
  simp only [← c100449, ← c100450, ← c100451] at e_1642_8
  have e_1642_9 := arithEq_of_rows h (row := 1642) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_9
  simp only [← c100452, ← c100453, k2, ← c100454] at e_1642_9
  have hr := e_1642_9
  simp only [e_1642_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1642 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16343)) := by
  have c100455 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.1
  have c100456 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.1
  have c100457 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.1
  have c100458 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100459 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100460 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1642_10 := arithEq_of_rows h (row := 1642) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_10
  simp only [← c100455, ← c100456, ← c100457] at e_1642_10
  have e_1642_11 := arithEq_of_rows h (row := 1642) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_11
  simp only [← c100458, ← c100459, k2, ← c100460] at e_1642_11
  have hr := e_1642_11
  simp only [e_1642_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1642 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16344)) := by
  have c100461 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100462 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100463 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100464 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100465 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100466 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1642_12 := arithEq_of_rows h (row := 1642) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_12
  simp only [← c100461, ← c100462, ← c100463] at e_1642_12
  have e_1642_13 := arithEq_of_rows h (row := 1642) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_13
  simp only [← c100464, ← c100465, k2, ← c100466] at e_1642_13
  have hr := e_1642_13
  simp only [e_1642_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1642 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16345)) := by
  have c100467 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100468 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100469 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100470 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100471 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100472 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1642_14 := arithEq_of_rows h (row := 1642) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_14
  simp only [← c100467, ← c100468, ← c100469] at e_1642_14
  have e_1642_15 := arithEq_of_rows h (row := 1642) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_15
  simp only [← c100470, ← c100471, k2, ← c100472] at e_1642_15
  have hr := e_1642_15
  simp only [e_1642_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1642 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16346)) := by
  have c100473 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100474 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100475 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100476 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100477 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100478 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1642_16 := arithEq_of_rows h (row := 1642) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_16
  simp only [← c100473, ← c100474, ← c100475] at e_1642_16
  have e_1642_17 := arithEq_of_rows h (row := 1642) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_17
  simp only [← c100476, ← c100477, k2, ← c100478] at e_1642_17
  have hr := e_1642_17
  simp only [e_1642_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1642 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16347)) := by
  have c100479 := (publicBatchWrapper86_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100480 := (publicBatchWrapper86_copies3140 a h).1
  have c100481 := (publicBatchWrapper86_copies3140 a h).2.1
  have c100482 := (publicBatchWrapper86_copies3140 a h).2.2.1
  have c100483 := (publicBatchWrapper86_copies3140 a h).2.2.2.1
  have c100484 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1642_18 := arithEq_of_rows h (row := 1642) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_18
  simp only [← c100479, ← c100480, ← c100481] at e_1642_18
  have e_1642_19 := arithEq_of_rows h (row := 1642) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_19
  simp only [← c100482, ← c100483, k2, ← c100484] at e_1642_19
  have hr := e_1642_19
  simp only [e_1642_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1643 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16348)) := by
  have c100485 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.1
  have c100486 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.1
  have c100487 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.1
  have c100488 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.1
  have c100489 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.1
  have c100490 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1643_0 := arithEq_of_rows h (row := 1643) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_0
  simp only [← c100485, ← c100486, ← c100487] at e_1643_0
  have e_1643_1 := arithEq_of_rows h (row := 1643) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_1
  simp only [← c100488, ← c100489, k2, ← c100490] at e_1643_1
  have hr := e_1643_1
  simp only [e_1643_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1643 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16349)) := by
  have c100491 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100492 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100493 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100494 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100495 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100496 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1643_2 := arithEq_of_rows h (row := 1643) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_2
  simp only [← c100491, ← c100492, ← c100493] at e_1643_2
  have e_1643_3 := arithEq_of_rows h (row := 1643) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_3
  simp only [← c100494, ← c100495, k2, ← c100496] at e_1643_3
  have hr := e_1643_3
  simp only [e_1643_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1643 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16350)) := by
  have c100497 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100498 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100499 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100500 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100501 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100502 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1643_4 := arithEq_of_rows h (row := 1643) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_4
  simp only [← c100497, ← c100498, ← c100499] at e_1643_4
  have e_1643_5 := arithEq_of_rows h (row := 1643) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_5
  simp only [← c100500, ← c100501, k2, ← c100502] at e_1643_5
  have hr := e_1643_5
  simp only [e_1643_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1643 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16351)) := by
  have c100503 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100504 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100505 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100506 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100507 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100508 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1643_6 := arithEq_of_rows h (row := 1643) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_6
  simp only [← c100503, ← c100504, ← c100505] at e_1643_6
  have e_1643_7 := arithEq_of_rows h (row := 1643) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_7
  simp only [← c100506, ← c100507, k2, ← c100508] at e_1643_7
  have hr := e_1643_7
  simp only [e_1643_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1643 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16352)) := by
  have c100509 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100510 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100511 := (publicBatchWrapper86_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100512 := (publicBatchWrapper86_copies3141 a h).1
  have c100513 := (publicBatchWrapper86_copies3141 a h).2.1
  have c100514 := (publicBatchWrapper86_copies3141 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1643_8 := arithEq_of_rows h (row := 1643) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_8
  simp only [← c100509, ← c100510, ← c100511] at e_1643_8
  have e_1643_9 := arithEq_of_rows h (row := 1643) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_9
  simp only [← c100512, ← c100513, k2, ← c100514] at e_1643_9
  have hr := e_1643_9
  simp only [e_1643_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1643 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16353)) := by
  have c100515 := (publicBatchWrapper86_copies3141 a h).2.2.2.1
  have c100516 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.1
  have c100517 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.1
  have c100518 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.1
  have c100519 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.1
  have c100520 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1643_10 := arithEq_of_rows h (row := 1643) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_10
  simp only [← c100515, ← c100516, ← c100517] at e_1643_10
  have e_1643_11 := arithEq_of_rows h (row := 1643) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_11
  simp only [← c100518, ← c100519, k2, ← c100520] at e_1643_11
  have hr := e_1643_11
  simp only [e_1643_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1643 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16354)) := by
  have c100521 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.1
  have c100522 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100523 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100524 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100525 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100526 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1643_12 := arithEq_of_rows h (row := 1643) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_12
  simp only [← c100521, ← c100522, ← c100523] at e_1643_12
  have e_1643_13 := arithEq_of_rows h (row := 1643) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_13
  simp only [← c100524, ← c100525, k2, ← c100526] at e_1643_13
  have hr := e_1643_13
  simp only [e_1643_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1643 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16355)) := by
  have c100527 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100528 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100529 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100530 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100531 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100532 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1643_14 := arithEq_of_rows h (row := 1643) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_14
  simp only [← c100527, ← c100528, ← c100529] at e_1643_14
  have e_1643_15 := arithEq_of_rows h (row := 1643) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_15
  simp only [← c100530, ← c100531, k2, ← c100532] at e_1643_15
  have hr := e_1643_15
  simp only [e_1643_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1643 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16356)) := by
  have c100533 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100534 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100535 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100536 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100537 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100538 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1643_16 := arithEq_of_rows h (row := 1643) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_16
  simp only [← c100533, ← c100534, ← c100535] at e_1643_16
  have e_1643_17 := arithEq_of_rows h (row := 1643) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_17
  simp only [← c100536, ← c100537, k2, ← c100538] at e_1643_17
  have hr := e_1643_17
  simp only [e_1643_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1643 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16357)) := by
  have c100539 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100540 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100541 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100542 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100543 := (publicBatchWrapper86_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100544 := (publicBatchWrapper86_copies3142 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1643_18 := arithEq_of_rows h (row := 1643) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_18
  simp only [← c100539, ← c100540, ← c100541] at e_1643_18
  have e_1643_19 := arithEq_of_rows h (row := 1643) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_19
  simp only [← c100542, ← c100543, k2, ← c100544] at e_1643_19
  have hr := e_1643_19
  simp only [e_1643_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1644 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16358)) := by
  have c100545 := (publicBatchWrapper86_copies3142 a h).2.1
  have c100546 := (publicBatchWrapper86_copies3142 a h).2.2.1
  have c100547 := (publicBatchWrapper86_copies3142 a h).2.2.2.1
  have c100548 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.1
  have c100549 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.1
  have c100550 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1644_0 := arithEq_of_rows h (row := 1644) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_0
  simp only [← c100545, ← c100546, ← c100547] at e_1644_0
  have e_1644_1 := arithEq_of_rows h (row := 1644) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_1
  simp only [← c100548, ← c100549, k2, ← c100550] at e_1644_1
  have hr := e_1644_1
  simp only [e_1644_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1644 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16359)) := by
  have c100551 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.1
  have c100552 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.1
  have c100553 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.1
  have c100554 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100555 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100556 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1644_2 := arithEq_of_rows h (row := 1644) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_2
  simp only [← c100551, ← c100552, ← c100553] at e_1644_2
  have e_1644_3 := arithEq_of_rows h (row := 1644) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_3
  simp only [← c100554, ← c100555, k2, ← c100556] at e_1644_3
  have hr := e_1644_3
  simp only [e_1644_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1644 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16360)) := by
  have c100557 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100558 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100559 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100560 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100561 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100562 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1644_4 := arithEq_of_rows h (row := 1644) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_4
  simp only [← c100557, ← c100558, ← c100559] at e_1644_4
  have e_1644_5 := arithEq_of_rows h (row := 1644) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_5
  simp only [← c100560, ← c100561, k2, ← c100562] at e_1644_5
  have hr := e_1644_5
  simp only [e_1644_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1644 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16361)) := by
  have c100563 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100564 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100565 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100566 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100567 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100568 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1644_6 := arithEq_of_rows h (row := 1644) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_6
  simp only [← c100563, ← c100564, ← c100565] at e_1644_6
  have e_1644_7 := arithEq_of_rows h (row := 1644) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_7
  simp only [← c100566, ← c100567, k2, ← c100568] at e_1644_7
  have hr := e_1644_7
  simp only [e_1644_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1644 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16362)) := by
  have c100569 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100570 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100571 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100572 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100573 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100574 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1644_8 := arithEq_of_rows h (row := 1644) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_8
  simp only [← c100569, ← c100570, ← c100571] at e_1644_8
  have e_1644_9 := arithEq_of_rows h (row := 1644) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_9
  simp only [← c100572, ← c100573, k2, ← c100574] at e_1644_9
  have hr := e_1644_9
  simp only [e_1644_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1644 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16363)) := by
  have c100575 := (publicBatchWrapper86_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100576 := (publicBatchWrapper86_copies3143 a h).1
  have c100577 := (publicBatchWrapper86_copies3143 a h).2.1
  have c100578 := (publicBatchWrapper86_copies3143 a h).2.2.1
  have c100579 := (publicBatchWrapper86_copies3143 a h).2.2.2.1
  have c100580 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1644_10 := arithEq_of_rows h (row := 1644) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_10
  simp only [← c100575, ← c100576, ← c100577] at e_1644_10
  have e_1644_11 := arithEq_of_rows h (row := 1644) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_11
  simp only [← c100578, ← c100579, k2, ← c100580] at e_1644_11
  have hr := e_1644_11
  simp only [e_1644_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1644 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16364)) := by
  have c100581 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.1
  have c100582 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.1
  have c100583 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.1
  have c100584 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.1
  have c100585 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.1
  have c100586 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1644_12 := arithEq_of_rows h (row := 1644) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_12
  simp only [← c100581, ← c100582, ← c100583] at e_1644_12
  have e_1644_13 := arithEq_of_rows h (row := 1644) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_13
  simp only [← c100584, ← c100585, k2, ← c100586] at e_1644_13
  have hr := e_1644_13
  simp only [e_1644_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1644 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16365)) := by
  have c100587 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100588 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100589 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100590 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100591 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100592 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1644_14 := arithEq_of_rows h (row := 1644) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_14
  simp only [← c100587, ← c100588, ← c100589] at e_1644_14
  have e_1644_15 := arithEq_of_rows h (row := 1644) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_15
  simp only [← c100590, ← c100591, k2, ← c100592] at e_1644_15
  have hr := e_1644_15
  simp only [e_1644_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1644 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16366)) := by
  have c100593 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100594 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100595 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100596 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100597 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100598 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1644_16 := arithEq_of_rows h (row := 1644) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_16
  simp only [← c100593, ← c100594, ← c100595] at e_1644_16
  have e_1644_17 := arithEq_of_rows h (row := 1644) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_17
  simp only [← c100596, ← c100597, k2, ← c100598] at e_1644_17
  have hr := e_1644_17
  simp only [e_1644_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1644 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16367)) := by
  have c100599 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100600 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100601 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100602 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100603 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100604 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1644_18 := arithEq_of_rows h (row := 1644) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_18
  simp only [← c100599, ← c100600, ← c100601] at e_1644_18
  have e_1644_19 := arithEq_of_rows h (row := 1644) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_19
  simp only [← c100602, ← c100603, k2, ← c100604] at e_1644_19
  have hr := e_1644_19
  simp only [e_1644_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1645 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16368)) := by
  have c100605 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100606 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100607 := (publicBatchWrapper86_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100608 := (publicBatchWrapper86_copies3144 a h).1
  have c100609 := (publicBatchWrapper86_copies3144 a h).2.1
  have c100610 := (publicBatchWrapper86_copies3144 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1645_0 := arithEq_of_rows h (row := 1645) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_0
  simp only [← c100605, ← c100606, ← c100607] at e_1645_0
  have e_1645_1 := arithEq_of_rows h (row := 1645) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_1
  simp only [← c100608, ← c100609, k2, ← c100610] at e_1645_1
  have hr := e_1645_1
  simp only [e_1645_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1645 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16369)) := by
  have c100611 := (publicBatchWrapper86_copies3144 a h).2.2.2.1
  have c100612 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.1
  have c100613 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.1
  have c100614 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.1
  have c100615 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.1
  have c100616 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1645_2 := arithEq_of_rows h (row := 1645) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_2
  simp only [← c100611, ← c100612, ← c100613] at e_1645_2
  have e_1645_3 := arithEq_of_rows h (row := 1645) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_3
  simp only [← c100614, ← c100615, k2, ← c100616] at e_1645_3
  have hr := e_1645_3
  simp only [e_1645_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1645 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16410)) := by
  have c100617 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.1
  have c100618 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100619 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100620 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100621 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100622 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1645_4 := arithEq_of_rows h (row := 1645) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_4
  simp only [← c100617, ← c100618, ← c100619] at e_1645_4
  have e_1645_5 := arithEq_of_rows h (row := 1645) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_5
  simp only [← c100620, ← c100621, k2, ← c100622] at e_1645_5
  have hr := e_1645_5
  simp only [e_1645_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1645 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16411)) := by
  have c100623 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100624 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100625 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100626 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100627 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100628 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1645_6 := arithEq_of_rows h (row := 1645) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_6
  simp only [← c100623, ← c100624, ← c100625] at e_1645_6
  have e_1645_7 := arithEq_of_rows h (row := 1645) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_7
  simp only [← c100626, ← c100627, k2, ← c100628] at e_1645_7
  have hr := e_1645_7
  simp only [e_1645_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1645 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16412)) := by
  have c100629 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100630 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100631 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100632 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100633 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100634 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1645_8 := arithEq_of_rows h (row := 1645) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_8
  simp only [← c100629, ← c100630, ← c100631] at e_1645_8
  have e_1645_9 := arithEq_of_rows h (row := 1645) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_9
  simp only [← c100632, ← c100633, k2, ← c100634] at e_1645_9
  have hr := e_1645_9
  simp only [e_1645_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1645 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16413)) := by
  have c100635 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100636 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100637 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100638 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100639 := (publicBatchWrapper86_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100640 := (publicBatchWrapper86_copies3145 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1645_10 := arithEq_of_rows h (row := 1645) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_10
  simp only [← c100635, ← c100636, ← c100637] at e_1645_10
  have e_1645_11 := arithEq_of_rows h (row := 1645) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_11
  simp only [← c100638, ← c100639, k2, ← c100640] at e_1645_11
  have hr := e_1645_11
  simp only [e_1645_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1645 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16414)) := by
  have c100641 := (publicBatchWrapper86_copies3145 a h).2.1
  have c100642 := (publicBatchWrapper86_copies3145 a h).2.2.1
  have c100643 := (publicBatchWrapper86_copies3145 a h).2.2.2.1
  have c100644 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.1
  have c100645 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.1
  have c100646 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1645_12 := arithEq_of_rows h (row := 1645) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_12
  simp only [← c100641, ← c100642, ← c100643] at e_1645_12
  have e_1645_13 := arithEq_of_rows h (row := 1645) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_13
  simp only [← c100644, ← c100645, k2, ← c100646] at e_1645_13
  have hr := e_1645_13
  simp only [e_1645_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1645 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16415)) := by
  have c100647 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.1
  have c100648 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.1
  have c100649 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.1
  have c100650 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100651 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100652 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1645_14 := arithEq_of_rows h (row := 1645) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_14
  simp only [← c100647, ← c100648, ← c100649] at e_1645_14
  have e_1645_15 := arithEq_of_rows h (row := 1645) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_15
  simp only [← c100650, ← c100651, k2, ← c100652] at e_1645_15
  have hr := e_1645_15
  simp only [e_1645_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1645 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16416)) := by
  have c100653 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100654 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100655 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100656 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100657 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100658 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1645_16 := arithEq_of_rows h (row := 1645) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_16
  simp only [← c100653, ← c100654, ← c100655] at e_1645_16
  have e_1645_17 := arithEq_of_rows h (row := 1645) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_17
  simp only [← c100656, ← c100657, k2, ← c100658] at e_1645_17
  have hr := e_1645_17
  simp only [e_1645_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1645 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16417)) := by
  have c100659 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100660 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100661 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100662 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100663 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100664 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1645_18 := arithEq_of_rows h (row := 1645) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_18
  simp only [← c100659, ← c100660, ← c100661] at e_1645_18
  have e_1645_19 := arithEq_of_rows h (row := 1645) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_19
  simp only [← c100662, ← c100663, k2, ← c100664] at e_1645_19
  have hr := e_1645_19
  simp only [e_1645_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1646 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16418)) := by
  have c100665 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100666 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100667 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100668 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100669 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100670 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1646_0 := arithEq_of_rows h (row := 1646) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_0
  simp only [← c100665, ← c100666, ← c100667] at e_1646_0
  have e_1646_1 := arithEq_of_rows h (row := 1646) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_1
  simp only [← c100668, ← c100669, k2, ← c100670] at e_1646_1
  have hr := e_1646_1
  simp only [e_1646_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1646 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16419)) := by
  have c100671 := (publicBatchWrapper86_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100672 := (publicBatchWrapper86_copies3146 a h).1
  have c100673 := (publicBatchWrapper86_copies3146 a h).2.1
  have c100674 := (publicBatchWrapper86_copies3146 a h).2.2.1
  have c100675 := (publicBatchWrapper86_copies3146 a h).2.2.2.1
  have c100676 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1646_2 := arithEq_of_rows h (row := 1646) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_2
  simp only [← c100671, ← c100672, ← c100673] at e_1646_2
  have e_1646_3 := arithEq_of_rows h (row := 1646) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_3
  simp only [← c100674, ← c100675, k2, ← c100676] at e_1646_3
  have hr := e_1646_3
  simp only [e_1646_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1646 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16420)) := by
  have c100677 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.1
  have c100678 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.1
  have c100679 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.1
  have c100680 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.1
  have c100681 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.1
  have c100682 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1646_4 := arithEq_of_rows h (row := 1646) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_4
  simp only [← c100677, ← c100678, ← c100679] at e_1646_4
  have e_1646_5 := arithEq_of_rows h (row := 1646) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_5
  simp only [← c100680, ← c100681, k2, ← c100682] at e_1646_5
  have hr := e_1646_5
  simp only [e_1646_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1646 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16421)) := by
  have c100683 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100684 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100685 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100686 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100687 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100688 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1646_6 := arithEq_of_rows h (row := 1646) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_6
  simp only [← c100683, ← c100684, ← c100685] at e_1646_6
  have e_1646_7 := arithEq_of_rows h (row := 1646) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_7
  simp only [← c100686, ← c100687, k2, ← c100688] at e_1646_7
  have hr := e_1646_7
  simp only [e_1646_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1646 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16422)) := by
  have c100689 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100690 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100691 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100692 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100693 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100694 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1646_8 := arithEq_of_rows h (row := 1646) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_8
  simp only [← c100689, ← c100690, ← c100691] at e_1646_8
  have e_1646_9 := arithEq_of_rows h (row := 1646) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_9
  simp only [← c100692, ← c100693, k2, ← c100694] at e_1646_9
  have hr := e_1646_9
  simp only [e_1646_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1646 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16423)) := by
  have c100695 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100696 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100697 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100698 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100699 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100700 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1646_10 := arithEq_of_rows h (row := 1646) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_10
  simp only [← c100695, ← c100696, ← c100697] at e_1646_10
  have e_1646_11 := arithEq_of_rows h (row := 1646) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_11
  simp only [← c100698, ← c100699, k2, ← c100700] at e_1646_11
  have hr := e_1646_11
  simp only [e_1646_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1646 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16424)) := by
  have c100701 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100702 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100703 := (publicBatchWrapper86_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100704 := (publicBatchWrapper86_copies3147 a h).1
  have c100705 := (publicBatchWrapper86_copies3147 a h).2.1
  have c100706 := (publicBatchWrapper86_copies3147 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1646_12 := arithEq_of_rows h (row := 1646) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_12
  simp only [← c100701, ← c100702, ← c100703] at e_1646_12
  have e_1646_13 := arithEq_of_rows h (row := 1646) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_13
  simp only [← c100704, ← c100705, k2, ← c100706] at e_1646_13
  have hr := e_1646_13
  simp only [e_1646_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1646 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16425)) := by
  have c100707 := (publicBatchWrapper86_copies3147 a h).2.2.2.1
  have c100708 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.1
  have c100709 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.1
  have c100710 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.1
  have c100711 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.1
  have c100712 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1646_14 := arithEq_of_rows h (row := 1646) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_14
  simp only [← c100707, ← c100708, ← c100709] at e_1646_14
  have e_1646_15 := arithEq_of_rows h (row := 1646) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_15
  simp only [← c100710, ← c100711, k2, ← c100712] at e_1646_15
  have hr := e_1646_15
  simp only [e_1646_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1646 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16426)) := by
  have c100713 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.1
  have c100714 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100715 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100716 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100717 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100718 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1646_16 := arithEq_of_rows h (row := 1646) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_16
  simp only [← c100713, ← c100714, ← c100715] at e_1646_16
  have e_1646_17 := arithEq_of_rows h (row := 1646) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_17
  simp only [← c100716, ← c100717, k2, ← c100718] at e_1646_17
  have hr := e_1646_17
  simp only [e_1646_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1646 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16427)) := by
  have c100719 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100720 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100721 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100722 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100723 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100724 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1646_18 := arithEq_of_rows h (row := 1646) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_18
  simp only [← c100719, ← c100720, ← c100721] at e_1646_18
  have e_1646_19 := arithEq_of_rows h (row := 1646) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_19
  simp only [← c100722, ← c100723, k2, ← c100724] at e_1646_19
  have hr := e_1646_19
  simp only [e_1646_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1647 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16428)) := by
  have c100725 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100726 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100727 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100728 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100729 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100730 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1647_0 := arithEq_of_rows h (row := 1647) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_0
  simp only [← c100725, ← c100726, ← c100727] at e_1647_0
  have e_1647_1 := arithEq_of_rows h (row := 1647) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_1
  simp only [← c100728, ← c100729, k2, ← c100730] at e_1647_1
  have hr := e_1647_1
  simp only [e_1647_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1647 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16429)) := by
  have c100731 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100732 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100733 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100734 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100735 := (publicBatchWrapper86_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100736 := (publicBatchWrapper86_copies3148 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1647_2 := arithEq_of_rows h (row := 1647) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_2
  simp only [← c100731, ← c100732, ← c100733] at e_1647_2
  have e_1647_3 := arithEq_of_rows h (row := 1647) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_3
  simp only [← c100734, ← c100735, k2, ← c100736] at e_1647_3
  have hr := e_1647_3
  simp only [e_1647_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1647 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16430)) := by
  have c100737 := (publicBatchWrapper86_copies3148 a h).2.1
  have c100738 := (publicBatchWrapper86_copies3148 a h).2.2.1
  have c100739 := (publicBatchWrapper86_copies3148 a h).2.2.2.1
  have c100740 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.1
  have c100741 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.1
  have c100742 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1647_4 := arithEq_of_rows h (row := 1647) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_4
  simp only [← c100737, ← c100738, ← c100739] at e_1647_4
  have e_1647_5 := arithEq_of_rows h (row := 1647) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_5
  simp only [← c100740, ← c100741, k2, ← c100742] at e_1647_5
  have hr := e_1647_5
  simp only [e_1647_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1647 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16431)) := by
  have c100743 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.1
  have c100744 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.1
  have c100745 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.1
  have c100746 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100747 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100748 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1647_6 := arithEq_of_rows h (row := 1647) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_6
  simp only [← c100743, ← c100744, ← c100745] at e_1647_6
  have e_1647_7 := arithEq_of_rows h (row := 1647) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_7
  simp only [← c100746, ← c100747, k2, ← c100748] at e_1647_7
  have hr := e_1647_7
  simp only [e_1647_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1647 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16432)) := by
  have c100749 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100750 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100751 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100752 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100753 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100754 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1647_8 := arithEq_of_rows h (row := 1647) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_8
  simp only [← c100749, ← c100750, ← c100751] at e_1647_8
  have e_1647_9 := arithEq_of_rows h (row := 1647) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_9
  simp only [← c100752, ← c100753, k2, ← c100754] at e_1647_9
  have hr := e_1647_9
  simp only [e_1647_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1647 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16433)) := by
  have c100755 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100756 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100757 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100758 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100759 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100760 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1647_10 := arithEq_of_rows h (row := 1647) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_10
  simp only [← c100755, ← c100756, ← c100757] at e_1647_10
  have e_1647_11 := arithEq_of_rows h (row := 1647) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_11
  simp only [← c100758, ← c100759, k2, ← c100760] at e_1647_11
  have hr := e_1647_11
  simp only [e_1647_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1647 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16434)) := by
  have c100761 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100762 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100763 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100764 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100765 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100766 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1647_12 := arithEq_of_rows h (row := 1647) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_12
  simp only [← c100761, ← c100762, ← c100763] at e_1647_12
  have e_1647_13 := arithEq_of_rows h (row := 1647) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_13
  simp only [← c100764, ← c100765, k2, ← c100766] at e_1647_13
  have hr := e_1647_13
  simp only [e_1647_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1647 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16435)) := by
  have c100767 := (publicBatchWrapper86_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100768 := (publicBatchWrapper86_copies3149 a h).1
  have c100769 := (publicBatchWrapper86_copies3149 a h).2.1
  have c100770 := (publicBatchWrapper86_copies3149 a h).2.2.1
  have c100771 := (publicBatchWrapper86_copies3149 a h).2.2.2.1
  have c100772 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1647_14 := arithEq_of_rows h (row := 1647) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_14
  simp only [← c100767, ← c100768, ← c100769] at e_1647_14
  have e_1647_15 := arithEq_of_rows h (row := 1647) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_15
  simp only [← c100770, ← c100771, k2, ← c100772] at e_1647_15
  have hr := e_1647_15
  simp only [e_1647_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1647 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16436)) := by
  have c100773 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.1
  have c100774 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.1
  have c100775 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.1
  have c100776 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.1
  have c100777 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.1
  have c100778 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1647_16 := arithEq_of_rows h (row := 1647) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_16
  simp only [← c100773, ← c100774, ← c100775] at e_1647_16
  have e_1647_17 := arithEq_of_rows h (row := 1647) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_17
  simp only [← c100776, ← c100777, k2, ← c100778] at e_1647_17
  have hr := e_1647_17
  simp only [e_1647_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1647 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16437)) := by
  have c100779 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100780 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100781 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100782 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100783 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100784 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1647_18 := arithEq_of_rows h (row := 1647) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_18
  simp only [← c100779, ← c100780, ← c100781] at e_1647_18
  have e_1647_19 := arithEq_of_rows h (row := 1647) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_19
  simp only [← c100782, ← c100783, k2, ← c100784] at e_1647_19
  have hr := e_1647_19
  simp only [e_1647_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1648 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16438)) := by
  have c100785 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100786 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100787 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100788 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100789 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100790 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1648_0 := arithEq_of_rows h (row := 1648) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_0
  simp only [← c100785, ← c100786, ← c100787] at e_1648_0
  have e_1648_1 := arithEq_of_rows h (row := 1648) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_1
  simp only [← c100788, ← c100789, k2, ← c100790] at e_1648_1
  have hr := e_1648_1
  simp only [e_1648_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1648 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16439)) := by
  have c100791 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100792 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100793 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100794 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100795 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100796 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1648_2 := arithEq_of_rows h (row := 1648) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_2
  simp only [← c100791, ← c100792, ← c100793] at e_1648_2
  have e_1648_3 := arithEq_of_rows h (row := 1648) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_3
  simp only [← c100794, ← c100795, k2, ← c100796] at e_1648_3
  have hr := e_1648_3
  simp only [e_1648_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1648 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16440)) := by
  have c100797 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100798 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100799 := (publicBatchWrapper86_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100800 := (publicBatchWrapper86_copies3150 a h).1
  have c100801 := (publicBatchWrapper86_copies3150 a h).2.1
  have c100802 := (publicBatchWrapper86_copies3150 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1648_4 := arithEq_of_rows h (row := 1648) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_4
  simp only [← c100797, ← c100798, ← c100799] at e_1648_4
  have e_1648_5 := arithEq_of_rows h (row := 1648) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_5
  simp only [← c100800, ← c100801, k2, ← c100802] at e_1648_5
  have hr := e_1648_5
  simp only [e_1648_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1648 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16441)) := by
  have c100803 := (publicBatchWrapper86_copies3150 a h).2.2.2.1
  have c100804 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.1
  have c100805 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.1
  have c100806 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.1
  have c100807 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.1
  have c100808 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1648_6 := arithEq_of_rows h (row := 1648) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_6
  simp only [← c100803, ← c100804, ← c100805] at e_1648_6
  have e_1648_7 := arithEq_of_rows h (row := 1648) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_7
  simp only [← c100806, ← c100807, k2, ← c100808] at e_1648_7
  have hr := e_1648_7
  simp only [e_1648_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1648 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16442)) := by
  have c100809 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.1
  have c100810 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100811 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100812 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100813 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100814 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1648_8 := arithEq_of_rows h (row := 1648) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_8
  simp only [← c100809, ← c100810, ← c100811] at e_1648_8
  have e_1648_9 := arithEq_of_rows h (row := 1648) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_9
  simp only [← c100812, ← c100813, k2, ← c100814] at e_1648_9
  have hr := e_1648_9
  simp only [e_1648_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1648 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16443)) := by
  have c100815 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100816 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100817 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100818 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100819 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100820 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1648_10 := arithEq_of_rows h (row := 1648) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_10
  simp only [← c100815, ← c100816, ← c100817] at e_1648_10
  have e_1648_11 := arithEq_of_rows h (row := 1648) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_11
  simp only [← c100818, ← c100819, k2, ← c100820] at e_1648_11
  have hr := e_1648_11
  simp only [e_1648_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1648 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16444)) := by
  have c100821 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100822 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100823 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100824 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100825 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100826 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1648_12 := arithEq_of_rows h (row := 1648) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_12
  simp only [← c100821, ← c100822, ← c100823] at e_1648_12
  have e_1648_13 := arithEq_of_rows h (row := 1648) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_13
  simp only [← c100824, ← c100825, k2, ← c100826] at e_1648_13
  have hr := e_1648_13
  simp only [e_1648_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1648 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16445)) := by
  have c100827 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100828 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100829 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100830 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100831 := (publicBatchWrapper86_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100832 := (publicBatchWrapper86_copies3151 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1648_14 := arithEq_of_rows h (row := 1648) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_14
  simp only [← c100827, ← c100828, ← c100829] at e_1648_14
  have e_1648_15 := arithEq_of_rows h (row := 1648) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_15
  simp only [← c100830, ← c100831, k2, ← c100832] at e_1648_15
  have hr := e_1648_15
  simp only [e_1648_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1648 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16446)) := by
  have c100833 := (publicBatchWrapper86_copies3151 a h).2.1
  have c100834 := (publicBatchWrapper86_copies3151 a h).2.2.1
  have c100835 := (publicBatchWrapper86_copies3151 a h).2.2.2.1
  have c100836 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.1
  have c100837 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.1
  have c100838 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1648_16 := arithEq_of_rows h (row := 1648) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_16
  simp only [← c100833, ← c100834, ← c100835] at e_1648_16
  have e_1648_17 := arithEq_of_rows h (row := 1648) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_17
  simp only [← c100836, ← c100837, k2, ← c100838] at e_1648_17
  have hr := e_1648_17
  simp only [e_1648_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1648 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16447)) := by
  have c100839 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.1
  have c100840 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.1
  have c100841 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.1
  have c100842 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100843 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100844 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1648_18 := arithEq_of_rows h (row := 1648) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_18
  simp only [← c100839, ← c100840, ← c100841] at e_1648_18
  have e_1648_19 := arithEq_of_rows h (row := 1648) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_19
  simp only [← c100842, ← c100843, k2, ← c100844] at e_1648_19
  have hr := e_1648_19
  simp only [e_1648_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1649 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16448)) := by
  have c100845 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100846 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100847 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100848 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100849 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100850 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1649_0 := arithEq_of_rows h (row := 1649) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_0
  simp only [← c100845, ← c100846, ← c100847] at e_1649_0
  have e_1649_1 := arithEq_of_rows h (row := 1649) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_1
  simp only [← c100848, ← c100849, k2, ← c100850] at e_1649_1
  have hr := e_1649_1
  simp only [e_1649_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1649 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16449)) := by
  have c100851 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100852 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100853 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100854 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100855 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100856 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1649_2 := arithEq_of_rows h (row := 1649) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_2
  simp only [← c100851, ← c100852, ← c100853] at e_1649_2
  have e_1649_3 := arithEq_of_rows h (row := 1649) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_3
  simp only [← c100854, ← c100855, k2, ← c100856] at e_1649_3
  have hr := e_1649_3
  simp only [e_1649_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1649 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16450)) := by
  have c100857 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100858 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100859 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100860 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100861 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100862 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1649_4 := arithEq_of_rows h (row := 1649) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_4
  simp only [← c100857, ← c100858, ← c100859] at e_1649_4
  have e_1649_5 := arithEq_of_rows h (row := 1649) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_5
  simp only [← c100860, ← c100861, k2, ← c100862] at e_1649_5
  have hr := e_1649_5
  simp only [e_1649_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1649 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16451)) := by
  have c100863 := (publicBatchWrapper86_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100864 := (publicBatchWrapper86_copies3152 a h).1
  have c100865 := (publicBatchWrapper86_copies3152 a h).2.1
  have c100866 := (publicBatchWrapper86_copies3152 a h).2.2.1
  have c100867 := (publicBatchWrapper86_copies3152 a h).2.2.2.1
  have c100868 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1649_6 := arithEq_of_rows h (row := 1649) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_6
  simp only [← c100863, ← c100864, ← c100865] at e_1649_6
  have e_1649_7 := arithEq_of_rows h (row := 1649) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_7
  simp only [← c100866, ← c100867, k2, ← c100868] at e_1649_7
  have hr := e_1649_7
  simp only [e_1649_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1649 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16452)) := by
  have c100869 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.1
  have c100870 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.1
  have c100871 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.1
  have c100872 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.1
  have c100873 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.1
  have c100874 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1649_8 := arithEq_of_rows h (row := 1649) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_8
  simp only [← c100869, ← c100870, ← c100871] at e_1649_8
  have e_1649_9 := arithEq_of_rows h (row := 1649) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_9
  simp only [← c100872, ← c100873, k2, ← c100874] at e_1649_9
  have hr := e_1649_9
  simp only [e_1649_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1649 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16453)) := by
  have c100875 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100876 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100877 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100878 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100879 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100880 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1649_10 := arithEq_of_rows h (row := 1649) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_10
  simp only [← c100875, ← c100876, ← c100877] at e_1649_10
  have e_1649_11 := arithEq_of_rows h (row := 1649) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_11
  simp only [← c100878, ← c100879, k2, ← c100880] at e_1649_11
  have hr := e_1649_11
  simp only [e_1649_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1649 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16454)) := by
  have c100881 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100882 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100883 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100884 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100885 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100886 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1649_12 := arithEq_of_rows h (row := 1649) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_12
  simp only [← c100881, ← c100882, ← c100883] at e_1649_12
  have e_1649_13 := arithEq_of_rows h (row := 1649) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_13
  simp only [← c100884, ← c100885, k2, ← c100886] at e_1649_13
  have hr := e_1649_13
  simp only [e_1649_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1649 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16455)) := by
  have c100887 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100888 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100889 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100890 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100891 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100892 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1649_14 := arithEq_of_rows h (row := 1649) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_14
  simp only [← c100887, ← c100888, ← c100889] at e_1649_14
  have e_1649_15 := arithEq_of_rows h (row := 1649) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_15
  simp only [← c100890, ← c100891, k2, ← c100892] at e_1649_15
  have hr := e_1649_15
  simp only [e_1649_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1649 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16456)) := by
  have c100893 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100894 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100895 := (publicBatchWrapper86_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100896 := (publicBatchWrapper86_copies3153 a h).1
  have c100897 := (publicBatchWrapper86_copies3153 a h).2.1
  have c100898 := (publicBatchWrapper86_copies3153 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1649_16 := arithEq_of_rows h (row := 1649) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_16
  simp only [← c100893, ← c100894, ← c100895] at e_1649_16
  have e_1649_17 := arithEq_of_rows h (row := 1649) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_17
  simp only [← c100896, ← c100897, k2, ← c100898] at e_1649_17
  have hr := e_1649_17
  simp only [e_1649_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1649 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16457)) := by
  have c100899 := (publicBatchWrapper86_copies3153 a h).2.2.2.1
  have c100900 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.1
  have c100901 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.1
  have c100902 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.1
  have c100903 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.1
  have c100904 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1649_18 := arithEq_of_rows h (row := 1649) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_18
  simp only [← c100899, ← c100900, ← c100901] at e_1649_18
  have e_1649_19 := arithEq_of_rows h (row := 1649) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_19
  simp only [← c100902, ← c100903, k2, ← c100904] at e_1649_19
  have hr := e_1649_19
  simp only [e_1649_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1650 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16458)) := by
  have c100905 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.1
  have c100906 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100907 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100908 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100909 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100910 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1650_0 := arithEq_of_rows h (row := 1650) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_0
  simp only [← c100905, ← c100906, ← c100907] at e_1650_0
  have e_1650_1 := arithEq_of_rows h (row := 1650) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_1
  simp only [← c100908, ← c100909, k2, ← c100910] at e_1650_1
  have hr := e_1650_1
  simp only [e_1650_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1650 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16459)) := by
  have c100911 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100912 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100913 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100914 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100915 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100916 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1650_2 := arithEq_of_rows h (row := 1650) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_2
  simp only [← c100911, ← c100912, ← c100913] at e_1650_2
  have e_1650_3 := arithEq_of_rows h (row := 1650) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_3
  simp only [← c100914, ← c100915, k2, ← c100916] at e_1650_3
  have hr := e_1650_3
  simp only [e_1650_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1650 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16460)) := by
  have c100917 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100918 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100919 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100920 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100921 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100922 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1650_4 := arithEq_of_rows h (row := 1650) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_4
  simp only [← c100917, ← c100918, ← c100919] at e_1650_4
  have e_1650_5 := arithEq_of_rows h (row := 1650) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_5
  simp only [← c100920, ← c100921, k2, ← c100922] at e_1650_5
  have hr := e_1650_5
  simp only [e_1650_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1650 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16461)) := by
  have c100923 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100924 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100925 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100926 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100927 := (publicBatchWrapper86_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100928 := (publicBatchWrapper86_copies3154 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1650_6 := arithEq_of_rows h (row := 1650) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_6
  simp only [← c100923, ← c100924, ← c100925] at e_1650_6
  have e_1650_7 := arithEq_of_rows h (row := 1650) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_7
  simp only [← c100926, ← c100927, k2, ← c100928] at e_1650_7
  have hr := e_1650_7
  simp only [e_1650_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1650 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16462)) := by
  have c100929 := (publicBatchWrapper86_copies3154 a h).2.1
  have c100930 := (publicBatchWrapper86_copies3154 a h).2.2.1
  have c100931 := (publicBatchWrapper86_copies3154 a h).2.2.2.1
  have c100932 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.1
  have c100933 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.1
  have c100934 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1650_8 := arithEq_of_rows h (row := 1650) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_8
  simp only [← c100929, ← c100930, ← c100931] at e_1650_8
  have e_1650_9 := arithEq_of_rows h (row := 1650) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_9
  simp only [← c100932, ← c100933, k2, ← c100934] at e_1650_9
  have hr := e_1650_9
  simp only [e_1650_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1650 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16463)) := by
  have c100935 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.1
  have c100936 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.1
  have c100937 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.1
  have c100938 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100939 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100940 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1650_10 := arithEq_of_rows h (row := 1650) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_10
  simp only [← c100935, ← c100936, ← c100937] at e_1650_10
  have e_1650_11 := arithEq_of_rows h (row := 1650) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_11
  simp only [← c100938, ← c100939, k2, ← c100940] at e_1650_11
  have hr := e_1650_11
  simp only [e_1650_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1650 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16464)) := by
  have c100941 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100942 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100943 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100944 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100945 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100946 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1650_12 := arithEq_of_rows h (row := 1650) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_12
  simp only [← c100941, ← c100942, ← c100943] at e_1650_12
  have e_1650_13 := arithEq_of_rows h (row := 1650) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_13
  simp only [← c100944, ← c100945, k2, ← c100946] at e_1650_13
  have hr := e_1650_13
  simp only [e_1650_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1650 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16465)) := by
  have c100947 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100948 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100949 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100950 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100951 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100952 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1650_14 := arithEq_of_rows h (row := 1650) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_14
  simp only [← c100947, ← c100948, ← c100949] at e_1650_14
  have e_1650_15 := arithEq_of_rows h (row := 1650) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_15
  simp only [← c100950, ← c100951, k2, ← c100952] at e_1650_15
  have hr := e_1650_15
  simp only [e_1650_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1650 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16466)) := by
  have c100953 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100954 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100955 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100956 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100957 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100958 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1650_16 := arithEq_of_rows h (row := 1650) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_16
  simp only [← c100953, ← c100954, ← c100955] at e_1650_16
  have e_1650_17 := arithEq_of_rows h (row := 1650) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_17
  simp only [← c100956, ← c100957, k2, ← c100958] at e_1650_17
  have hr := e_1650_17
  simp only [e_1650_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1650 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16467)) := by
  have c100959 := (publicBatchWrapper86_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100960 := (publicBatchWrapper86_copies3155 a h).1
  have c100961 := (publicBatchWrapper86_copies3155 a h).2.1
  have c100962 := (publicBatchWrapper86_copies3155 a h).2.2.1
  have c100963 := (publicBatchWrapper86_copies3155 a h).2.2.2.1
  have c100964 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1650_18 := arithEq_of_rows h (row := 1650) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_18
  simp only [← c100959, ← c100960, ← c100961] at e_1650_18
  have e_1650_19 := arithEq_of_rows h (row := 1650) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_19
  simp only [← c100962, ← c100963, k2, ← c100964] at e_1650_19
  have hr := e_1650_19
  simp only [e_1650_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1651 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16468)) := by
  have c100965 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.1
  have c100966 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.1
  have c100967 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.1
  have c100968 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.1
  have c100969 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.1
  have c100970 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1651_0 := arithEq_of_rows h (row := 1651) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_0
  simp only [← c100965, ← c100966, ← c100967] at e_1651_0
  have e_1651_1 := arithEq_of_rows h (row := 1651) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_1
  simp only [← c100968, ← c100969, k2, ← c100970] at e_1651_1
  have hr := e_1651_1
  simp only [e_1651_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1651 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16469)) := by
  have c100971 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100972 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100973 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100974 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100975 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100976 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1651_2 := arithEq_of_rows h (row := 1651) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_2
  simp only [← c100971, ← c100972, ← c100973] at e_1651_2
  have e_1651_3 := arithEq_of_rows h (row := 1651) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_3
  simp only [← c100974, ← c100975, k2, ← c100976] at e_1651_3
  have hr := e_1651_3
  simp only [e_1651_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1651 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16470)) := by
  have c100977 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100978 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100979 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100980 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100981 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100982 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1651_4 := arithEq_of_rows h (row := 1651) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_4
  simp only [← c100977, ← c100978, ← c100979] at e_1651_4
  have e_1651_5 := arithEq_of_rows h (row := 1651) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_5
  simp only [← c100980, ← c100981, k2, ← c100982] at e_1651_5
  have hr := e_1651_5
  simp only [e_1651_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1651 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16471)) := by
  have c100983 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100984 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100985 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100986 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100987 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100988 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1651_6 := arithEq_of_rows h (row := 1651) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_6
  simp only [← c100983, ← c100984, ← c100985] at e_1651_6
  have e_1651_7 := arithEq_of_rows h (row := 1651) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_7
  simp only [← c100986, ← c100987, k2, ← c100988] at e_1651_7
  have hr := e_1651_7
  simp only [e_1651_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1651 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16472)) := by
  have c100989 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100990 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100991 := (publicBatchWrapper86_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100992 := (publicBatchWrapper86_copies3156 a h).1
  have c100993 := (publicBatchWrapper86_copies3156 a h).2.1
  have c100994 := (publicBatchWrapper86_copies3156 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1651_8 := arithEq_of_rows h (row := 1651) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_8
  simp only [← c100989, ← c100990, ← c100991] at e_1651_8
  have e_1651_9 := arithEq_of_rows h (row := 1651) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_9
  simp only [← c100992, ← c100993, k2, ← c100994] at e_1651_9
  have hr := e_1651_9
  simp only [e_1651_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1651 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16473)) := by
  have c100995 := (publicBatchWrapper86_copies3156 a h).2.2.2.1
  have c100996 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.1
  have c100997 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.1
  have c100998 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.1
  have c100999 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.1
  have c101000 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1651_10 := arithEq_of_rows h (row := 1651) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_10
  simp only [← c100995, ← c100996, ← c100997] at e_1651_10
  have e_1651_11 := arithEq_of_rows h (row := 1651) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_11
  simp only [← c100998, ← c100999, k2, ← c101000] at e_1651_11
  have hr := e_1651_11
  simp only [e_1651_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1651 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16474)) := by
  have c101001 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.1
  have c101002 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101003 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101004 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101005 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101006 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1651_12 := arithEq_of_rows h (row := 1651) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_12
  simp only [← c101001, ← c101002, ← c101003] at e_1651_12
  have e_1651_13 := arithEq_of_rows h (row := 1651) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_13
  simp only [← c101004, ← c101005, k2, ← c101006] at e_1651_13
  have hr := e_1651_13
  simp only [e_1651_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1651 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16475)) := by
  have c101007 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101008 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101009 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101010 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101011 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101012 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1651_14 := arithEq_of_rows h (row := 1651) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_14
  simp only [← c101007, ← c101008, ← c101009] at e_1651_14
  have e_1651_15 := arithEq_of_rows h (row := 1651) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_15
  simp only [← c101010, ← c101011, k2, ← c101012] at e_1651_15
  have hr := e_1651_15
  simp only [e_1651_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1651 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16476)) := by
  have c101013 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101014 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101015 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101016 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101017 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101018 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1651_16 := arithEq_of_rows h (row := 1651) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_16
  simp only [← c101013, ← c101014, ← c101015] at e_1651_16
  have e_1651_17 := arithEq_of_rows h (row := 1651) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_17
  simp only [← c101016, ← c101017, k2, ← c101018] at e_1651_17
  have hr := e_1651_17
  simp only [e_1651_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1651 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16477)) := by
  have c101019 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101020 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101021 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101022 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101023 := (publicBatchWrapper86_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101024 := (publicBatchWrapper86_copies3157 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1651_18 := arithEq_of_rows h (row := 1651) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_18
  simp only [← c101019, ← c101020, ← c101021] at e_1651_18
  have e_1651_19 := arithEq_of_rows h (row := 1651) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_19
  simp only [← c101022, ← c101023, k2, ← c101024] at e_1651_19
  have hr := e_1651_19
  simp only [e_1651_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1652 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16478)) := by
  have c101025 := (publicBatchWrapper86_copies3157 a h).2.1
  have c101026 := (publicBatchWrapper86_copies3157 a h).2.2.1
  have c101027 := (publicBatchWrapper86_copies3157 a h).2.2.2.1
  have c101028 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.1
  have c101029 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.1
  have c101030 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1652_0 := arithEq_of_rows h (row := 1652) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_0
  simp only [← c101025, ← c101026, ← c101027] at e_1652_0
  have e_1652_1 := arithEq_of_rows h (row := 1652) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_1
  simp only [← c101028, ← c101029, k2, ← c101030] at e_1652_1
  have hr := e_1652_1
  simp only [e_1652_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1652 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16479)) := by
  have c101031 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.1
  have c101032 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.1
  have c101033 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.1
  have c101034 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101035 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101036 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1652_2 := arithEq_of_rows h (row := 1652) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_2
  simp only [← c101031, ← c101032, ← c101033] at e_1652_2
  have e_1652_3 := arithEq_of_rows h (row := 1652) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_3
  simp only [← c101034, ← c101035, k2, ← c101036] at e_1652_3
  have hr := e_1652_3
  simp only [e_1652_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1652 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16480)) := by
  have c101037 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101038 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101039 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101040 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101041 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101042 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1652_4 := arithEq_of_rows h (row := 1652) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_4
  simp only [← c101037, ← c101038, ← c101039] at e_1652_4
  have e_1652_5 := arithEq_of_rows h (row := 1652) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_5
  simp only [← c101040, ← c101041, k2, ← c101042] at e_1652_5
  have hr := e_1652_5
  simp only [e_1652_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1652 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16481)) := by
  have c101043 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101044 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101045 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101046 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101047 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101048 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1652_6 := arithEq_of_rows h (row := 1652) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_6
  simp only [← c101043, ← c101044, ← c101045] at e_1652_6
  have e_1652_7 := arithEq_of_rows h (row := 1652) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_7
  simp only [← c101046, ← c101047, k2, ← c101048] at e_1652_7
  have hr := e_1652_7
  simp only [e_1652_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1652 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16482)) := by
  have c101049 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101050 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101051 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101052 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101053 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101054 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1652_8 := arithEq_of_rows h (row := 1652) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_8
  simp only [← c101049, ← c101050, ← c101051] at e_1652_8
  have e_1652_9 := arithEq_of_rows h (row := 1652) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_9
  simp only [← c101052, ← c101053, k2, ← c101054] at e_1652_9
  have hr := e_1652_9
  simp only [e_1652_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1652 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16483)) := by
  have c101055 := (publicBatchWrapper86_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101056 := (publicBatchWrapper86_copies3158 a h).1
  have c101057 := (publicBatchWrapper86_copies3158 a h).2.1
  have c101058 := (publicBatchWrapper86_copies3158 a h).2.2.1
  have c101059 := (publicBatchWrapper86_copies3158 a h).2.2.2.1
  have c101060 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1652_10 := arithEq_of_rows h (row := 1652) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_10
  simp only [← c101055, ← c101056, ← c101057] at e_1652_10
  have e_1652_11 := arithEq_of_rows h (row := 1652) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_11
  simp only [← c101058, ← c101059, k2, ← c101060] at e_1652_11
  have hr := e_1652_11
  simp only [e_1652_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1652 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16484)) := by
  have c101061 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.1
  have c101062 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.1
  have c101063 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.1
  have c101064 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.1
  have c101065 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.1
  have c101066 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1652_12 := arithEq_of_rows h (row := 1652) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_12
  simp only [← c101061, ← c101062, ← c101063] at e_1652_12
  have e_1652_13 := arithEq_of_rows h (row := 1652) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_13
  simp only [← c101064, ← c101065, k2, ← c101066] at e_1652_13
  have hr := e_1652_13
  simp only [e_1652_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1652 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16485)) := by
  have c101067 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101068 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101069 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101070 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101071 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101072 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1652_14 := arithEq_of_rows h (row := 1652) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_14
  simp only [← c101067, ← c101068, ← c101069] at e_1652_14
  have e_1652_15 := arithEq_of_rows h (row := 1652) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_15
  simp only [← c101070, ← c101071, k2, ← c101072] at e_1652_15
  have hr := e_1652_15
  simp only [e_1652_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1652 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16486)) := by
  have c101073 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101074 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101075 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101076 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101077 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101078 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1652_16 := arithEq_of_rows h (row := 1652) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_16
  simp only [← c101073, ← c101074, ← c101075] at e_1652_16
  have e_1652_17 := arithEq_of_rows h (row := 1652) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_17
  simp only [← c101076, ← c101077, k2, ← c101078] at e_1652_17
  have hr := e_1652_17
  simp only [e_1652_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1652 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16487)) := by
  have c101079 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101080 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101081 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101082 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101083 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101084 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1652_18 := arithEq_of_rows h (row := 1652) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_18
  simp only [← c101079, ← c101080, ← c101081] at e_1652_18
  have e_1652_19 := arithEq_of_rows h (row := 1652) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_19
  simp only [← c101082, ← c101083, k2, ← c101084] at e_1652_19
  have hr := e_1652_19
  simp only [e_1652_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1653 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16488)) := by
  have c101085 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101086 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101087 := (publicBatchWrapper86_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101088 := (publicBatchWrapper86_copies3159 a h).1
  have c101089 := (publicBatchWrapper86_copies3159 a h).2.1
  have c101090 := (publicBatchWrapper86_copies3159 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1653_0 := arithEq_of_rows h (row := 1653) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_0
  simp only [← c101085, ← c101086, ← c101087] at e_1653_0
  have e_1653_1 := arithEq_of_rows h (row := 1653) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_1
  simp only [← c101088, ← c101089, k2, ← c101090] at e_1653_1
  have hr := e_1653_1
  simp only [e_1653_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1653 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16489)) := by
  have c101091 := (publicBatchWrapper86_copies3159 a h).2.2.2.1
  have c101092 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.1
  have c101093 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.1
  have c101094 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.1
  have c101095 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.1
  have c101096 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1653_2 := arithEq_of_rows h (row := 1653) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_2
  simp only [← c101091, ← c101092, ← c101093] at e_1653_2
  have e_1653_3 := arithEq_of_rows h (row := 1653) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_3
  simp only [← c101094, ← c101095, k2, ← c101096] at e_1653_3
  have hr := e_1653_3
  simp only [e_1653_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1653 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16490)) := by
  have c101097 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.1
  have c101098 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101099 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101100 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101101 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101102 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1653_4 := arithEq_of_rows h (row := 1653) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_4
  simp only [← c101097, ← c101098, ← c101099] at e_1653_4
  have e_1653_5 := arithEq_of_rows h (row := 1653) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_5
  simp only [← c101100, ← c101101, k2, ← c101102] at e_1653_5
  have hr := e_1653_5
  simp only [e_1653_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1653 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16491)) := by
  have c101103 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101104 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101105 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101106 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101107 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101108 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1653_6 := arithEq_of_rows h (row := 1653) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_6
  simp only [← c101103, ← c101104, ← c101105] at e_1653_6
  have e_1653_7 := arithEq_of_rows h (row := 1653) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_7
  simp only [← c101106, ← c101107, k2, ← c101108] at e_1653_7
  have hr := e_1653_7
  simp only [e_1653_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1653 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16492)) := by
  have c101109 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101110 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101111 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101112 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101113 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101114 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1653_8 := arithEq_of_rows h (row := 1653) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_8
  simp only [← c101109, ← c101110, ← c101111] at e_1653_8
  have e_1653_9 := arithEq_of_rows h (row := 1653) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_9
  simp only [← c101112, ← c101113, k2, ← c101114] at e_1653_9
  have hr := e_1653_9
  simp only [e_1653_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1653 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16493)) := by
  have c101115 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101116 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101117 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101118 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101119 := (publicBatchWrapper86_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101120 := (publicBatchWrapper86_copies3160 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1653_10 := arithEq_of_rows h (row := 1653) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_10
  simp only [← c101115, ← c101116, ← c101117] at e_1653_10
  have e_1653_11 := arithEq_of_rows h (row := 1653) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_11
  simp only [← c101118, ← c101119, k2, ← c101120] at e_1653_11
  have hr := e_1653_11
  simp only [e_1653_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1653 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16494)) := by
  have c101121 := (publicBatchWrapper86_copies3160 a h).2.1
  have c101122 := (publicBatchWrapper86_copies3160 a h).2.2.1
  have c101123 := (publicBatchWrapper86_copies3160 a h).2.2.2.1
  have c101124 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.1
  have c101125 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.1
  have c101126 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1653_12 := arithEq_of_rows h (row := 1653) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_12
  simp only [← c101121, ← c101122, ← c101123] at e_1653_12
  have e_1653_13 := arithEq_of_rows h (row := 1653) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_13
  simp only [← c101124, ← c101125, k2, ← c101126] at e_1653_13
  have hr := e_1653_13
  simp only [e_1653_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1653 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16495)) := by
  have c101127 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.1
  have c101128 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.1
  have c101129 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.1
  have c101130 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101131 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101132 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1653_14 := arithEq_of_rows h (row := 1653) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_14
  simp only [← c101127, ← c101128, ← c101129] at e_1653_14
  have e_1653_15 := arithEq_of_rows h (row := 1653) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_15
  simp only [← c101130, ← c101131, k2, ← c101132] at e_1653_15
  have hr := e_1653_15
  simp only [e_1653_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1653 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16496)) := by
  have c101133 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101134 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101135 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101136 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101137 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101138 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1653_16 := arithEq_of_rows h (row := 1653) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_16
  simp only [← c101133, ← c101134, ← c101135] at e_1653_16
  have e_1653_17 := arithEq_of_rows h (row := 1653) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_17
  simp only [← c101136, ← c101137, k2, ← c101138] at e_1653_17
  have hr := e_1653_17
  simp only [e_1653_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1653 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16497)) := by
  have c101139 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101140 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101141 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101142 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101143 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101144 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1653_18 := arithEq_of_rows h (row := 1653) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_18
  simp only [← c101139, ← c101140, ← c101141] at e_1653_18
  have e_1653_19 := arithEq_of_rows h (row := 1653) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_19
  simp only [← c101142, ← c101143, k2, ← c101144] at e_1653_19
  have hr := e_1653_19
  simp only [e_1653_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1654 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16498)) := by
  have c101145 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101146 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101147 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101148 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101149 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101150 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1654_0 := arithEq_of_rows h (row := 1654) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_0
  simp only [← c101145, ← c101146, ← c101147] at e_1654_0
  have e_1654_1 := arithEq_of_rows h (row := 1654) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_1
  simp only [← c101148, ← c101149, k2, ← c101150] at e_1654_1
  have hr := e_1654_1
  simp only [e_1654_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1654 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16499)) := by
  have c101151 := (publicBatchWrapper86_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101152 := (publicBatchWrapper86_copies3161 a h).1
  have c101153 := (publicBatchWrapper86_copies3161 a h).2.1
  have c101154 := (publicBatchWrapper86_copies3161 a h).2.2.1
  have c101155 := (publicBatchWrapper86_copies3161 a h).2.2.2.1
  have c101156 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1654_2 := arithEq_of_rows h (row := 1654) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_2
  simp only [← c101151, ← c101152, ← c101153] at e_1654_2
  have e_1654_3 := arithEq_of_rows h (row := 1654) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_3
  simp only [← c101154, ← c101155, k2, ← c101156] at e_1654_3
  have hr := e_1654_3
  simp only [e_1654_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1654 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16500)) := by
  have c101157 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.1
  have c101158 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.1
  have c101159 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.1
  have c101160 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.1
  have c101161 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.1
  have c101162 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1654_4 := arithEq_of_rows h (row := 1654) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_4
  simp only [← c101157, ← c101158, ← c101159] at e_1654_4
  have e_1654_5 := arithEq_of_rows h (row := 1654) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_5
  simp only [← c101160, ← c101161, k2, ← c101162] at e_1654_5
  have hr := e_1654_5
  simp only [e_1654_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1654 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16501)) := by
  have c101163 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101164 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101165 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101166 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101167 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101168 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1654_6 := arithEq_of_rows h (row := 1654) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_6
  simp only [← c101163, ← c101164, ← c101165] at e_1654_6
  have e_1654_7 := arithEq_of_rows h (row := 1654) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_7
  simp only [← c101166, ← c101167, k2, ← c101168] at e_1654_7
  have hr := e_1654_7
  simp only [e_1654_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1654 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16502)) := by
  have c101169 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101170 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101171 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101172 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101173 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101174 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1654_8 := arithEq_of_rows h (row := 1654) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_8
  simp only [← c101169, ← c101170, ← c101171] at e_1654_8
  have e_1654_9 := arithEq_of_rows h (row := 1654) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_9
  simp only [← c101172, ← c101173, k2, ← c101174] at e_1654_9
  have hr := e_1654_9
  simp only [e_1654_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1654 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16503)) := by
  have c101175 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101176 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101177 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101178 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101179 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101180 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1654_10 := arithEq_of_rows h (row := 1654) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_10
  simp only [← c101175, ← c101176, ← c101177] at e_1654_10
  have e_1654_11 := arithEq_of_rows h (row := 1654) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_11
  simp only [← c101178, ← c101179, k2, ← c101180] at e_1654_11
  have hr := e_1654_11
  simp only [e_1654_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1654 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16504)) := by
  have c101181 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101182 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101183 := (publicBatchWrapper86_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101184 := (publicBatchWrapper86_copies3162 a h).1
  have c101185 := (publicBatchWrapper86_copies3162 a h).2.1
  have c101186 := (publicBatchWrapper86_copies3162 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1654_12 := arithEq_of_rows h (row := 1654) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_12
  simp only [← c101181, ← c101182, ← c101183] at e_1654_12
  have e_1654_13 := arithEq_of_rows h (row := 1654) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_13
  simp only [← c101184, ← c101185, k2, ← c101186] at e_1654_13
  have hr := e_1654_13
  simp only [e_1654_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1654 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16505)) := by
  have c101187 := (publicBatchWrapper86_copies3162 a h).2.2.2.1
  have c101188 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.1
  have c101189 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.1
  have c101190 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.1
  have c101191 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.1
  have c101192 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1654_14 := arithEq_of_rows h (row := 1654) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_14
  simp only [← c101187, ← c101188, ← c101189] at e_1654_14
  have e_1654_15 := arithEq_of_rows h (row := 1654) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_15
  simp only [← c101190, ← c101191, k2, ← c101192] at e_1654_15
  have hr := e_1654_15
  simp only [e_1654_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1654 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16506)) := by
  have c101193 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.1
  have c101194 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101195 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101196 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101197 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101198 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1654_16 := arithEq_of_rows h (row := 1654) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_16
  simp only [← c101193, ← c101194, ← c101195] at e_1654_16
  have e_1654_17 := arithEq_of_rows h (row := 1654) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_17
  simp only [← c101196, ← c101197, k2, ← c101198] at e_1654_17
  have hr := e_1654_17
  simp only [e_1654_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1654 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16507)) := by
  have c101199 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101200 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101201 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101202 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101203 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101204 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1654_18 := arithEq_of_rows h (row := 1654) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_18
  simp only [← c101199, ← c101200, ← c101201] at e_1654_18
  have e_1654_19 := arithEq_of_rows h (row := 1654) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_19
  simp only [← c101202, ← c101203, k2, ← c101204] at e_1654_19
  have hr := e_1654_19
  simp only [e_1654_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1655 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16508)) := by
  have c101205 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101206 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101207 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101208 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101209 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101210 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1655_0 := arithEq_of_rows h (row := 1655) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_0
  simp only [← c101205, ← c101206, ← c101207] at e_1655_0
  have e_1655_1 := arithEq_of_rows h (row := 1655) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_1
  simp only [← c101208, ← c101209, k2, ← c101210] at e_1655_1
  have hr := e_1655_1
  simp only [e_1655_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1655 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16509)) := by
  have c101211 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101212 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101213 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101214 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101215 := (publicBatchWrapper86_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101216 := (publicBatchWrapper86_copies3163 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1655_2 := arithEq_of_rows h (row := 1655) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_2
  simp only [← c101211, ← c101212, ← c101213] at e_1655_2
  have e_1655_3 := arithEq_of_rows h (row := 1655) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_3
  simp only [← c101214, ← c101215, k2, ← c101216] at e_1655_3
  have hr := e_1655_3
  simp only [e_1655_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1655 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16510)) := by
  have c101217 := (publicBatchWrapper86_copies3163 a h).2.1
  have c101218 := (publicBatchWrapper86_copies3163 a h).2.2.1
  have c101219 := (publicBatchWrapper86_copies3163 a h).2.2.2.1
  have c101220 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.1
  have c101221 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.1
  have c101222 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1655_4 := arithEq_of_rows h (row := 1655) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_4
  simp only [← c101217, ← c101218, ← c101219] at e_1655_4
  have e_1655_5 := arithEq_of_rows h (row := 1655) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_5
  simp only [← c101220, ← c101221, k2, ← c101222] at e_1655_5
  have hr := e_1655_5
  simp only [e_1655_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1655 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16511)) := by
  have c101223 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.1
  have c101224 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.1
  have c101225 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.1
  have c101226 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101227 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101228 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1655_6 := arithEq_of_rows h (row := 1655) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_6
  simp only [← c101223, ← c101224, ← c101225] at e_1655_6
  have e_1655_7 := arithEq_of_rows h (row := 1655) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_7
  simp only [← c101226, ← c101227, k2, ← c101228] at e_1655_7
  have hr := e_1655_7
  simp only [e_1655_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1655 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16512)) := by
  have c101229 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101230 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101231 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101232 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101233 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101234 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1655_8 := arithEq_of_rows h (row := 1655) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_8
  simp only [← c101229, ← c101230, ← c101231] at e_1655_8
  have e_1655_9 := arithEq_of_rows h (row := 1655) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_9
  simp only [← c101232, ← c101233, k2, ← c101234] at e_1655_9
  have hr := e_1655_9
  simp only [e_1655_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1655 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16513)) := by
  have c101235 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101236 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101237 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101238 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101239 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101240 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1655_10 := arithEq_of_rows h (row := 1655) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_10
  simp only [← c101235, ← c101236, ← c101237] at e_1655_10
  have e_1655_11 := arithEq_of_rows h (row := 1655) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_11
  simp only [← c101238, ← c101239, k2, ← c101240] at e_1655_11
  have hr := e_1655_11
  simp only [e_1655_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1655 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16514)) := by
  have c101241 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101242 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101243 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101244 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101245 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101246 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1655_12 := arithEq_of_rows h (row := 1655) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_12
  simp only [← c101241, ← c101242, ← c101243] at e_1655_12
  have e_1655_13 := arithEq_of_rows h (row := 1655) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_13
  simp only [← c101244, ← c101245, k2, ← c101246] at e_1655_13
  have hr := e_1655_13
  simp only [e_1655_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1655 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16515)) := by
  have c101247 := (publicBatchWrapper86_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101248 := (publicBatchWrapper86_copies3164 a h).1
  have c101249 := (publicBatchWrapper86_copies3164 a h).2.1
  have c101250 := (publicBatchWrapper86_copies3164 a h).2.2.1
  have c101251 := (publicBatchWrapper86_copies3164 a h).2.2.2.1
  have c101252 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1655_14 := arithEq_of_rows h (row := 1655) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_14
  simp only [← c101247, ← c101248, ← c101249] at e_1655_14
  have e_1655_15 := arithEq_of_rows h (row := 1655) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_15
  simp only [← c101250, ← c101251, k2, ← c101252] at e_1655_15
  have hr := e_1655_15
  simp only [e_1655_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1655 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16516)) := by
  have c101253 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.1
  have c101254 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.1
  have c101255 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.1
  have c101256 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.1
  have c101257 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.1
  have c101258 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1655_16 := arithEq_of_rows h (row := 1655) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_16
  simp only [← c101253, ← c101254, ← c101255] at e_1655_16
  have e_1655_17 := arithEq_of_rows h (row := 1655) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_17
  simp only [← c101256, ← c101257, k2, ← c101258] at e_1655_17
  have hr := e_1655_17
  simp only [e_1655_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1655 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16517)) := by
  have c101259 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101260 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101261 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101262 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101263 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101264 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1655_18 := arithEq_of_rows h (row := 1655) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_18
  simp only [← c101259, ← c101260, ← c101261] at e_1655_18
  have e_1655_19 := arithEq_of_rows h (row := 1655) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_19
  simp only [← c101262, ← c101263, k2, ← c101264] at e_1655_19
  have hr := e_1655_19
  simp only [e_1655_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1656 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16518)) := by
  have c101265 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101266 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101267 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101268 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101269 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101270 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1656_0 := arithEq_of_rows h (row := 1656) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_0
  simp only [← c101265, ← c101266, ← c101267] at e_1656_0
  have e_1656_1 := arithEq_of_rows h (row := 1656) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_1
  simp only [← c101268, ← c101269, k2, ← c101270] at e_1656_1
  have hr := e_1656_1
  simp only [e_1656_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1656 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16519)) := by
  have c101271 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101272 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101273 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101274 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101275 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101276 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1656_2 := arithEq_of_rows h (row := 1656) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_2
  simp only [← c101271, ← c101272, ← c101273] at e_1656_2
  have e_1656_3 := arithEq_of_rows h (row := 1656) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_3
  simp only [← c101274, ← c101275, k2, ← c101276] at e_1656_3
  have hr := e_1656_3
  simp only [e_1656_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1656 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16520)) := by
  have c101277 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101278 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101279 := (publicBatchWrapper86_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101280 := (publicBatchWrapper86_copies3165 a h).1
  have c101281 := (publicBatchWrapper86_copies3165 a h).2.1
  have c101282 := (publicBatchWrapper86_copies3165 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1656_4 := arithEq_of_rows h (row := 1656) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_4
  simp only [← c101277, ← c101278, ← c101279] at e_1656_4
  have e_1656_5 := arithEq_of_rows h (row := 1656) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_5
  simp only [← c101280, ← c101281, k2, ← c101282] at e_1656_5
  have hr := e_1656_5
  simp only [e_1656_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1656 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16521)) := by
  have c101283 := (publicBatchWrapper86_copies3165 a h).2.2.2.1
  have c101284 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.1
  have c101285 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.1
  have c101286 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.1
  have c101287 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.1
  have c101288 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1656_6 := arithEq_of_rows h (row := 1656) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_6
  simp only [← c101283, ← c101284, ← c101285] at e_1656_6
  have e_1656_7 := arithEq_of_rows h (row := 1656) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_7
  simp only [← c101286, ← c101287, k2, ← c101288] at e_1656_7
  have hr := e_1656_7
  simp only [e_1656_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1656 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16522)) := by
  have c101289 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.1
  have c101290 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101291 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101292 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101293 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101294 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1656_8 := arithEq_of_rows h (row := 1656) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_8
  simp only [← c101289, ← c101290, ← c101291] at e_1656_8
  have e_1656_9 := arithEq_of_rows h (row := 1656) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_9
  simp only [← c101292, ← c101293, k2, ← c101294] at e_1656_9
  have hr := e_1656_9
  simp only [e_1656_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1656 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16523)) := by
  have c101295 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101296 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101297 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101298 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101299 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101300 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1656_10 := arithEq_of_rows h (row := 1656) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_10
  simp only [← c101295, ← c101296, ← c101297] at e_1656_10
  have e_1656_11 := arithEq_of_rows h (row := 1656) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_11
  simp only [← c101298, ← c101299, k2, ← c101300] at e_1656_11
  have hr := e_1656_11
  simp only [e_1656_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1656 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16524)) := by
  have c101301 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101302 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101303 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101304 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101305 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101306 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1656_12 := arithEq_of_rows h (row := 1656) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_12
  simp only [← c101301, ← c101302, ← c101303] at e_1656_12
  have e_1656_13 := arithEq_of_rows h (row := 1656) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_13
  simp only [← c101304, ← c101305, k2, ← c101306] at e_1656_13
  have hr := e_1656_13
  simp only [e_1656_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1656 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16525)) := by
  have c101307 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101308 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101309 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101310 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101311 := (publicBatchWrapper86_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101312 := (publicBatchWrapper86_copies3166 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1656_14 := arithEq_of_rows h (row := 1656) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_14
  simp only [← c101307, ← c101308, ← c101309] at e_1656_14
  have e_1656_15 := arithEq_of_rows h (row := 1656) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_15
  simp only [← c101310, ← c101311, k2, ← c101312] at e_1656_15
  have hr := e_1656_15
  simp only [e_1656_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1656 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16526)) := by
  have c101313 := (publicBatchWrapper86_copies3166 a h).2.1
  have c101314 := (publicBatchWrapper86_copies3166 a h).2.2.1
  have c101315 := (publicBatchWrapper86_copies3166 a h).2.2.2.1
  have c101316 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.1
  have c101317 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.1
  have c101318 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1656_16 := arithEq_of_rows h (row := 1656) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_16
  simp only [← c101313, ← c101314, ← c101315] at e_1656_16
  have e_1656_17 := arithEq_of_rows h (row := 1656) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_17
  simp only [← c101316, ← c101317, k2, ← c101318] at e_1656_17
  have hr := e_1656_17
  simp only [e_1656_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1656 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16527)) := by
  have c101319 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.1
  have c101320 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.1
  have c101321 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.1
  have c101322 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101323 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101324 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1656_18 := arithEq_of_rows h (row := 1656) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_18
  simp only [← c101319, ← c101320, ← c101321] at e_1656_18
  have e_1656_19 := arithEq_of_rows h (row := 1656) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_19
  simp only [← c101322, ← c101323, k2, ← c101324] at e_1656_19
  have hr := e_1656_19
  simp only [e_1656_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1657 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16528)) := by
  have c101325 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101326 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101327 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101328 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101329 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101330 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1657_0 := arithEq_of_rows h (row := 1657) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_0
  simp only [← c101325, ← c101326, ← c101327] at e_1657_0
  have e_1657_1 := arithEq_of_rows h (row := 1657) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_1
  simp only [← c101328, ← c101329, k2, ← c101330] at e_1657_1
  have hr := e_1657_1
  simp only [e_1657_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1657 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16529)) := by
  have c101331 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101332 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101333 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101334 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101335 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101336 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1657_2 := arithEq_of_rows h (row := 1657) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_2
  simp only [← c101331, ← c101332, ← c101333] at e_1657_2
  have e_1657_3 := arithEq_of_rows h (row := 1657) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_3
  simp only [← c101334, ← c101335, k2, ← c101336] at e_1657_3
  have hr := e_1657_3
  simp only [e_1657_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1657 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16530)) := by
  have c101337 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101338 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101339 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101340 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101341 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101342 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1657_4 := arithEq_of_rows h (row := 1657) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_4
  simp only [← c101337, ← c101338, ← c101339] at e_1657_4
  have e_1657_5 := arithEq_of_rows h (row := 1657) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_5
  simp only [← c101340, ← c101341, k2, ← c101342] at e_1657_5
  have hr := e_1657_5
  simp only [e_1657_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1657 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16531)) := by
  have c101343 := (publicBatchWrapper86_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101344 := (publicBatchWrapper86_copies3167 a h).1
  have c101345 := (publicBatchWrapper86_copies3167 a h).2.1
  have c101346 := (publicBatchWrapper86_copies3167 a h).2.2.1
  have c101347 := (publicBatchWrapper86_copies3167 a h).2.2.2.1
  have c101348 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1657_6 := arithEq_of_rows h (row := 1657) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_6
  simp only [← c101343, ← c101344, ← c101345] at e_1657_6
  have e_1657_7 := arithEq_of_rows h (row := 1657) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_7
  simp only [← c101346, ← c101347, k2, ← c101348] at e_1657_7
  have hr := e_1657_7
  simp only [e_1657_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1657 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16532)) := by
  have c101349 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.1
  have c101350 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.1
  have c101351 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.1
  have c101352 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.1
  have c101353 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.1
  have c101354 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1657_8 := arithEq_of_rows h (row := 1657) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_8
  simp only [← c101349, ← c101350, ← c101351] at e_1657_8
  have e_1657_9 := arithEq_of_rows h (row := 1657) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_9
  simp only [← c101352, ← c101353, k2, ← c101354] at e_1657_9
  have hr := e_1657_9
  simp only [e_1657_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1657 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16533)) := by
  have c101355 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101356 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101357 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101358 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101359 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101360 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1657_10 := arithEq_of_rows h (row := 1657) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_10
  simp only [← c101355, ← c101356, ← c101357] at e_1657_10
  have e_1657_11 := arithEq_of_rows h (row := 1657) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_11
  simp only [← c101358, ← c101359, k2, ← c101360] at e_1657_11
  have hr := e_1657_11
  simp only [e_1657_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1657 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16534)) := by
  have c101361 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101362 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101363 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101364 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101365 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101366 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1657_12 := arithEq_of_rows h (row := 1657) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_12
  simp only [← c101361, ← c101362, ← c101363] at e_1657_12
  have e_1657_13 := arithEq_of_rows h (row := 1657) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_13
  simp only [← c101364, ← c101365, k2, ← c101366] at e_1657_13
  have hr := e_1657_13
  simp only [e_1657_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1657 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16535)) := by
  have c101367 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101368 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101369 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101370 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101371 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101372 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1657_14 := arithEq_of_rows h (row := 1657) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_14
  simp only [← c101367, ← c101368, ← c101369] at e_1657_14
  have e_1657_15 := arithEq_of_rows h (row := 1657) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_15
  simp only [← c101370, ← c101371, k2, ← c101372] at e_1657_15
  have hr := e_1657_15
  simp only [e_1657_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1657 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16536)) := by
  have c101373 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101374 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101375 := (publicBatchWrapper86_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101376 := (publicBatchWrapper86_copies3168 a h).1
  have c101377 := (publicBatchWrapper86_copies3168 a h).2.1
  have c101378 := (publicBatchWrapper86_copies3168 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1657_16 := arithEq_of_rows h (row := 1657) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_16
  simp only [← c101373, ← c101374, ← c101375] at e_1657_16
  have e_1657_17 := arithEq_of_rows h (row := 1657) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_17
  simp only [← c101376, ← c101377, k2, ← c101378] at e_1657_17
  have hr := e_1657_17
  simp only [e_1657_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1657 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16537)) := by
  have c101379 := (publicBatchWrapper86_copies3168 a h).2.2.2.1
  have c101380 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.1
  have c101381 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.1
  have c101382 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.1
  have c101383 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.1
  have c101384 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1657_18 := arithEq_of_rows h (row := 1657) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_18
  simp only [← c101379, ← c101380, ← c101381] at e_1657_18
  have e_1657_19 := arithEq_of_rows h (row := 1657) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_19
  simp only [← c101382, ← c101383, k2, ← c101384] at e_1657_19
  have hr := e_1657_19
  simp only [e_1657_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1658 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16538)) := by
  have c101385 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.1
  have c101386 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101387 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101388 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101389 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101390 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1658_0 := arithEq_of_rows h (row := 1658) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_0
  simp only [← c101385, ← c101386, ← c101387] at e_1658_0
  have e_1658_1 := arithEq_of_rows h (row := 1658) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_1
  simp only [← c101388, ← c101389, k2, ← c101390] at e_1658_1
  have hr := e_1658_1
  simp only [e_1658_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1658 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16539)) := by
  have c101391 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101392 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101393 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101394 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101395 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101396 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1658_2 := arithEq_of_rows h (row := 1658) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_2
  simp only [← c101391, ← c101392, ← c101393] at e_1658_2
  have e_1658_3 := arithEq_of_rows h (row := 1658) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_3
  simp only [← c101394, ← c101395, k2, ← c101396] at e_1658_3
  have hr := e_1658_3
  simp only [e_1658_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1658 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16540)) := by
  have c101397 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101398 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101399 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101400 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101401 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101402 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1658_4 := arithEq_of_rows h (row := 1658) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_4
  simp only [← c101397, ← c101398, ← c101399] at e_1658_4
  have e_1658_5 := arithEq_of_rows h (row := 1658) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_5
  simp only [← c101400, ← c101401, k2, ← c101402] at e_1658_5
  have hr := e_1658_5
  simp only [e_1658_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1658 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16541)) := by
  have c101403 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101404 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101405 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101406 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101407 := (publicBatchWrapper86_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101408 := (publicBatchWrapper86_copies3169 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1658_6 := arithEq_of_rows h (row := 1658) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_6
  simp only [← c101403, ← c101404, ← c101405] at e_1658_6
  have e_1658_7 := arithEq_of_rows h (row := 1658) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_7
  simp only [← c101406, ← c101407, k2, ← c101408] at e_1658_7
  have hr := e_1658_7
  simp only [e_1658_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1658 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16542)) := by
  have c101409 := (publicBatchWrapper86_copies3169 a h).2.1
  have c101410 := (publicBatchWrapper86_copies3169 a h).2.2.1
  have c101411 := (publicBatchWrapper86_copies3169 a h).2.2.2.1
  have c101412 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.1
  have c101413 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.1
  have c101414 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1658_8 := arithEq_of_rows h (row := 1658) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_8
  simp only [← c101409, ← c101410, ← c101411] at e_1658_8
  have e_1658_9 := arithEq_of_rows h (row := 1658) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_9
  simp only [← c101412, ← c101413, k2, ← c101414] at e_1658_9
  have hr := e_1658_9
  simp only [e_1658_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1658 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16543)) := by
  have c101415 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.1
  have c101416 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.1
  have c101417 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.1
  have c101418 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101419 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101420 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1658_10 := arithEq_of_rows h (row := 1658) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_10
  simp only [← c101415, ← c101416, ← c101417] at e_1658_10
  have e_1658_11 := arithEq_of_rows h (row := 1658) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_11
  simp only [← c101418, ← c101419, k2, ← c101420] at e_1658_11
  have hr := e_1658_11
  simp only [e_1658_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1658 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16544)) := by
  have c101421 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101422 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101423 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101424 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101425 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101426 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1658_12 := arithEq_of_rows h (row := 1658) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_12
  simp only [← c101421, ← c101422, ← c101423] at e_1658_12
  have e_1658_13 := arithEq_of_rows h (row := 1658) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_13
  simp only [← c101424, ← c101425, k2, ← c101426] at e_1658_13
  have hr := e_1658_13
  simp only [e_1658_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1658 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16545)) := by
  have c101427 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101428 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101429 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101430 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101431 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101432 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1658_14 := arithEq_of_rows h (row := 1658) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_14
  simp only [← c101427, ← c101428, ← c101429] at e_1658_14
  have e_1658_15 := arithEq_of_rows h (row := 1658) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_15
  simp only [← c101430, ← c101431, k2, ← c101432] at e_1658_15
  have hr := e_1658_15
  simp only [e_1658_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1658 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16546)) := by
  have c101433 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101434 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101435 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101436 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101437 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101438 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1658_16 := arithEq_of_rows h (row := 1658) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_16
  simp only [← c101433, ← c101434, ← c101435] at e_1658_16
  have e_1658_17 := arithEq_of_rows h (row := 1658) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_17
  simp only [← c101436, ← c101437, k2, ← c101438] at e_1658_17
  have hr := e_1658_17
  simp only [e_1658_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1658 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16547)) := by
  have c101439 := (publicBatchWrapper86_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101440 := (publicBatchWrapper86_copies3170 a h).1
  have c101441 := (publicBatchWrapper86_copies3170 a h).2.1
  have c101442 := (publicBatchWrapper86_copies3170 a h).2.2.1
  have c101443 := (publicBatchWrapper86_copies3170 a h).2.2.2.1
  have c101444 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1658_18 := arithEq_of_rows h (row := 1658) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_18
  simp only [← c101439, ← c101440, ← c101441] at e_1658_18
  have e_1658_19 := arithEq_of_rows h (row := 1658) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_19
  simp only [← c101442, ← c101443, k2, ← c101444] at e_1658_19
  have hr := e_1658_19
  simp only [e_1658_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1659 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16548)) := by
  have c101445 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.1
  have c101446 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.1
  have c101447 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.1
  have c101448 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.1
  have c101449 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.1
  have c101450 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1659_0 := arithEq_of_rows h (row := 1659) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_0
  simp only [← c101445, ← c101446, ← c101447] at e_1659_0
  have e_1659_1 := arithEq_of_rows h (row := 1659) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_1
  simp only [← c101448, ← c101449, k2, ← c101450] at e_1659_1
  have hr := e_1659_1
  simp only [e_1659_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1659 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16549)) := by
  have c101451 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101452 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101453 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101454 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101455 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101456 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1659_2 := arithEq_of_rows h (row := 1659) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_2
  simp only [← c101451, ← c101452, ← c101453] at e_1659_2
  have e_1659_3 := arithEq_of_rows h (row := 1659) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_3
  simp only [← c101454, ← c101455, k2, ← c101456] at e_1659_3
  have hr := e_1659_3
  simp only [e_1659_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1659 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16550)) := by
  have c101457 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101458 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101459 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101460 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101461 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101462 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1659_4 := arithEq_of_rows h (row := 1659) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_4
  simp only [← c101457, ← c101458, ← c101459] at e_1659_4
  have e_1659_5 := arithEq_of_rows h (row := 1659) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_5
  simp only [← c101460, ← c101461, k2, ← c101462] at e_1659_5
  have hr := e_1659_5
  simp only [e_1659_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1659 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16551)) := by
  have c101463 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101464 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101465 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101466 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101467 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101468 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1659_6 := arithEq_of_rows h (row := 1659) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_6
  simp only [← c101463, ← c101464, ← c101465] at e_1659_6
  have e_1659_7 := arithEq_of_rows h (row := 1659) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_7
  simp only [← c101466, ← c101467, k2, ← c101468] at e_1659_7
  have hr := e_1659_7
  simp only [e_1659_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1659 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16552)) := by
  have c101469 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101470 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101471 := (publicBatchWrapper86_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101472 := (publicBatchWrapper86_copies3171 a h).1
  have c101473 := (publicBatchWrapper86_copies3171 a h).2.1
  have c101474 := (publicBatchWrapper86_copies3171 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1659_8 := arithEq_of_rows h (row := 1659) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_8
  simp only [← c101469, ← c101470, ← c101471] at e_1659_8
  have e_1659_9 := arithEq_of_rows h (row := 1659) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_9
  simp only [← c101472, ← c101473, k2, ← c101474] at e_1659_9
  have hr := e_1659_9
  simp only [e_1659_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1659 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16553)) := by
  have c101475 := (publicBatchWrapper86_copies3171 a h).2.2.2.1
  have c101476 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.1
  have c101477 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.1
  have c101478 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.1
  have c101479 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.1
  have c101480 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1659_10 := arithEq_of_rows h (row := 1659) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_10
  simp only [← c101475, ← c101476, ← c101477] at e_1659_10
  have e_1659_11 := arithEq_of_rows h (row := 1659) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_11
  simp only [← c101478, ← c101479, k2, ← c101480] at e_1659_11
  have hr := e_1659_11
  simp only [e_1659_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1659 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16554)) := by
  have c101481 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.1
  have c101482 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101483 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101484 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101485 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101486 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1659_12 := arithEq_of_rows h (row := 1659) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_12
  simp only [← c101481, ← c101482, ← c101483] at e_1659_12
  have e_1659_13 := arithEq_of_rows h (row := 1659) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_13
  simp only [← c101484, ← c101485, k2, ← c101486] at e_1659_13
  have hr := e_1659_13
  simp only [e_1659_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1659 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16555)) := by
  have c101487 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101488 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101489 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101490 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101491 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101492 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1659_14 := arithEq_of_rows h (row := 1659) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_14
  simp only [← c101487, ← c101488, ← c101489] at e_1659_14
  have e_1659_15 := arithEq_of_rows h (row := 1659) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_15
  simp only [← c101490, ← c101491, k2, ← c101492] at e_1659_15
  have hr := e_1659_15
  simp only [e_1659_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1659 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16556)) := by
  have c101493 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101494 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101495 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101496 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101497 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101498 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1659_16 := arithEq_of_rows h (row := 1659) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_16
  simp only [← c101493, ← c101494, ← c101495] at e_1659_16
  have e_1659_17 := arithEq_of_rows h (row := 1659) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_17
  simp only [← c101496, ← c101497, k2, ← c101498] at e_1659_17
  have hr := e_1659_17
  simp only [e_1659_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1659 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16557)) := by
  have c101499 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101500 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101501 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101502 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101503 := (publicBatchWrapper86_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101504 := (publicBatchWrapper86_copies3172 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1659_18 := arithEq_of_rows h (row := 1659) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_18
  simp only [← c101499, ← c101500, ← c101501] at e_1659_18
  have e_1659_19 := arithEq_of_rows h (row := 1659) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_19
  simp only [← c101502, ← c101503, k2, ← c101504] at e_1659_19
  have hr := e_1659_19
  simp only [e_1659_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1660 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16558)) := by
  have c101505 := (publicBatchWrapper86_copies3172 a h).2.1
  have c101506 := (publicBatchWrapper86_copies3172 a h).2.2.1
  have c101507 := (publicBatchWrapper86_copies3172 a h).2.2.2.1
  have c101508 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.1
  have c101509 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.1
  have c101510 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1660_0 := arithEq_of_rows h (row := 1660) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_0
  simp only [← c101505, ← c101506, ← c101507] at e_1660_0
  have e_1660_1 := arithEq_of_rows h (row := 1660) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_1
  simp only [← c101508, ← c101509, k2, ← c101510] at e_1660_1
  have hr := e_1660_1
  simp only [e_1660_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1660 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16559)) := by
  have c101511 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.1
  have c101512 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.1
  have c101513 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.1
  have c101514 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101515 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101516 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1660_2 := arithEq_of_rows h (row := 1660) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_2
  simp only [← c101511, ← c101512, ← c101513] at e_1660_2
  have e_1660_3 := arithEq_of_rows h (row := 1660) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_3
  simp only [← c101514, ← c101515, k2, ← c101516] at e_1660_3
  have hr := e_1660_3
  simp only [e_1660_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1660 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16560)) := by
  have c101517 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101518 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101519 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101520 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101521 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101522 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1660_4 := arithEq_of_rows h (row := 1660) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_4
  simp only [← c101517, ← c101518, ← c101519] at e_1660_4
  have e_1660_5 := arithEq_of_rows h (row := 1660) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_5
  simp only [← c101520, ← c101521, k2, ← c101522] at e_1660_5
  have hr := e_1660_5
  simp only [e_1660_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1660 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16561)) := by
  have c101523 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101524 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101525 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101526 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101527 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101528 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1660_6 := arithEq_of_rows h (row := 1660) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_6
  simp only [← c101523, ← c101524, ← c101525] at e_1660_6
  have e_1660_7 := arithEq_of_rows h (row := 1660) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_7
  simp only [← c101526, ← c101527, k2, ← c101528] at e_1660_7
  have hr := e_1660_7
  simp only [e_1660_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1660 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16562)) := by
  have c101529 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101530 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101531 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101532 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101533 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101534 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1660_8 := arithEq_of_rows h (row := 1660) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_8
  simp only [← c101529, ← c101530, ← c101531] at e_1660_8
  have e_1660_9 := arithEq_of_rows h (row := 1660) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_9
  simp only [← c101532, ← c101533, k2, ← c101534] at e_1660_9
  have hr := e_1660_9
  simp only [e_1660_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1660 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16563)) := by
  have c101535 := (publicBatchWrapper86_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101536 := (publicBatchWrapper86_copies3173 a h).1
  have c101537 := (publicBatchWrapper86_copies3173 a h).2.1
  have c101538 := (publicBatchWrapper86_copies3173 a h).2.2.1
  have c101539 := (publicBatchWrapper86_copies3173 a h).2.2.2.1
  have c101540 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1660_10 := arithEq_of_rows h (row := 1660) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_10
  simp only [← c101535, ← c101536, ← c101537] at e_1660_10
  have e_1660_11 := arithEq_of_rows h (row := 1660) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_11
  simp only [← c101538, ← c101539, k2, ← c101540] at e_1660_11
  have hr := e_1660_11
  simp only [e_1660_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1660 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16564)) := by
  have c101541 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.1
  have c101542 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.1
  have c101543 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.1
  have c101544 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.1
  have c101545 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.1
  have c101546 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1660_12 := arithEq_of_rows h (row := 1660) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_12
  simp only [← c101541, ← c101542, ← c101543] at e_1660_12
  have e_1660_13 := arithEq_of_rows h (row := 1660) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_13
  simp only [← c101544, ← c101545, k2, ← c101546] at e_1660_13
  have hr := e_1660_13
  simp only [e_1660_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1660 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16565)) := by
  have c101547 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101548 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101549 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101550 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101551 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101552 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1660_14 := arithEq_of_rows h (row := 1660) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_14
  simp only [← c101547, ← c101548, ← c101549] at e_1660_14
  have e_1660_15 := arithEq_of_rows h (row := 1660) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_15
  simp only [← c101550, ← c101551, k2, ← c101552] at e_1660_15
  have hr := e_1660_15
  simp only [e_1660_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1660 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16566)) := by
  have c101553 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101554 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101555 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101556 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101557 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101558 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1660_16 := arithEq_of_rows h (row := 1660) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_16
  simp only [← c101553, ← c101554, ← c101555] at e_1660_16
  have e_1660_17 := arithEq_of_rows h (row := 1660) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_17
  simp only [← c101556, ← c101557, k2, ← c101558] at e_1660_17
  have hr := e_1660_17
  simp only [e_1660_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1660 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16567)) := by
  have c101559 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101560 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101561 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101562 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101563 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101564 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1660_18 := arithEq_of_rows h (row := 1660) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_18
  simp only [← c101559, ← c101560, ← c101561] at e_1660_18
  have e_1660_19 := arithEq_of_rows h (row := 1660) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_19
  simp only [← c101562, ← c101563, k2, ← c101564] at e_1660_19
  have hr := e_1660_19
  simp only [e_1660_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1661 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16568)) := by
  have c101565 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101566 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101567 := (publicBatchWrapper86_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101568 := (publicBatchWrapper86_copies3174 a h).1
  have c101569 := (publicBatchWrapper86_copies3174 a h).2.1
  have c101570 := (publicBatchWrapper86_copies3174 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1661_0 := arithEq_of_rows h (row := 1661) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_0
  simp only [← c101565, ← c101566, ← c101567] at e_1661_0
  have e_1661_1 := arithEq_of_rows h (row := 1661) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_1
  simp only [← c101568, ← c101569, k2, ← c101570] at e_1661_1
  have hr := e_1661_1
  simp only [e_1661_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1661 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16569)) := by
  have c101571 := (publicBatchWrapper86_copies3174 a h).2.2.2.1
  have c101572 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.1
  have c101573 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.1
  have c101574 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.1
  have c101575 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.1
  have c101576 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1661_2 := arithEq_of_rows h (row := 1661) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_2
  simp only [← c101571, ← c101572, ← c101573] at e_1661_2
  have e_1661_3 := arithEq_of_rows h (row := 1661) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_3
  simp only [← c101574, ← c101575, k2, ← c101576] at e_1661_3
  have hr := e_1661_3
  simp only [e_1661_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1661 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16610)) := by
  have c101577 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.1
  have c101578 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101579 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101580 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101581 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101582 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1661_4 := arithEq_of_rows h (row := 1661) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_4
  simp only [← c101577, ← c101578, ← c101579] at e_1661_4
  have e_1661_5 := arithEq_of_rows h (row := 1661) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_5
  simp only [← c101580, ← c101581, k2, ← c101582] at e_1661_5
  have hr := e_1661_5
  simp only [e_1661_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1661 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16611)) := by
  have c101583 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101584 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101585 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101586 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101587 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101588 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1661_6 := arithEq_of_rows h (row := 1661) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_6
  simp only [← c101583, ← c101584, ← c101585] at e_1661_6
  have e_1661_7 := arithEq_of_rows h (row := 1661) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_7
  simp only [← c101586, ← c101587, k2, ← c101588] at e_1661_7
  have hr := e_1661_7
  simp only [e_1661_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1661 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16612)) := by
  have c101589 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101590 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101591 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101592 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101593 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101594 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1661_8 := arithEq_of_rows h (row := 1661) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_8
  simp only [← c101589, ← c101590, ← c101591] at e_1661_8
  have e_1661_9 := arithEq_of_rows h (row := 1661) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_9
  simp only [← c101592, ← c101593, k2, ← c101594] at e_1661_9
  have hr := e_1661_9
  simp only [e_1661_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1661 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16613)) := by
  have c101595 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101596 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101597 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101598 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101599 := (publicBatchWrapper86_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101600 := (publicBatchWrapper86_copies3175 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1661_10 := arithEq_of_rows h (row := 1661) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_10
  simp only [← c101595, ← c101596, ← c101597] at e_1661_10
  have e_1661_11 := arithEq_of_rows h (row := 1661) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_11
  simp only [← c101598, ← c101599, k2, ← c101600] at e_1661_11
  have hr := e_1661_11
  simp only [e_1661_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1661 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16614)) := by
  have c101601 := (publicBatchWrapper86_copies3175 a h).2.1
  have c101602 := (publicBatchWrapper86_copies3175 a h).2.2.1
  have c101603 := (publicBatchWrapper86_copies3175 a h).2.2.2.1
  have c101604 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.1
  have c101605 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.1
  have c101606 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1661_12 := arithEq_of_rows h (row := 1661) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_12
  simp only [← c101601, ← c101602, ← c101603] at e_1661_12
  have e_1661_13 := arithEq_of_rows h (row := 1661) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_13
  simp only [← c101604, ← c101605, k2, ← c101606] at e_1661_13
  have hr := e_1661_13
  simp only [e_1661_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1661 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16615)) := by
  have c101607 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.1
  have c101608 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.1
  have c101609 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.1
  have c101610 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101611 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101612 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1661_14 := arithEq_of_rows h (row := 1661) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_14
  simp only [← c101607, ← c101608, ← c101609] at e_1661_14
  have e_1661_15 := arithEq_of_rows h (row := 1661) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_15
  simp only [← c101610, ← c101611, k2, ← c101612] at e_1661_15
  have hr := e_1661_15
  simp only [e_1661_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1661 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16616)) := by
  have c101613 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101614 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101615 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101616 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101617 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101618 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1661_16 := arithEq_of_rows h (row := 1661) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_16
  simp only [← c101613, ← c101614, ← c101615] at e_1661_16
  have e_1661_17 := arithEq_of_rows h (row := 1661) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_17
  simp only [← c101616, ← c101617, k2, ← c101618] at e_1661_17
  have hr := e_1661_17
  simp only [e_1661_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1661 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16617)) := by
  have c101619 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101620 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101621 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101622 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101623 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101624 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1661_18 := arithEq_of_rows h (row := 1661) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_18
  simp only [← c101619, ← c101620, ← c101621] at e_1661_18
  have e_1661_19 := arithEq_of_rows h (row := 1661) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_19
  simp only [← c101622, ← c101623, k2, ← c101624] at e_1661_19
  have hr := e_1661_19
  simp only [e_1661_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1662 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16618)) := by
  have c101625 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101626 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101627 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101628 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101629 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101630 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1662_0 := arithEq_of_rows h (row := 1662) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_0
  simp only [← c101625, ← c101626, ← c101627] at e_1662_0
  have e_1662_1 := arithEq_of_rows h (row := 1662) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_1
  simp only [← c101628, ← c101629, k2, ← c101630] at e_1662_1
  have hr := e_1662_1
  simp only [e_1662_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1662 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16619)) := by
  have c101631 := (publicBatchWrapper86_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101632 := (publicBatchWrapper86_copies3176 a h).1
  have c101633 := (publicBatchWrapper86_copies3176 a h).2.1
  have c101634 := (publicBatchWrapper86_copies3176 a h).2.2.1
  have c101635 := (publicBatchWrapper86_copies3176 a h).2.2.2.1
  have c101636 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1662_2 := arithEq_of_rows h (row := 1662) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_2
  simp only [← c101631, ← c101632, ← c101633] at e_1662_2
  have e_1662_3 := arithEq_of_rows h (row := 1662) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_3
  simp only [← c101634, ← c101635, k2, ← c101636] at e_1662_3
  have hr := e_1662_3
  simp only [e_1662_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1662 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16620)) := by
  have c101637 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.1
  have c101638 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.1
  have c101639 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.1
  have c101640 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.1
  have c101641 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.1
  have c101642 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1662_4 := arithEq_of_rows h (row := 1662) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_4
  simp only [← c101637, ← c101638, ← c101639] at e_1662_4
  have e_1662_5 := arithEq_of_rows h (row := 1662) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_5
  simp only [← c101640, ← c101641, k2, ← c101642] at e_1662_5
  have hr := e_1662_5
  simp only [e_1662_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1662 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16621)) := by
  have c101643 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101644 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101645 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101646 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101647 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101648 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1662_6 := arithEq_of_rows h (row := 1662) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_6
  simp only [← c101643, ← c101644, ← c101645] at e_1662_6
  have e_1662_7 := arithEq_of_rows h (row := 1662) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_7
  simp only [← c101646, ← c101647, k2, ← c101648] at e_1662_7
  have hr := e_1662_7
  simp only [e_1662_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1662 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16622)) := by
  have c101649 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101650 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101651 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101652 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101653 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101654 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1662_8 := arithEq_of_rows h (row := 1662) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_8
  simp only [← c101649, ← c101650, ← c101651] at e_1662_8
  have e_1662_9 := arithEq_of_rows h (row := 1662) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_9
  simp only [← c101652, ← c101653, k2, ← c101654] at e_1662_9
  have hr := e_1662_9
  simp only [e_1662_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1662 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16623)) := by
  have c101655 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101656 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101657 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101658 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101659 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101660 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1662_10 := arithEq_of_rows h (row := 1662) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_10
  simp only [← c101655, ← c101656, ← c101657] at e_1662_10
  have e_1662_11 := arithEq_of_rows h (row := 1662) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_11
  simp only [← c101658, ← c101659, k2, ← c101660] at e_1662_11
  have hr := e_1662_11
  simp only [e_1662_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1662 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16624)) := by
  have c101661 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101662 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101663 := (publicBatchWrapper86_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101664 := (publicBatchWrapper86_copies3177 a h).1
  have c101665 := (publicBatchWrapper86_copies3177 a h).2.1
  have c101666 := (publicBatchWrapper86_copies3177 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1662_12 := arithEq_of_rows h (row := 1662) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_12
  simp only [← c101661, ← c101662, ← c101663] at e_1662_12
  have e_1662_13 := arithEq_of_rows h (row := 1662) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_13
  simp only [← c101664, ← c101665, k2, ← c101666] at e_1662_13
  have hr := e_1662_13
  simp only [e_1662_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1662 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16625)) := by
  have c101667 := (publicBatchWrapper86_copies3177 a h).2.2.2.1
  have c101668 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.1
  have c101669 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.1
  have c101670 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.1
  have c101671 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.1
  have c101672 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1662_14 := arithEq_of_rows h (row := 1662) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_14
  simp only [← c101667, ← c101668, ← c101669] at e_1662_14
  have e_1662_15 := arithEq_of_rows h (row := 1662) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_15
  simp only [← c101670, ← c101671, k2, ← c101672] at e_1662_15
  have hr := e_1662_15
  simp only [e_1662_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1662 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16626)) := by
  have c101673 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.1
  have c101674 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101675 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101676 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101677 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101678 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1662_16 := arithEq_of_rows h (row := 1662) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_16
  simp only [← c101673, ← c101674, ← c101675] at e_1662_16
  have e_1662_17 := arithEq_of_rows h (row := 1662) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_17
  simp only [← c101676, ← c101677, k2, ← c101678] at e_1662_17
  have hr := e_1662_17
  simp only [e_1662_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1662 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16627)) := by
  have c101679 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101680 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101681 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101682 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101683 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101684 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1662_18 := arithEq_of_rows h (row := 1662) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_18
  simp only [← c101679, ← c101680, ← c101681] at e_1662_18
  have e_1662_19 := arithEq_of_rows h (row := 1662) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_19
  simp only [← c101682, ← c101683, k2, ← c101684] at e_1662_19
  have hr := e_1662_19
  simp only [e_1662_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1663 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16628)) := by
  have c101685 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101686 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101687 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101688 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101689 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101690 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1663_0 := arithEq_of_rows h (row := 1663) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_0
  simp only [← c101685, ← c101686, ← c101687] at e_1663_0
  have e_1663_1 := arithEq_of_rows h (row := 1663) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_1
  simp only [← c101688, ← c101689, k2, ← c101690] at e_1663_1
  have hr := e_1663_1
  simp only [e_1663_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1663 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16629)) := by
  have c101691 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101692 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101693 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101694 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101695 := (publicBatchWrapper86_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101696 := (publicBatchWrapper86_copies3178 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1663_2 := arithEq_of_rows h (row := 1663) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_2
  simp only [← c101691, ← c101692, ← c101693] at e_1663_2
  have e_1663_3 := arithEq_of_rows h (row := 1663) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_3
  simp only [← c101694, ← c101695, k2, ← c101696] at e_1663_3
  have hr := e_1663_3
  simp only [e_1663_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1663 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16630)) := by
  have c101697 := (publicBatchWrapper86_copies3178 a h).2.1
  have c101698 := (publicBatchWrapper86_copies3178 a h).2.2.1
  have c101699 := (publicBatchWrapper86_copies3178 a h).2.2.2.1
  have c101700 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.1
  have c101701 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.1
  have c101702 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1663_4 := arithEq_of_rows h (row := 1663) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_4
  simp only [← c101697, ← c101698, ← c101699] at e_1663_4
  have e_1663_5 := arithEq_of_rows h (row := 1663) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_5
  simp only [← c101700, ← c101701, k2, ← c101702] at e_1663_5
  have hr := e_1663_5
  simp only [e_1663_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1663 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16631)) := by
  have c101703 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.1
  have c101704 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.1
  have c101705 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.1
  have c101706 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101707 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101708 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1663_6 := arithEq_of_rows h (row := 1663) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_6
  simp only [← c101703, ← c101704, ← c101705] at e_1663_6
  have e_1663_7 := arithEq_of_rows h (row := 1663) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_7
  simp only [← c101706, ← c101707, k2, ← c101708] at e_1663_7
  have hr := e_1663_7
  simp only [e_1663_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1663 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16632)) := by
  have c101709 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101710 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101711 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101712 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101713 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101714 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1663_8 := arithEq_of_rows h (row := 1663) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_8
  simp only [← c101709, ← c101710, ← c101711] at e_1663_8
  have e_1663_9 := arithEq_of_rows h (row := 1663) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_9
  simp only [← c101712, ← c101713, k2, ← c101714] at e_1663_9
  have hr := e_1663_9
  simp only [e_1663_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1663 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16633)) := by
  have c101715 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101716 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101717 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101718 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101719 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101720 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1663_10 := arithEq_of_rows h (row := 1663) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_10
  simp only [← c101715, ← c101716, ← c101717] at e_1663_10
  have e_1663_11 := arithEq_of_rows h (row := 1663) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_11
  simp only [← c101718, ← c101719, k2, ← c101720] at e_1663_11
  have hr := e_1663_11
  simp only [e_1663_10] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
