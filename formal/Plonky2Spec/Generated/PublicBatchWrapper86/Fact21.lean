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

theorem publicBatchWrapper86_f15953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1636 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16279)) := by
  have c100071 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.1
  have c100072 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.1
  have c100073 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.1
  have c100074 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100075 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100076 := (publicBatchWrapper86_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1636_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_2
  simp only [← c100071, ← c100072, ← c100073] at e_1636_2
  have e_1636_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1636 (by decide)
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
  have e_1636_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_4
  simp only [← c100077, ← c100078, ← c100079] at e_1636_4
  have e_1636_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1636 (by decide)
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
  have e_1636_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_6
  simp only [← c100083, ← c100084, ← c100085] at e_1636_6
  have e_1636_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1636 (by decide)
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
  have e_1636_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_8
  simp only [← c100089, ← c100090, ← c100091] at e_1636_8
  have e_1636_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1636 (by decide)
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
  have e_1636_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_10
  simp only [← c100095, ← c100096, ← c100097] at e_1636_10
  have e_1636_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1636 (by decide)
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
  have e_1636_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_12
  simp only [← c100101, ← c100102, ← c100103] at e_1636_12
  have e_1636_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1636 (by decide)
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
  have e_1636_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_14
  simp only [← c100107, ← c100108, ← c100109] at e_1636_14
  have e_1636_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1636 (by decide)
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
  have e_1636_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_16
  simp only [← c100113, ← c100114, ← c100115] at e_1636_16
  have e_1636_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1636 (by decide)
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
  have e_1636_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_18
  simp only [← c100119, ← c100120, ← c100121] at e_1636_18
  have e_1636_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1636 (by decide)
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
  have e_1637_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_0
  simp only [← c100125, ← c100126, ← c100127] at e_1637_0
  have e_1637_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1637 (by decide)
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
  have e_1637_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_2
  simp only [← c100131, ← c100132, ← c100133] at e_1637_2
  have e_1637_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1637 (by decide)
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
  have e_1637_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_4
  simp only [← c100137, ← c100138, ← c100139] at e_1637_4
  have e_1637_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1637 (by decide)
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
  have e_1637_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_6
  simp only [← c100143, ← c100144, ← c100145] at e_1637_6
  have e_1637_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1637 (by decide)
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
  have e_1637_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_8
  simp only [← c100149, ← c100150, ← c100151] at e_1637_8
  have e_1637_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1637 (by decide)
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
  have e_1637_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_10
  simp only [← c100155, ← c100156, ← c100157] at e_1637_10
  have e_1637_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1637 (by decide)
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
  have e_1637_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_12
  simp only [← c100161, ← c100162, ← c100163] at e_1637_12
  have e_1637_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1637 (by decide)
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
  have e_1637_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_14
  simp only [← c100167, ← c100168, ← c100169] at e_1637_14
  have e_1637_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1637 (by decide)
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
  have e_1637_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_16
  simp only [← c100173, ← c100174, ← c100175] at e_1637_16
  have e_1637_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1637 (by decide)
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
  have e_1637_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_18
  simp only [← c100179, ← c100180, ← c100181] at e_1637_18
  have e_1637_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1637 (by decide)
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
  have e_1638_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_0
  simp only [← c100185, ← c100186, ← c100187] at e_1638_0
  have e_1638_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1638 (by decide)
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
  have e_1638_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_2
  simp only [← c100191, ← c100192, ← c100193] at e_1638_2
  have e_1638_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1638 (by decide)
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
  have e_1638_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_4
  simp only [← c100197, ← c100198, ← c100199] at e_1638_4
  have e_1638_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1638 (by decide)
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
  have e_1638_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_6
  simp only [← c100203, ← c100204, ← c100205] at e_1638_6
  have e_1638_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1638 (by decide)
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
  have e_1638_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_8
  simp only [← c100209, ← c100210, ← c100211] at e_1638_8
  have e_1638_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1638 (by decide)
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
  have e_1638_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_10
  simp only [← c100215, ← c100216, ← c100217] at e_1638_10
  have e_1638_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1638 (by decide)
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
  have e_1638_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_12
  simp only [← c100221, ← c100222, ← c100223] at e_1638_12
  have e_1638_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1638 (by decide)
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
  have e_1638_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_14
  simp only [← c100227, ← c100228, ← c100229] at e_1638_14
  have e_1638_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1638 (by decide)
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
  have e_1638_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_16
  simp only [← c100233, ← c100234, ← c100235] at e_1638_16
  have e_1638_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1638 (by decide)
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
  have e_1638_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_18
  simp only [← c100239, ← c100240, ← c100241] at e_1638_18
  have e_1638_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1638 (by decide)
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
  have e_1639_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_0
  simp only [← c100245, ← c100246, ← c100247] at e_1639_0
  have e_1639_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1639 (by decide)
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
  have e_1639_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_2
  simp only [← c100251, ← c100252, ← c100253] at e_1639_2
  have e_1639_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1639 (by decide)
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
  have e_1639_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_4
  simp only [← c100257, ← c100258, ← c100259] at e_1639_4
  have e_1639_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1639 (by decide)
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
  have e_1639_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_6
  simp only [← c100263, ← c100264, ← c100265] at e_1639_6
  have e_1639_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1639 (by decide)
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
  have e_1639_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_8
  simp only [← c100269, ← c100270, ← c100271] at e_1639_8
  have e_1639_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1639 (by decide)
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
  have e_1639_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_10
  simp only [← c100275, ← c100276, ← c100277] at e_1639_10
  have e_1639_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1639 (by decide)
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
  have e_1639_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_12
  simp only [← c100281, ← c100282, ← c100283] at e_1639_12
  have e_1639_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1639 (by decide)
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
  have e_1639_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_14
  simp only [← c100287, ← c100288, ← c100289] at e_1639_14
  have e_1639_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1639 (by decide)
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
  have e_1639_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_16
  simp only [← c100293, ← c100294, ← c100295] at e_1639_16
  have e_1639_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1639 (by decide)
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
  have e_1639_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_18
  simp only [← c100299, ← c100300, ← c100301] at e_1639_18
  have e_1639_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1639 (by decide)
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
  have e_1640_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_0
  simp only [← c100305, ← c100306, ← c100307] at e_1640_0
  have e_1640_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1640 (by decide)
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
  have e_1640_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_2
  simp only [← c100311, ← c100312, ← c100313] at e_1640_2
  have e_1640_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1640 (by decide)
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
  have e_1640_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_4
  simp only [← c100317, ← c100318, ← c100319] at e_1640_4
  have e_1640_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1640 (by decide)
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
  have e_1640_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_6
  simp only [← c100323, ← c100324, ← c100325] at e_1640_6
  have e_1640_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1640 (by decide)
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
  have e_1640_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_8
  simp only [← c100329, ← c100330, ← c100331] at e_1640_8
  have e_1640_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1640 (by decide)
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
  have e_1640_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_10
  simp only [← c100335, ← c100336, ← c100337] at e_1640_10
  have e_1640_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1640 (by decide)
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
  have e_1640_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_12
  simp only [← c100341, ← c100342, ← c100343] at e_1640_12
  have e_1640_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1640 (by decide)
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
  have e_1640_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_14
  simp only [← c100347, ← c100348, ← c100349] at e_1640_14
  have e_1640_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1640 (by decide)
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
  have e_1640_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_16
  simp only [← c100353, ← c100354, ← c100355] at e_1640_16
  have e_1640_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1640 (by decide)
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
  have e_1640_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_18
  simp only [← c100359, ← c100360, ← c100361] at e_1640_18
  have e_1640_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1640 (by decide)
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
  have e_1641_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_0
  simp only [← c100365, ← c100366, ← c100367] at e_1641_0
  have e_1641_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1641 (by decide)
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
  have e_1641_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_2
  simp only [← c100371, ← c100372, ← c100373] at e_1641_2
  have e_1641_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1641 (by decide)
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
  have e_1641_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_4
  simp only [← c100377, ← c100378, ← c100379] at e_1641_4
  have e_1641_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1641 (by decide)
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
  have e_1641_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_6
  simp only [← c100383, ← c100384, ← c100385] at e_1641_6
  have e_1641_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1641 (by decide)
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
  have e_1641_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_8
  simp only [← c100389, ← c100390, ← c100391] at e_1641_8
  have e_1641_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1641 (by decide)
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
  have e_1641_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_10
  simp only [← c100395, ← c100396, ← c100397] at e_1641_10
  have e_1641_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1641 (by decide)
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
  have e_1641_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_12
  simp only [← c100401, ← c100402, ← c100403] at e_1641_12
  have e_1641_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1641 (by decide)
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
  have e_1641_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_14
  simp only [← c100407, ← c100408, ← c100409] at e_1641_14
  have e_1641_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1641 (by decide)
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
  have e_1641_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_16
  simp only [← c100413, ← c100414, ← c100415] at e_1641_16
  have e_1641_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1641 (by decide)
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
  have e_1641_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_18
  simp only [← c100419, ← c100420, ← c100421] at e_1641_18
  have e_1641_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1641 (by decide)
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
  have e_1642_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_0
  simp only [← c100425, ← c100426, ← c100427] at e_1642_0
  have e_1642_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1642 (by decide)
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
  have e_1642_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_2
  simp only [← c100431, ← c100432, ← c100433] at e_1642_2
  have e_1642_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1642 (by decide)
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
  have e_1642_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_4
  simp only [← c100437, ← c100438, ← c100439] at e_1642_4
  have e_1642_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1642 (by decide)
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
  have e_1642_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_6
  simp only [← c100443, ← c100444, ← c100445] at e_1642_6
  have e_1642_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1642 (by decide)
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
  have e_1642_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_8
  simp only [← c100449, ← c100450, ← c100451] at e_1642_8
  have e_1642_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1642 (by decide)
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
  have e_1642_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_10
  simp only [← c100455, ← c100456, ← c100457] at e_1642_10
  have e_1642_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1642 (by decide)
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
  have e_1642_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_12
  simp only [← c100461, ← c100462, ← c100463] at e_1642_12
  have e_1642_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1642 (by decide)
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
  have e_1642_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_14
  simp only [← c100467, ← c100468, ← c100469] at e_1642_14
  have e_1642_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1642 (by decide)
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
  have e_1642_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_16
  simp only [← c100473, ← c100474, ← c100475] at e_1642_16
  have e_1642_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1642 (by decide)
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
  have e_1642_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_18
  simp only [← c100479, ← c100480, ← c100481] at e_1642_18
  have e_1642_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1642 (by decide)
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
  have e_1643_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_0
  simp only [← c100485, ← c100486, ← c100487] at e_1643_0
  have e_1643_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1643 (by decide)
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
  have e_1643_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_2
  simp only [← c100491, ← c100492, ← c100493] at e_1643_2
  have e_1643_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1643 (by decide)
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
  have e_1643_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_4
  simp only [← c100497, ← c100498, ← c100499] at e_1643_4
  have e_1643_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1643 (by decide)
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
  have e_1643_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_6
  simp only [← c100503, ← c100504, ← c100505] at e_1643_6
  have e_1643_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1643 (by decide)
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
  have e_1643_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_8
  simp only [← c100509, ← c100510, ← c100511] at e_1643_8
  have e_1643_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1643 (by decide)
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
  have e_1643_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_10
  simp only [← c100515, ← c100516, ← c100517] at e_1643_10
  have e_1643_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1643 (by decide)
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
  have e_1643_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_12
  simp only [← c100521, ← c100522, ← c100523] at e_1643_12
  have e_1643_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1643 (by decide)
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
  have e_1643_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_14
  simp only [← c100527, ← c100528, ← c100529] at e_1643_14
  have e_1643_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1643 (by decide)
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
  have e_1643_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_16
  simp only [← c100533, ← c100534, ← c100535] at e_1643_16
  have e_1643_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1643 (by decide)
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
  have e_1643_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_18
  simp only [← c100539, ← c100540, ← c100541] at e_1643_18
  have e_1643_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1643 (by decide)
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
  have e_1644_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_0
  simp only [← c100545, ← c100546, ← c100547] at e_1644_0
  have e_1644_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1644 (by decide)
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
  have e_1644_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_2
  simp only [← c100551, ← c100552, ← c100553] at e_1644_2
  have e_1644_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1644 (by decide)
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
  have e_1644_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_4
  simp only [← c100557, ← c100558, ← c100559] at e_1644_4
  have e_1644_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1644 (by decide)
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
  have e_1644_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_6
  simp only [← c100563, ← c100564, ← c100565] at e_1644_6
  have e_1644_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1644 (by decide)
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
  have e_1644_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_8
  simp only [← c100569, ← c100570, ← c100571] at e_1644_8
  have e_1644_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1644 (by decide)
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
  have e_1644_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_10
  simp only [← c100575, ← c100576, ← c100577] at e_1644_10
  have e_1644_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1644 (by decide)
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
  have e_1644_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_12
  simp only [← c100581, ← c100582, ← c100583] at e_1644_12
  have e_1644_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1644 (by decide)
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
  have e_1644_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_14
  simp only [← c100587, ← c100588, ← c100589] at e_1644_14
  have e_1644_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1644 (by decide)
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
  have e_1644_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_16
  simp only [← c100593, ← c100594, ← c100595] at e_1644_16
  have e_1644_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1644 (by decide)
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
  have e_1644_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_18
  simp only [← c100599, ← c100600, ← c100601] at e_1644_18
  have e_1644_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1644 (by decide)
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
  have e_1645_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_0
  simp only [← c100605, ← c100606, ← c100607] at e_1645_0
  have e_1645_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1645 (by decide)
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
  have e_1645_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_2
  simp only [← c100611, ← c100612, ← c100613] at e_1645_2
  have e_1645_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1645 (by decide)
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
  have e_1645_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_4
  simp only [← c100617, ← c100618, ← c100619] at e_1645_4
  have e_1645_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1645 (by decide)
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
  have e_1645_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_6
  simp only [← c100623, ← c100624, ← c100625] at e_1645_6
  have e_1645_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1645 (by decide)
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
  have e_1645_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_8
  simp only [← c100629, ← c100630, ← c100631] at e_1645_8
  have e_1645_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1645 (by decide)
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
  have e_1645_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_10
  simp only [← c100635, ← c100636, ← c100637] at e_1645_10
  have e_1645_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1645 (by decide)
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
  have e_1645_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_12
  simp only [← c100641, ← c100642, ← c100643] at e_1645_12
  have e_1645_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1645 (by decide)
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
  have e_1645_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_14
  simp only [← c100647, ← c100648, ← c100649] at e_1645_14
  have e_1645_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1645 (by decide)
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
  have e_1645_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_16
  simp only [← c100653, ← c100654, ← c100655] at e_1645_16
  have e_1645_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1645 (by decide)
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
  have e_1645_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_18
  simp only [← c100659, ← c100660, ← c100661] at e_1645_18
  have e_1645_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1645 (by decide)
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
  have e_1646_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_0
  simp only [← c100665, ← c100666, ← c100667] at e_1646_0
  have e_1646_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1646 (by decide)
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
  have e_1646_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_2
  simp only [← c100671, ← c100672, ← c100673] at e_1646_2
  have e_1646_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1646 (by decide)
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
  have e_1646_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_4
  simp only [← c100677, ← c100678, ← c100679] at e_1646_4
  have e_1646_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1646 (by decide)
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
  have e_1646_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_6
  simp only [← c100683, ← c100684, ← c100685] at e_1646_6
  have e_1646_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1646 (by decide)
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
  have e_1646_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_8
  simp only [← c100689, ← c100690, ← c100691] at e_1646_8
  have e_1646_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1646 (by decide)
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
  have e_1646_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_10
  simp only [← c100695, ← c100696, ← c100697] at e_1646_10
  have e_1646_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1646 (by decide)
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
  have e_1646_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_12
  simp only [← c100701, ← c100702, ← c100703] at e_1646_12
  have e_1646_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1646 (by decide)
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
  have e_1646_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_14
  simp only [← c100707, ← c100708, ← c100709] at e_1646_14
  have e_1646_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1646 (by decide)
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
  have e_1646_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_16
  simp only [← c100713, ← c100714, ← c100715] at e_1646_16
  have e_1646_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1646 (by decide)
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
  have e_1646_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_18
  simp only [← c100719, ← c100720, ← c100721] at e_1646_18
  have e_1646_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1646 (by decide)
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
  have e_1647_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_0
  simp only [← c100725, ← c100726, ← c100727] at e_1647_0
  have e_1647_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1647 (by decide)
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
  have e_1647_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_2
  simp only [← c100731, ← c100732, ← c100733] at e_1647_2
  have e_1647_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1647 (by decide)
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
  have e_1647_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_4
  simp only [← c100737, ← c100738, ← c100739] at e_1647_4
  have e_1647_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1647 (by decide)
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
  have e_1647_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_6
  simp only [← c100743, ← c100744, ← c100745] at e_1647_6
  have e_1647_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1647 (by decide)
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
  have e_1647_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_8
  simp only [← c100749, ← c100750, ← c100751] at e_1647_8
  have e_1647_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1647 (by decide)
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
  have e_1647_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_10
  simp only [← c100755, ← c100756, ← c100757] at e_1647_10
  have e_1647_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1647 (by decide)
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
  have e_1647_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_12
  simp only [← c100761, ← c100762, ← c100763] at e_1647_12
  have e_1647_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1647 (by decide)
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
  have e_1647_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_14
  simp only [← c100767, ← c100768, ← c100769] at e_1647_14
  have e_1647_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1647 (by decide)
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
  have e_1647_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_16
  simp only [← c100773, ← c100774, ← c100775] at e_1647_16
  have e_1647_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1647 (by decide)
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
  have e_1647_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_18
  simp only [← c100779, ← c100780, ← c100781] at e_1647_18
  have e_1647_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1647 (by decide)
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
  have e_1648_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_0
  simp only [← c100785, ← c100786, ← c100787] at e_1648_0
  have e_1648_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1648 (by decide)
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
  have e_1648_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_2
  simp only [← c100791, ← c100792, ← c100793] at e_1648_2
  have e_1648_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1648 (by decide)
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
  have e_1648_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_4
  simp only [← c100797, ← c100798, ← c100799] at e_1648_4
  have e_1648_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1648 (by decide)
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
  have e_1648_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_6
  simp only [← c100803, ← c100804, ← c100805] at e_1648_6
  have e_1648_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1648 (by decide)
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
  have e_1648_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_8
  simp only [← c100809, ← c100810, ← c100811] at e_1648_8
  have e_1648_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1648 (by decide)
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
  have e_1648_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_10
  simp only [← c100815, ← c100816, ← c100817] at e_1648_10
  have e_1648_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1648 (by decide)
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
  have e_1648_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_12
  simp only [← c100821, ← c100822, ← c100823] at e_1648_12
  have e_1648_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1648 (by decide)
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
  have e_1648_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_14
  simp only [← c100827, ← c100828, ← c100829] at e_1648_14
  have e_1648_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1648 (by decide)
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
  have e_1648_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_16
  simp only [← c100833, ← c100834, ← c100835] at e_1648_16
  have e_1648_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1648 (by decide)
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
  have e_1648_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_18
  simp only [← c100839, ← c100840, ← c100841] at e_1648_18
  have e_1648_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1648 (by decide)
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
  have e_1649_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_0
  simp only [← c100845, ← c100846, ← c100847] at e_1649_0
  have e_1649_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1649 (by decide)
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
  have e_1649_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_2
  simp only [← c100851, ← c100852, ← c100853] at e_1649_2
  have e_1649_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1649 (by decide)
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
  have e_1649_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_4
  simp only [← c100857, ← c100858, ← c100859] at e_1649_4
  have e_1649_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1649 (by decide)
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
  have e_1649_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_6
  simp only [← c100863, ← c100864, ← c100865] at e_1649_6
  have e_1649_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1649 (by decide)
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
  have e_1649_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_8
  simp only [← c100869, ← c100870, ← c100871] at e_1649_8
  have e_1649_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1649 (by decide)
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
  have e_1649_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_10
  simp only [← c100875, ← c100876, ← c100877] at e_1649_10
  have e_1649_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1649 (by decide)
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
  have e_1649_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_12
  simp only [← c100881, ← c100882, ← c100883] at e_1649_12
  have e_1649_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1649 (by decide)
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
  have e_1649_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_14
  simp only [← c100887, ← c100888, ← c100889] at e_1649_14
  have e_1649_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1649 (by decide)
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
  have e_1649_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_16
  simp only [← c100893, ← c100894, ← c100895] at e_1649_16
  have e_1649_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1649 (by decide)
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
  have e_1649_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_18
  simp only [← c100899, ← c100900, ← c100901] at e_1649_18
  have e_1649_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1649 (by decide)
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
  have e_1650_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_0
  simp only [← c100905, ← c100906, ← c100907] at e_1650_0
  have e_1650_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1650 (by decide)
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
  have e_1650_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_2
  simp only [← c100911, ← c100912, ← c100913] at e_1650_2
  have e_1650_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1650 (by decide)
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
  have e_1650_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_4
  simp only [← c100917, ← c100918, ← c100919] at e_1650_4
  have e_1650_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1650 (by decide)
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
  have e_1650_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_6
  simp only [← c100923, ← c100924, ← c100925] at e_1650_6
  have e_1650_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1650 (by decide)
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
  have e_1650_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_8
  simp only [← c100929, ← c100930, ← c100931] at e_1650_8
  have e_1650_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1650 (by decide)
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
  have e_1650_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_10
  simp only [← c100935, ← c100936, ← c100937] at e_1650_10
  have e_1650_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1650 (by decide)
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
  have e_1650_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_12
  simp only [← c100941, ← c100942, ← c100943] at e_1650_12
  have e_1650_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1650 (by decide)
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
  have e_1650_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_14
  simp only [← c100947, ← c100948, ← c100949] at e_1650_14
  have e_1650_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1650 (by decide)
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
  have e_1650_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_16
  simp only [← c100953, ← c100954, ← c100955] at e_1650_16
  have e_1650_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1650 (by decide)
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
  have e_1650_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_18
  simp only [← c100959, ← c100960, ← c100961] at e_1650_18
  have e_1650_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1650 (by decide)
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
  have e_1651_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_0
  simp only [← c100965, ← c100966, ← c100967] at e_1651_0
  have e_1651_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1651 (by decide)
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
  have e_1651_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_2
  simp only [← c100971, ← c100972, ← c100973] at e_1651_2
  have e_1651_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1651 (by decide)
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
  have e_1651_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_4
  simp only [← c100977, ← c100978, ← c100979] at e_1651_4
  have e_1651_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1651 (by decide)
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
  have e_1651_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_6
  simp only [← c100983, ← c100984, ← c100985] at e_1651_6
  have e_1651_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1651 (by decide)
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
  have e_1651_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_8
  simp only [← c100989, ← c100990, ← c100991] at e_1651_8
  have e_1651_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1651 (by decide)
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
  have e_1651_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_10
  simp only [← c100995, ← c100996, ← c100997] at e_1651_10
  have e_1651_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1651 (by decide)
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
  have e_1651_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_12
  simp only [← c101001, ← c101002, ← c101003] at e_1651_12
  have e_1651_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1651 (by decide)
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
  have e_1651_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_14
  simp only [← c101007, ← c101008, ← c101009] at e_1651_14
  have e_1651_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1651 (by decide)
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
  have e_1651_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_16
  simp only [← c101013, ← c101014, ← c101015] at e_1651_16
  have e_1651_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1651 (by decide)
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
  have e_1651_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_18
  simp only [← c101019, ← c101020, ← c101021] at e_1651_18
  have e_1651_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1651 (by decide)
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
  have e_1652_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_0
  simp only [← c101025, ← c101026, ← c101027] at e_1652_0
  have e_1652_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1652 (by decide)
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
  have e_1652_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_2
  simp only [← c101031, ← c101032, ← c101033] at e_1652_2
  have e_1652_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1652 (by decide)
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
  have e_1652_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_4
  simp only [← c101037, ← c101038, ← c101039] at e_1652_4
  have e_1652_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1652 (by decide)
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
  have e_1652_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_6
  simp only [← c101043, ← c101044, ← c101045] at e_1652_6
  have e_1652_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1652 (by decide)
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
  have e_1652_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_8
  simp only [← c101049, ← c101050, ← c101051] at e_1652_8
  have e_1652_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1652 (by decide)
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
  have e_1652_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_10
  simp only [← c101055, ← c101056, ← c101057] at e_1652_10
  have e_1652_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1652 (by decide)
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
  have e_1652_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_12
  simp only [← c101061, ← c101062, ← c101063] at e_1652_12
  have e_1652_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1652 (by decide)
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
  have e_1652_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_14
  simp only [← c101067, ← c101068, ← c101069] at e_1652_14
  have e_1652_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1652 (by decide)
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
  have e_1652_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_16
  simp only [← c101073, ← c101074, ← c101075] at e_1652_16
  have e_1652_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1652 (by decide)
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
  have e_1652_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_18
  simp only [← c101079, ← c101080, ← c101081] at e_1652_18
  have e_1652_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1652 (by decide)
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
  have e_1653_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_0
  simp only [← c101085, ← c101086, ← c101087] at e_1653_0
  have e_1653_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1653 (by decide)
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
  have e_1653_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_2
  simp only [← c101091, ← c101092, ← c101093] at e_1653_2
  have e_1653_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1653 (by decide)
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
  have e_1653_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_4
  simp only [← c101097, ← c101098, ← c101099] at e_1653_4
  have e_1653_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1653 (by decide)
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
  have e_1653_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_6
  simp only [← c101103, ← c101104, ← c101105] at e_1653_6
  have e_1653_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1653 (by decide)
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
  have e_1653_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_8
  simp only [← c101109, ← c101110, ← c101111] at e_1653_8
  have e_1653_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1653 (by decide)
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
  have e_1653_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_10
  simp only [← c101115, ← c101116, ← c101117] at e_1653_10
  have e_1653_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1653 (by decide)
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
  have e_1653_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_12
  simp only [← c101121, ← c101122, ← c101123] at e_1653_12
  have e_1653_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1653 (by decide)
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
  have e_1653_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_14
  simp only [← c101127, ← c101128, ← c101129] at e_1653_14
  have e_1653_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1653 (by decide)
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
  have e_1653_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_16
  simp only [← c101133, ← c101134, ← c101135] at e_1653_16
  have e_1653_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1653 (by decide)
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
  have e_1653_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_18
  simp only [← c101139, ← c101140, ← c101141] at e_1653_18
  have e_1653_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1653 (by decide)
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
  have e_1654_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_0
  simp only [← c101145, ← c101146, ← c101147] at e_1654_0
  have e_1654_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1654 (by decide)
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
  have e_1654_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_2
  simp only [← c101151, ← c101152, ← c101153] at e_1654_2
  have e_1654_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1654 (by decide)
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
  have e_1654_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_4
  simp only [← c101157, ← c101158, ← c101159] at e_1654_4
  have e_1654_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1654 (by decide)
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
  have e_1654_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_6
  simp only [← c101163, ← c101164, ← c101165] at e_1654_6
  have e_1654_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1654 (by decide)
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
  have e_1654_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_8
  simp only [← c101169, ← c101170, ← c101171] at e_1654_8
  have e_1654_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1654 (by decide)
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
  have e_1654_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_10
  simp only [← c101175, ← c101176, ← c101177] at e_1654_10
  have e_1654_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1654 (by decide)
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
  have e_1654_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_12
  simp only [← c101181, ← c101182, ← c101183] at e_1654_12
  have e_1654_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1654 (by decide)
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
  have e_1654_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_14
  simp only [← c101187, ← c101188, ← c101189] at e_1654_14
  have e_1654_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1654 (by decide)
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
  have e_1654_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_16
  simp only [← c101193, ← c101194, ← c101195] at e_1654_16
  have e_1654_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1654 (by decide)
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
  have e_1654_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_18
  simp only [← c101199, ← c101200, ← c101201] at e_1654_18
  have e_1654_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1654 (by decide)
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
  have e_1655_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_0
  simp only [← c101205, ← c101206, ← c101207] at e_1655_0
  have e_1655_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1655 (by decide)
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
  have e_1655_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_2
  simp only [← c101211, ← c101212, ← c101213] at e_1655_2
  have e_1655_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1655 (by decide)
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
  have e_1655_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_4
  simp only [← c101217, ← c101218, ← c101219] at e_1655_4
  have e_1655_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1655 (by decide)
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
  have e_1655_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_6
  simp only [← c101223, ← c101224, ← c101225] at e_1655_6
  have e_1655_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1655 (by decide)
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
  have e_1655_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_8
  simp only [← c101229, ← c101230, ← c101231] at e_1655_8
  have e_1655_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1655 (by decide)
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
  have e_1655_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_10
  simp only [← c101235, ← c101236, ← c101237] at e_1655_10
  have e_1655_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1655 (by decide)
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
  have e_1655_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_12
  simp only [← c101241, ← c101242, ← c101243] at e_1655_12
  have e_1655_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1655 (by decide)
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
  have e_1655_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_14
  simp only [← c101247, ← c101248, ← c101249] at e_1655_14
  have e_1655_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1655 (by decide)
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
  have e_1655_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_16
  simp only [← c101253, ← c101254, ← c101255] at e_1655_16
  have e_1655_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1655 (by decide)
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
  have e_1655_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_18
  simp only [← c101259, ← c101260, ← c101261] at e_1655_18
  have e_1655_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1655 (by decide)
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
  have e_1656_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_0
  simp only [← c101265, ← c101266, ← c101267] at e_1656_0
  have e_1656_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1656 (by decide)
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
  have e_1656_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_2
  simp only [← c101271, ← c101272, ← c101273] at e_1656_2
  have e_1656_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1656 (by decide)
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
  have e_1656_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_4
  simp only [← c101277, ← c101278, ← c101279] at e_1656_4
  have e_1656_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1656 (by decide)
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
  have e_1656_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_6
  simp only [← c101283, ← c101284, ← c101285] at e_1656_6
  have e_1656_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1656 (by decide)
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
  have e_1656_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_8
  simp only [← c101289, ← c101290, ← c101291] at e_1656_8
  have e_1656_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1656 (by decide)
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
  have e_1656_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_10
  simp only [← c101295, ← c101296, ← c101297] at e_1656_10
  have e_1656_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1656 (by decide)
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
  have e_1656_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_12
  simp only [← c101301, ← c101302, ← c101303] at e_1656_12
  have e_1656_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1656 (by decide)
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
  have e_1656_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_14
  simp only [← c101307, ← c101308, ← c101309] at e_1656_14
  have e_1656_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1656 (by decide)
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
  have e_1656_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_16
  simp only [← c101313, ← c101314, ← c101315] at e_1656_16
  have e_1656_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1656 (by decide)
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
  have e_1656_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_18
  simp only [← c101319, ← c101320, ← c101321] at e_1656_18
  have e_1656_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1656 (by decide)
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
  have e_1657_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_0
  simp only [← c101325, ← c101326, ← c101327] at e_1657_0
  have e_1657_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1657 (by decide)
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
  have e_1657_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_2
  simp only [← c101331, ← c101332, ← c101333] at e_1657_2
  have e_1657_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1657 (by decide)
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
  have e_1657_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_4
  simp only [← c101337, ← c101338, ← c101339] at e_1657_4
  have e_1657_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1657 (by decide)
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
  have e_1657_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_6
  simp only [← c101343, ← c101344, ← c101345] at e_1657_6
  have e_1657_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1657 (by decide)
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
  have e_1657_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_8
  simp only [← c101349, ← c101350, ← c101351] at e_1657_8
  have e_1657_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1657 (by decide)
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
  have e_1657_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_10
  simp only [← c101355, ← c101356, ← c101357] at e_1657_10
  have e_1657_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1657 (by decide)
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
  have e_1657_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_12
  simp only [← c101361, ← c101362, ← c101363] at e_1657_12
  have e_1657_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1657 (by decide)
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
  have e_1657_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_14
  simp only [← c101367, ← c101368, ← c101369] at e_1657_14
  have e_1657_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1657 (by decide)
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
  have e_1657_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_16
  simp only [← c101373, ← c101374, ← c101375] at e_1657_16
  have e_1657_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1657 (by decide)
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
  have e_1657_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_18
  simp only [← c101379, ← c101380, ← c101381] at e_1657_18
  have e_1657_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1657 (by decide)
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
  have e_1658_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_0
  simp only [← c101385, ← c101386, ← c101387] at e_1658_0
  have e_1658_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1658 (by decide)
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
  have e_1658_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_2
  simp only [← c101391, ← c101392, ← c101393] at e_1658_2
  have e_1658_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1658 (by decide)
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
  have e_1658_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_4
  simp only [← c101397, ← c101398, ← c101399] at e_1658_4
  have e_1658_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1658 (by decide)
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
  have e_1658_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_6
  simp only [← c101403, ← c101404, ← c101405] at e_1658_6
  have e_1658_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1658 (by decide)
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
  have e_1658_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_8
  simp only [← c101409, ← c101410, ← c101411] at e_1658_8
  have e_1658_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1658 (by decide)
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
  have e_1658_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_10
  simp only [← c101415, ← c101416, ← c101417] at e_1658_10
  have e_1658_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1658 (by decide)
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
  have e_1658_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_12
  simp only [← c101421, ← c101422, ← c101423] at e_1658_12
  have e_1658_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1658 (by decide)
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
  have e_1658_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_14
  simp only [← c101427, ← c101428, ← c101429] at e_1658_14
  have e_1658_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1658 (by decide)
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
  have e_1658_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_16
  simp only [← c101433, ← c101434, ← c101435] at e_1658_16
  have e_1658_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1658 (by decide)
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
  have e_1658_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_18
  simp only [← c101439, ← c101440, ← c101441] at e_1658_18
  have e_1658_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1658 (by decide)
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
  have e_1659_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_0
  simp only [← c101445, ← c101446, ← c101447] at e_1659_0
  have e_1659_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1659 (by decide)
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
  have e_1659_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_2
  simp only [← c101451, ← c101452, ← c101453] at e_1659_2
  have e_1659_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1659 (by decide)
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
  have e_1659_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_4
  simp only [← c101457, ← c101458, ← c101459] at e_1659_4
  have e_1659_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1659 (by decide)
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
  have e_1659_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_6
  simp only [← c101463, ← c101464, ← c101465] at e_1659_6
  have e_1659_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1659 (by decide)
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
  have e_1659_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_8
  simp only [← c101469, ← c101470, ← c101471] at e_1659_8
  have e_1659_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1659 (by decide)
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
  have e_1659_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_10
  simp only [← c101475, ← c101476, ← c101477] at e_1659_10
  have e_1659_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1659 (by decide)
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
  have e_1659_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_12
  simp only [← c101481, ← c101482, ← c101483] at e_1659_12
  have e_1659_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1659 (by decide)
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
  have e_1659_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_14
  simp only [← c101487, ← c101488, ← c101489] at e_1659_14
  have e_1659_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1659 (by decide)
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
  have e_1659_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_16
  simp only [← c101493, ← c101494, ← c101495] at e_1659_16
  have e_1659_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1659 (by decide)
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
  have e_1659_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_18
  simp only [← c101499, ← c101500, ← c101501] at e_1659_18
  have e_1659_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1659 (by decide)
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
  have e_1660_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_0
  simp only [← c101505, ← c101506, ← c101507] at e_1660_0
  have e_1660_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1660 (by decide)
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
  have e_1660_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_2
  simp only [← c101511, ← c101512, ← c101513] at e_1660_2
  have e_1660_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1660 (by decide)
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
  have e_1660_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_4
  simp only [← c101517, ← c101518, ← c101519] at e_1660_4
  have e_1660_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1660 (by decide)
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
  have e_1660_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_6
  simp only [← c101523, ← c101524, ← c101525] at e_1660_6
  have e_1660_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1660 (by decide)
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
  have e_1660_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_8
  simp only [← c101529, ← c101530, ← c101531] at e_1660_8
  have e_1660_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1660 (by decide)
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
  have e_1660_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_10
  simp only [← c101535, ← c101536, ← c101537] at e_1660_10
  have e_1660_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1660 (by decide)
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
  have e_1660_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_12
  simp only [← c101541, ← c101542, ← c101543] at e_1660_12
  have e_1660_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1660 (by decide)
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
  have e_1660_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_14
  simp only [← c101547, ← c101548, ← c101549] at e_1660_14
  have e_1660_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1660 (by decide)
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
  have e_1660_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_16
  simp only [← c101553, ← c101554, ← c101555] at e_1660_16
  have e_1660_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1660 (by decide)
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
  have e_1660_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_18
  simp only [← c101559, ← c101560, ← c101561] at e_1660_18
  have e_1660_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1660 (by decide)
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
  have e_1661_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_0
  simp only [← c101565, ← c101566, ← c101567] at e_1661_0
  have e_1661_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1661 (by decide)
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
  have e_1661_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_2
  simp only [← c101571, ← c101572, ← c101573] at e_1661_2
  have e_1661_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1661 (by decide)
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
  have e_1661_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_4
  simp only [← c101577, ← c101578, ← c101579] at e_1661_4
  have e_1661_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1661 (by decide)
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
  have e_1661_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_6
  simp only [← c101583, ← c101584, ← c101585] at e_1661_6
  have e_1661_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1661 (by decide)
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
  have e_1661_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_8
  simp only [← c101589, ← c101590, ← c101591] at e_1661_8
  have e_1661_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1661 (by decide)
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
  have e_1661_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_10
  simp only [← c101595, ← c101596, ← c101597] at e_1661_10
  have e_1661_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1661 (by decide)
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
  have e_1661_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_12
  simp only [← c101601, ← c101602, ← c101603] at e_1661_12
  have e_1661_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1661 (by decide)
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
  have e_1661_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_14
  simp only [← c101607, ← c101608, ← c101609] at e_1661_14
  have e_1661_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1661 (by decide)
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
  have e_1661_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_16
  simp only [← c101613, ← c101614, ← c101615] at e_1661_16
  have e_1661_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1661 (by decide)
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
  have e_1661_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_18
  simp only [← c101619, ← c101620, ← c101621] at e_1661_18
  have e_1661_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1661 (by decide)
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
  have e_1662_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_0
  simp only [← c101625, ← c101626, ← c101627] at e_1662_0
  have e_1662_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1662 (by decide)
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
  have e_1662_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_2
  simp only [← c101631, ← c101632, ← c101633] at e_1662_2
  have e_1662_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1662 (by decide)
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
  have e_1662_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_4
  simp only [← c101637, ← c101638, ← c101639] at e_1662_4
  have e_1662_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1662 (by decide)
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
  have e_1662_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_6
  simp only [← c101643, ← c101644, ← c101645] at e_1662_6
  have e_1662_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1662 (by decide)
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
  have e_1662_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_8
  simp only [← c101649, ← c101650, ← c101651] at e_1662_8
  have e_1662_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1662 (by decide)
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
  have e_1662_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_10
  simp only [← c101655, ← c101656, ← c101657] at e_1662_10
  have e_1662_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1662 (by decide)
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
  have e_1662_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_12
  simp only [← c101661, ← c101662, ← c101663] at e_1662_12
  have e_1662_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1662 (by decide)
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
  have e_1662_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_14
  simp only [← c101667, ← c101668, ← c101669] at e_1662_14
  have e_1662_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1662 (by decide)
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
  have e_1662_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_16
  simp only [← c101673, ← c101674, ← c101675] at e_1662_16
  have e_1662_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1662 (by decide)
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
  have e_1662_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_18
  simp only [← c101679, ← c101680, ← c101681] at e_1662_18
  have e_1662_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1662 (by decide)
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
  have e_1663_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_0
  simp only [← c101685, ← c101686, ← c101687] at e_1663_0
  have e_1663_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1663 (by decide)
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
  have e_1663_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_2
  simp only [← c101691, ← c101692, ← c101693] at e_1663_2
  have e_1663_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1663 (by decide)
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
  have e_1663_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_4
  simp only [← c101697, ← c101698, ← c101699] at e_1663_4
  have e_1663_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1663 (by decide)
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
  have e_1663_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_6
  simp only [← c101703, ← c101704, ← c101705] at e_1663_6
  have e_1663_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1663 (by decide)
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
  have e_1663_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_8
  simp only [← c101709, ← c101710, ← c101711] at e_1663_8
  have e_1663_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1663 (by decide)
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
  have e_1663_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_10
  simp only [← c101715, ← c101716, ← c101717] at e_1663_10
  have e_1663_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_11
  simp only [← c101718, ← c101719, k2, ← c101720] at e_1663_11
  have hr := e_1663_11
  simp only [e_1663_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1663 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16634)) := by
  have c101721 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101722 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101723 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101724 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101725 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101726 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1663_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_12
  simp only [← c101721, ← c101722, ← c101723] at e_1663_12
  have e_1663_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_13
  simp only [← c101724, ← c101725, k2, ← c101726] at e_1663_13
  have hr := e_1663_13
  simp only [e_1663_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1663 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16635)) := by
  have c101727 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101728 := (publicBatchWrapper86_copies3179 a h).1
  have c101729 := (publicBatchWrapper86_copies3179 a h).2.1
  have c101730 := (publicBatchWrapper86_copies3179 a h).2.2.1
  have c101731 := (publicBatchWrapper86_copies3179 a h).2.2.2.1
  have c101732 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1663_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_14
  simp only [← c101727, ← c101728, ← c101729] at e_1663_14
  have e_1663_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_15
  simp only [← c101730, ← c101731, k2, ← c101732] at e_1663_15
  have hr := e_1663_15
  simp only [e_1663_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1663 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16636)) := by
  have c101733 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.1
  have c101734 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.1
  have c101735 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.1
  have c101736 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.1
  have c101737 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.1
  have c101738 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1663_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_16
  simp only [← c101733, ← c101734, ← c101735] at e_1663_16
  have e_1663_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_17
  simp only [← c101736, ← c101737, k2, ← c101738] at e_1663_17
  have hr := e_1663_17
  simp only [e_1663_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1663 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16637)) := by
  have c101739 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101740 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101741 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101742 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101743 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101744 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1663_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_18
  simp only [← c101739, ← c101740, ← c101741] at e_1663_18
  have e_1663_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_19
  simp only [← c101742, ← c101743, k2, ← c101744] at e_1663_19
  have hr := e_1663_19
  simp only [e_1663_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16638)) := by
  have c101745 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101746 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101747 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101748 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101749 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101750 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_0
  simp only [← c101745, ← c101746, ← c101747] at e_1664_0
  have e_1664_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_1
  simp only [← c101748, ← c101749, k2, ← c101750] at e_1664_1
  have hr := e_1664_1
  simp only [e_1664_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16639)) := by
  have c101751 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101752 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101753 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101754 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101755 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101756 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_2
  simp only [← c101751, ← c101752, ← c101753] at e_1664_2
  have e_1664_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_3
  simp only [← c101754, ← c101755, k2, ← c101756] at e_1664_3
  have hr := e_1664_3
  simp only [e_1664_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16640)) := by
  have c101757 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101758 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101759 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101760 := (publicBatchWrapper86_copies3180 a h).1
  have c101761 := (publicBatchWrapper86_copies3180 a h).2.1
  have c101762 := (publicBatchWrapper86_copies3180 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_4
  simp only [← c101757, ← c101758, ← c101759] at e_1664_4
  have e_1664_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_5
  simp only [← c101760, ← c101761, k2, ← c101762] at e_1664_5
  have hr := e_1664_5
  simp only [e_1664_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16641)) := by
  have c101763 := (publicBatchWrapper86_copies3180 a h).2.2.2.1
  have c101764 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.1
  have c101765 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.1
  have c101766 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.1
  have c101767 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.1
  have c101768 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_6
  simp only [← c101763, ← c101764, ← c101765] at e_1664_6
  have e_1664_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_7
  simp only [← c101766, ← c101767, k2, ← c101768] at e_1664_7
  have hr := e_1664_7
  simp only [e_1664_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16642)) := by
  have c101769 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.1
  have c101770 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101771 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101772 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101773 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101774 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_8
  simp only [← c101769, ← c101770, ← c101771] at e_1664_8
  have e_1664_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_9
  simp only [← c101772, ← c101773, k2, ← c101774] at e_1664_9
  have hr := e_1664_9
  simp only [e_1664_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16643)) := by
  have c101775 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101776 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101777 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101778 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101779 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101780 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_10
  simp only [← c101775, ← c101776, ← c101777] at e_1664_10
  have e_1664_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_11
  simp only [← c101778, ← c101779, k2, ← c101780] at e_1664_11
  have hr := e_1664_11
  simp only [e_1664_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16644)) := by
  have c101781 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101782 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101783 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101784 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101785 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101786 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_12
  simp only [← c101781, ← c101782, ← c101783] at e_1664_12
  have e_1664_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_13
  simp only [← c101784, ← c101785, k2, ← c101786] at e_1664_13
  have hr := e_1664_13
  simp only [e_1664_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16645)) := by
  have c101787 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101788 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101789 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101790 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101791 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101792 := (publicBatchWrapper86_copies3181 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_14
  simp only [← c101787, ← c101788, ← c101789] at e_1664_14
  have e_1664_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_15
  simp only [← c101790, ← c101791, k2, ← c101792] at e_1664_15
  have hr := e_1664_15
  simp only [e_1664_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16646)) := by
  have c101793 := (publicBatchWrapper86_copies3181 a h).2.1
  have c101794 := (publicBatchWrapper86_copies3181 a h).2.2.1
  have c101795 := (publicBatchWrapper86_copies3181 a h).2.2.2.1
  have c101796 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.1
  have c101797 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.1
  have c101798 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_16
  simp only [← c101793, ← c101794, ← c101795] at e_1664_16
  have e_1664_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_17
  simp only [← c101796, ← c101797, k2, ← c101798] at e_1664_17
  have hr := e_1664_17
  simp only [e_1664_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16647)) := by
  have c101799 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.1
  have c101800 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.1
  have c101801 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.1
  have c101802 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101803 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101804 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_18
  simp only [← c101799, ← c101800, ← c101801] at e_1664_18
  have e_1664_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_19
  simp only [← c101802, ← c101803, k2, ← c101804] at e_1664_19
  have hr := e_1664_19
  simp only [e_1664_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16648)) := by
  have c101805 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101806 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101807 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101808 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101809 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101810 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_0
  simp only [← c101805, ← c101806, ← c101807] at e_1665_0
  have e_1665_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_1
  simp only [← c101808, ← c101809, k2, ← c101810] at e_1665_1
  have hr := e_1665_1
  simp only [e_1665_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16649)) := by
  have c101811 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101812 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101813 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101814 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101815 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101816 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_2
  simp only [← c101811, ← c101812, ← c101813] at e_1665_2
  have e_1665_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_3
  simp only [← c101814, ← c101815, k2, ← c101816] at e_1665_3
  have hr := e_1665_3
  simp only [e_1665_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16650)) := by
  have c101817 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101818 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101819 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101820 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101821 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101822 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_4
  simp only [← c101817, ← c101818, ← c101819] at e_1665_4
  have e_1665_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_5
  simp only [← c101820, ← c101821, k2, ← c101822] at e_1665_5
  have hr := e_1665_5
  simp only [e_1665_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16651)) := by
  have c101823 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101824 := (publicBatchWrapper86_copies3182 a h).1
  have c101825 := (publicBatchWrapper86_copies3182 a h).2.1
  have c101826 := (publicBatchWrapper86_copies3182 a h).2.2.1
  have c101827 := (publicBatchWrapper86_copies3182 a h).2.2.2.1
  have c101828 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_6
  simp only [← c101823, ← c101824, ← c101825] at e_1665_6
  have e_1665_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_7
  simp only [← c101826, ← c101827, k2, ← c101828] at e_1665_7
  have hr := e_1665_7
  simp only [e_1665_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16652)) := by
  have c101829 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.1
  have c101830 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.1
  have c101831 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.1
  have c101832 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.1
  have c101833 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.1
  have c101834 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_8
  simp only [← c101829, ← c101830, ← c101831] at e_1665_8
  have e_1665_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_9
  simp only [← c101832, ← c101833, k2, ← c101834] at e_1665_9
  have hr := e_1665_9
  simp only [e_1665_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16653)) := by
  have c101835 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101836 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101837 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101838 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101839 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101840 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_10
  simp only [← c101835, ← c101836, ← c101837] at e_1665_10
  have e_1665_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_11
  simp only [← c101838, ← c101839, k2, ← c101840] at e_1665_11
  have hr := e_1665_11
  simp only [e_1665_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16654)) := by
  have c101841 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101842 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101843 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101844 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101845 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101846 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_12
  simp only [← c101841, ← c101842, ← c101843] at e_1665_12
  have e_1665_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_13
  simp only [← c101844, ← c101845, k2, ← c101846] at e_1665_13
  have hr := e_1665_13
  simp only [e_1665_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16655)) := by
  have c101847 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101848 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101849 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101850 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101851 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101852 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_14
  simp only [← c101847, ← c101848, ← c101849] at e_1665_14
  have e_1665_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_15
  simp only [← c101850, ← c101851, k2, ← c101852] at e_1665_15
  have hr := e_1665_15
  simp only [e_1665_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16656)) := by
  have c101853 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101854 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101855 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101856 := (publicBatchWrapper86_copies3183 a h).1
  have c101857 := (publicBatchWrapper86_copies3183 a h).2.1
  have c101858 := (publicBatchWrapper86_copies3183 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_16
  simp only [← c101853, ← c101854, ← c101855] at e_1665_16
  have e_1665_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_17
  simp only [← c101856, ← c101857, k2, ← c101858] at e_1665_17
  have hr := e_1665_17
  simp only [e_1665_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16657)) := by
  have c101859 := (publicBatchWrapper86_copies3183 a h).2.2.2.1
  have c101860 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.1
  have c101861 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.1
  have c101862 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.1
  have c101863 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.1
  have c101864 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_18
  simp only [← c101859, ← c101860, ← c101861] at e_1665_18
  have e_1665_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_19
  simp only [← c101862, ← c101863, k2, ← c101864] at e_1665_19
  have hr := e_1665_19
  simp only [e_1665_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16658)) := by
  have c101865 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.1
  have c101866 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101867 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101868 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101869 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101870 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_0
  simp only [← c101865, ← c101866, ← c101867] at e_1666_0
  have e_1666_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_1
  simp only [← c101868, ← c101869, k2, ← c101870] at e_1666_1
  have hr := e_1666_1
  simp only [e_1666_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16659)) := by
  have c101871 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101872 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101873 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101874 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101875 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101876 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_2
  simp only [← c101871, ← c101872, ← c101873] at e_1666_2
  have e_1666_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_3
  simp only [← c101874, ← c101875, k2, ← c101876] at e_1666_3
  have hr := e_1666_3
  simp only [e_1666_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16660)) := by
  have c101877 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101878 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101879 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101880 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101881 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101882 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_4
  simp only [← c101877, ← c101878, ← c101879] at e_1666_4
  have e_1666_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_5
  simp only [← c101880, ← c101881, k2, ← c101882] at e_1666_5
  have hr := e_1666_5
  simp only [e_1666_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16661)) := by
  have c101883 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101884 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101885 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101886 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101887 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101888 := (publicBatchWrapper86_copies3184 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_6
  simp only [← c101883, ← c101884, ← c101885] at e_1666_6
  have e_1666_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_7
  simp only [← c101886, ← c101887, k2, ← c101888] at e_1666_7
  have hr := e_1666_7
  simp only [e_1666_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16662)) := by
  have c101889 := (publicBatchWrapper86_copies3184 a h).2.1
  have c101890 := (publicBatchWrapper86_copies3184 a h).2.2.1
  have c101891 := (publicBatchWrapper86_copies3184 a h).2.2.2.1
  have c101892 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.1
  have c101893 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.1
  have c101894 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_8
  simp only [← c101889, ← c101890, ← c101891] at e_1666_8
  have e_1666_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_9
  simp only [← c101892, ← c101893, k2, ← c101894] at e_1666_9
  have hr := e_1666_9
  simp only [e_1666_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16663)) := by
  have c101895 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.1
  have c101896 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.1
  have c101897 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.1
  have c101898 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101899 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101900 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_10
  simp only [← c101895, ← c101896, ← c101897] at e_1666_10
  have e_1666_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_11
  simp only [← c101898, ← c101899, k2, ← c101900] at e_1666_11
  have hr := e_1666_11
  simp only [e_1666_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16664)) := by
  have c101901 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101902 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101903 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101904 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101905 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101906 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_12
  simp only [← c101901, ← c101902, ← c101903] at e_1666_12
  have e_1666_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_13
  simp only [← c101904, ← c101905, k2, ← c101906] at e_1666_13
  have hr := e_1666_13
  simp only [e_1666_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16665)) := by
  have c101907 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101908 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101909 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101910 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101911 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101912 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_14
  simp only [← c101907, ← c101908, ← c101909] at e_1666_14
  have e_1666_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_15
  simp only [← c101910, ← c101911, k2, ← c101912] at e_1666_15
  have hr := e_1666_15
  simp only [e_1666_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16666)) := by
  have c101913 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101914 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101915 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101916 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101917 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101918 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_16
  simp only [← c101913, ← c101914, ← c101915] at e_1666_16
  have e_1666_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_17
  simp only [← c101916, ← c101917, k2, ← c101918] at e_1666_17
  have hr := e_1666_17
  simp only [e_1666_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16667)) := by
  have c101919 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101920 := (publicBatchWrapper86_copies3185 a h).1
  have c101921 := (publicBatchWrapper86_copies3185 a h).2.1
  have c101922 := (publicBatchWrapper86_copies3185 a h).2.2.1
  have c101923 := (publicBatchWrapper86_copies3185 a h).2.2.2.1
  have c101924 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_18
  simp only [← c101919, ← c101920, ← c101921] at e_1666_18
  have e_1666_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_19
  simp only [← c101922, ← c101923, k2, ← c101924] at e_1666_19
  have hr := e_1666_19
  simp only [e_1666_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16668)) := by
  have c101925 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.1
  have c101926 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.1
  have c101927 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.1
  have c101928 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.1
  have c101929 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.1
  have c101930 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_0
  simp only [← c101925, ← c101926, ← c101927] at e_1667_0
  have e_1667_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_1
  simp only [← c101928, ← c101929, k2, ← c101930] at e_1667_1
  have hr := e_1667_1
  simp only [e_1667_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16669)) := by
  have c101931 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101932 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101933 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101934 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101935 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101936 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_2
  simp only [← c101931, ← c101932, ← c101933] at e_1667_2
  have e_1667_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_3
  simp only [← c101934, ← c101935, k2, ← c101936] at e_1667_3
  have hr := e_1667_3
  simp only [e_1667_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16670)) := by
  have c101937 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101938 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101939 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101940 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101941 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101942 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_4
  simp only [← c101937, ← c101938, ← c101939] at e_1667_4
  have e_1667_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_5
  simp only [← c101940, ← c101941, k2, ← c101942] at e_1667_5
  have hr := e_1667_5
  simp only [e_1667_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16671)) := by
  have c101943 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101944 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101945 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101946 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101947 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101948 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_6
  simp only [← c101943, ← c101944, ← c101945] at e_1667_6
  have e_1667_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_7
  simp only [← c101946, ← c101947, k2, ← c101948] at e_1667_7
  have hr := e_1667_7
  simp only [e_1667_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16672)) := by
  have c101949 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101950 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101951 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101952 := (publicBatchWrapper86_copies3186 a h).1
  have c101953 := (publicBatchWrapper86_copies3186 a h).2.1
  have c101954 := (publicBatchWrapper86_copies3186 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_8
  simp only [← c101949, ← c101950, ← c101951] at e_1667_8
  have e_1667_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_9
  simp only [← c101952, ← c101953, k2, ← c101954] at e_1667_9
  have hr := e_1667_9
  simp only [e_1667_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16673)) := by
  have c101955 := (publicBatchWrapper86_copies3186 a h).2.2.2.1
  have c101956 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.1
  have c101957 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.1
  have c101958 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.1
  have c101959 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.1
  have c101960 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_10
  simp only [← c101955, ← c101956, ← c101957] at e_1667_10
  have e_1667_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_11
  simp only [← c101958, ← c101959, k2, ← c101960] at e_1667_11
  have hr := e_1667_11
  simp only [e_1667_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16674)) := by
  have c101961 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.1
  have c101962 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101963 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101964 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101965 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101966 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_12
  simp only [← c101961, ← c101962, ← c101963] at e_1667_12
  have e_1667_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_13
  simp only [← c101964, ← c101965, k2, ← c101966] at e_1667_13
  have hr := e_1667_13
  simp only [e_1667_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16675)) := by
  have c101967 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101968 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101969 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101970 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101971 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101972 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_14
  simp only [← c101967, ← c101968, ← c101969] at e_1667_14
  have e_1667_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_15
  simp only [← c101970, ← c101971, k2, ← c101972] at e_1667_15
  have hr := e_1667_15
  simp only [e_1667_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16676)) := by
  have c101973 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101974 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101975 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101976 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101977 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101978 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_16
  simp only [← c101973, ← c101974, ← c101975] at e_1667_16
  have e_1667_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_17
  simp only [← c101976, ← c101977, k2, ← c101978] at e_1667_17
  have hr := e_1667_17
  simp only [e_1667_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16677)) := by
  have c101979 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101980 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101981 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101982 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101983 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101984 := (publicBatchWrapper86_copies3187 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_18
  simp only [← c101979, ← c101980, ← c101981] at e_1667_18
  have e_1667_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_19
  simp only [← c101982, ← c101983, k2, ← c101984] at e_1667_19
  have hr := e_1667_19
  simp only [e_1667_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16678)) := by
  have c101985 := (publicBatchWrapper86_copies3187 a h).2.1
  have c101986 := (publicBatchWrapper86_copies3187 a h).2.2.1
  have c101987 := (publicBatchWrapper86_copies3187 a h).2.2.2.1
  have c101988 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.1
  have c101989 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.1
  have c101990 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_0
  simp only [← c101985, ← c101986, ← c101987] at e_1668_0
  have e_1668_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_1
  simp only [← c101988, ← c101989, k2, ← c101990] at e_1668_1
  have hr := e_1668_1
  simp only [e_1668_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16679)) := by
  have c101991 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.1
  have c101992 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.1
  have c101993 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.1
  have c101994 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101995 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101996 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_2
  simp only [← c101991, ← c101992, ← c101993] at e_1668_2
  have e_1668_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_3
  simp only [← c101994, ← c101995, k2, ← c101996] at e_1668_3
  have hr := e_1668_3
  simp only [e_1668_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16680)) := by
  have c101997 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101998 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101999 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102000 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102001 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102002 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_4
  simp only [← c101997, ← c101998, ← c101999] at e_1668_4
  have e_1668_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_5
  simp only [← c102000, ← c102001, k2, ← c102002] at e_1668_5
  have hr := e_1668_5
  simp only [e_1668_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16681)) := by
  have c102003 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102004 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102005 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102006 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102007 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102008 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_6
  simp only [← c102003, ← c102004, ← c102005] at e_1668_6
  have e_1668_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_7
  simp only [← c102006, ← c102007, k2, ← c102008] at e_1668_7
  have hr := e_1668_7
  simp only [e_1668_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16682)) := by
  have c102009 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102010 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102011 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102012 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102013 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102014 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_8
  simp only [← c102009, ← c102010, ← c102011] at e_1668_8
  have e_1668_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_9
  simp only [← c102012, ← c102013, k2, ← c102014] at e_1668_9
  have hr := e_1668_9
  simp only [e_1668_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16683)) := by
  have c102015 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102016 := (publicBatchWrapper86_copies3188 a h).1
  have c102017 := (publicBatchWrapper86_copies3188 a h).2.1
  have c102018 := (publicBatchWrapper86_copies3188 a h).2.2.1
  have c102019 := (publicBatchWrapper86_copies3188 a h).2.2.2.1
  have c102020 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_10
  simp only [← c102015, ← c102016, ← c102017] at e_1668_10
  have e_1668_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_11
  simp only [← c102018, ← c102019, k2, ← c102020] at e_1668_11
  have hr := e_1668_11
  simp only [e_1668_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16684)) := by
  have c102021 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.1
  have c102022 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.1
  have c102023 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.1
  have c102024 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.1
  have c102025 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.1
  have c102026 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_12
  simp only [← c102021, ← c102022, ← c102023] at e_1668_12
  have e_1668_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_13
  simp only [← c102024, ← c102025, k2, ← c102026] at e_1668_13
  have hr := e_1668_13
  simp only [e_1668_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16685)) := by
  have c102027 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102028 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102029 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102030 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102031 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102032 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_14
  simp only [← c102027, ← c102028, ← c102029] at e_1668_14
  have e_1668_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_15
  simp only [← c102030, ← c102031, k2, ← c102032] at e_1668_15
  have hr := e_1668_15
  simp only [e_1668_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16686)) := by
  have c102033 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102034 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102035 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102036 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102037 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102038 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_16
  simp only [← c102033, ← c102034, ← c102035] at e_1668_16
  have e_1668_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_17
  simp only [← c102036, ← c102037, k2, ← c102038] at e_1668_17
  have hr := e_1668_17
  simp only [e_1668_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16687)) := by
  have c102039 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102040 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102041 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102042 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102043 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102044 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_18
  simp only [← c102039, ← c102040, ← c102041] at e_1668_18
  have e_1668_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_19
  simp only [← c102042, ← c102043, k2, ← c102044] at e_1668_19
  have hr := e_1668_19
  simp only [e_1668_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16688)) := by
  have c102045 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102046 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102047 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102048 := (publicBatchWrapper86_copies3189 a h).1
  have c102049 := (publicBatchWrapper86_copies3189 a h).2.1
  have c102050 := (publicBatchWrapper86_copies3189 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_0
  simp only [← c102045, ← c102046, ← c102047] at e_1669_0
  have e_1669_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_1
  simp only [← c102048, ← c102049, k2, ← c102050] at e_1669_1
  have hr := e_1669_1
  simp only [e_1669_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16689)) := by
  have c102051 := (publicBatchWrapper86_copies3189 a h).2.2.2.1
  have c102052 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.1
  have c102053 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.1
  have c102054 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.1
  have c102055 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.1
  have c102056 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_2
  simp only [← c102051, ← c102052, ← c102053] at e_1669_2
  have e_1669_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_3
  simp only [← c102054, ← c102055, k2, ← c102056] at e_1669_3
  have hr := e_1669_3
  simp only [e_1669_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16690)) := by
  have c102057 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.1
  have c102058 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102059 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102060 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102061 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102062 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_4
  simp only [← c102057, ← c102058, ← c102059] at e_1669_4
  have e_1669_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_5
  simp only [← c102060, ← c102061, k2, ← c102062] at e_1669_5
  have hr := e_1669_5
  simp only [e_1669_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16691)) := by
  have c102063 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102064 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102065 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102066 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102067 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102068 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_6
  simp only [← c102063, ← c102064, ← c102065] at e_1669_6
  have e_1669_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_7
  simp only [← c102066, ← c102067, k2, ← c102068] at e_1669_7
  have hr := e_1669_7
  simp only [e_1669_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16692)) := by
  have c102069 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102070 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102071 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102072 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102073 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102074 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_8
  simp only [← c102069, ← c102070, ← c102071] at e_1669_8
  have e_1669_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_9
  simp only [← c102072, ← c102073, k2, ← c102074] at e_1669_9
  have hr := e_1669_9
  simp only [e_1669_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16693)) := by
  have c102075 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102076 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102077 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102078 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102079 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102080 := (publicBatchWrapper86_copies3190 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_10
  simp only [← c102075, ← c102076, ← c102077] at e_1669_10
  have e_1669_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_11
  simp only [← c102078, ← c102079, k2, ← c102080] at e_1669_11
  have hr := e_1669_11
  simp only [e_1669_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16694)) := by
  have c102081 := (publicBatchWrapper86_copies3190 a h).2.1
  have c102082 := (publicBatchWrapper86_copies3190 a h).2.2.1
  have c102083 := (publicBatchWrapper86_copies3190 a h).2.2.2.1
  have c102084 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.1
  have c102085 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.1
  have c102086 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_12
  simp only [← c102081, ← c102082, ← c102083] at e_1669_12
  have e_1669_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_13
  simp only [← c102084, ← c102085, k2, ← c102086] at e_1669_13
  have hr := e_1669_13
  simp only [e_1669_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16695)) := by
  have c102087 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.1
  have c102088 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.1
  have c102089 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.1
  have c102090 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102091 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102092 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_14
  simp only [← c102087, ← c102088, ← c102089] at e_1669_14
  have e_1669_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_15
  simp only [← c102090, ← c102091, k2, ← c102092] at e_1669_15
  have hr := e_1669_15
  simp only [e_1669_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16696)) := by
  have c102093 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102094 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102095 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102096 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102097 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102098 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_16
  simp only [← c102093, ← c102094, ← c102095] at e_1669_16
  have e_1669_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_17
  simp only [← c102096, ← c102097, k2, ← c102098] at e_1669_17
  have hr := e_1669_17
  simp only [e_1669_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16697)) := by
  have c102099 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102100 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102101 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102102 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102103 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102104 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_18
  simp only [← c102099, ← c102100, ← c102101] at e_1669_18
  have e_1669_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_19
  simp only [← c102102, ← c102103, k2, ← c102104] at e_1669_19
  have hr := e_1669_19
  simp only [e_1669_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16698)) := by
  have c102105 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102106 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102107 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102108 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102109 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102110 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_0
  simp only [← c102105, ← c102106, ← c102107] at e_1670_0
  have e_1670_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_1
  simp only [← c102108, ← c102109, k2, ← c102110] at e_1670_1
  have hr := e_1670_1
  simp only [e_1670_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16699)) := by
  have c102111 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102112 := (publicBatchWrapper86_copies3191 a h).1
  have c102113 := (publicBatchWrapper86_copies3191 a h).2.1
  have c102114 := (publicBatchWrapper86_copies3191 a h).2.2.1
  have c102115 := (publicBatchWrapper86_copies3191 a h).2.2.2.1
  have c102116 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_2
  simp only [← c102111, ← c102112, ← c102113] at e_1670_2
  have e_1670_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_3
  simp only [← c102114, ← c102115, k2, ← c102116] at e_1670_3
  have hr := e_1670_3
  simp only [e_1670_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16700)) := by
  have c102117 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.1
  have c102118 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.1
  have c102119 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.1
  have c102120 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.1
  have c102121 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.1
  have c102122 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_4
  simp only [← c102117, ← c102118, ← c102119] at e_1670_4
  have e_1670_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_5
  simp only [← c102120, ← c102121, k2, ← c102122] at e_1670_5
  have hr := e_1670_5
  simp only [e_1670_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16701)) := by
  have c102123 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102124 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102125 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102126 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102127 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102128 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_6
  simp only [← c102123, ← c102124, ← c102125] at e_1670_6
  have e_1670_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_7
  simp only [← c102126, ← c102127, k2, ← c102128] at e_1670_7
  have hr := e_1670_7
  simp only [e_1670_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16702)) := by
  have c102129 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102130 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102131 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102132 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102133 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102134 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_8
  simp only [← c102129, ← c102130, ← c102131] at e_1670_8
  have e_1670_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_9
  simp only [← c102132, ← c102133, k2, ← c102134] at e_1670_9
  have hr := e_1670_9
  simp only [e_1670_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16703)) := by
  have c102135 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102136 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102137 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102138 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102139 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102140 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_10
  simp only [← c102135, ← c102136, ← c102137] at e_1670_10
  have e_1670_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_11
  simp only [← c102138, ← c102139, k2, ← c102140] at e_1670_11
  have hr := e_1670_11
  simp only [e_1670_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16704)) := by
  have c102141 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102142 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102143 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102144 := (publicBatchWrapper86_copies3192 a h).1
  have c102145 := (publicBatchWrapper86_copies3192 a h).2.1
  have c102146 := (publicBatchWrapper86_copies3192 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_12
  simp only [← c102141, ← c102142, ← c102143] at e_1670_12
  have e_1670_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_13
  simp only [← c102144, ← c102145, k2, ← c102146] at e_1670_13
  have hr := e_1670_13
  simp only [e_1670_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16705)) := by
  have c102147 := (publicBatchWrapper86_copies3192 a h).2.2.2.1
  have c102148 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.1
  have c102149 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.1
  have c102150 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.1
  have c102151 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.1
  have c102152 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_14
  simp only [← c102147, ← c102148, ← c102149] at e_1670_14
  have e_1670_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_15
  simp only [← c102150, ← c102151, k2, ← c102152] at e_1670_15
  have hr := e_1670_15
  simp only [e_1670_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16706)) := by
  have c102153 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.1
  have c102154 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102155 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102156 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102157 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102158 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_16
  simp only [← c102153, ← c102154, ← c102155] at e_1670_16
  have e_1670_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_17
  simp only [← c102156, ← c102157, k2, ← c102158] at e_1670_17
  have hr := e_1670_17
  simp only [e_1670_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16707)) := by
  have c102159 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102160 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102161 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102162 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102163 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102164 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_18
  simp only [← c102159, ← c102160, ← c102161] at e_1670_18
  have e_1670_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_19
  simp only [← c102162, ← c102163, k2, ← c102164] at e_1670_19
  have hr := e_1670_19
  simp only [e_1670_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16708)) := by
  have c102165 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102166 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102167 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102168 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102169 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102170 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_0
  simp only [← c102165, ← c102166, ← c102167] at e_1671_0
  have e_1671_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_1
  simp only [← c102168, ← c102169, k2, ← c102170] at e_1671_1
  have hr := e_1671_1
  simp only [e_1671_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16709)) := by
  have c102171 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102172 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102173 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102174 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102175 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102176 := (publicBatchWrapper86_copies3193 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_2
  simp only [← c102171, ← c102172, ← c102173] at e_1671_2
  have e_1671_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_3
  simp only [← c102174, ← c102175, k2, ← c102176] at e_1671_3
  have hr := e_1671_3
  simp only [e_1671_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16710)) := by
  have c102177 := (publicBatchWrapper86_copies3193 a h).2.1
  have c102178 := (publicBatchWrapper86_copies3193 a h).2.2.1
  have c102179 := (publicBatchWrapper86_copies3193 a h).2.2.2.1
  have c102180 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.1
  have c102181 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.1
  have c102182 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_4
  simp only [← c102177, ← c102178, ← c102179] at e_1671_4
  have e_1671_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_5
  simp only [← c102180, ← c102181, k2, ← c102182] at e_1671_5
  have hr := e_1671_5
  simp only [e_1671_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16711)) := by
  have c102183 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.1
  have c102184 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.1
  have c102185 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.1
  have c102186 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102187 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102188 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_6
  simp only [← c102183, ← c102184, ← c102185] at e_1671_6
  have e_1671_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_7
  simp only [← c102186, ← c102187, k2, ← c102188] at e_1671_7
  have hr := e_1671_7
  simp only [e_1671_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16712)) := by
  have c102189 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102190 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102191 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102192 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102193 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102194 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_8
  simp only [← c102189, ← c102190, ← c102191] at e_1671_8
  have e_1671_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_9
  simp only [← c102192, ← c102193, k2, ← c102194] at e_1671_9
  have hr := e_1671_9
  simp only [e_1671_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16713)) := by
  have c102195 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102196 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102197 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102198 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102199 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102200 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_10
  simp only [← c102195, ← c102196, ← c102197] at e_1671_10
  have e_1671_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_11
  simp only [← c102198, ← c102199, k2, ← c102200] at e_1671_11
  have hr := e_1671_11
  simp only [e_1671_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16714)) := by
  have c102201 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102202 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102203 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102204 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102205 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102206 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_12
  simp only [← c102201, ← c102202, ← c102203] at e_1671_12
  have e_1671_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_13
  simp only [← c102204, ← c102205, k2, ← c102206] at e_1671_13
  have hr := e_1671_13
  simp only [e_1671_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16715)) := by
  have c102207 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102208 := (publicBatchWrapper86_copies3194 a h).1
  have c102209 := (publicBatchWrapper86_copies3194 a h).2.1
  have c102210 := (publicBatchWrapper86_copies3194 a h).2.2.1
  have c102211 := (publicBatchWrapper86_copies3194 a h).2.2.2.1
  have c102212 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_14
  simp only [← c102207, ← c102208, ← c102209] at e_1671_14
  have e_1671_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_15
  simp only [← c102210, ← c102211, k2, ← c102212] at e_1671_15
  have hr := e_1671_15
  simp only [e_1671_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16716)) := by
  have c102213 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.1
  have c102214 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.1
  have c102215 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.1
  have c102216 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.1
  have c102217 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.1
  have c102218 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_16
  simp only [← c102213, ← c102214, ← c102215] at e_1671_16
  have e_1671_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_17
  simp only [← c102216, ← c102217, k2, ← c102218] at e_1671_17
  have hr := e_1671_17
  simp only [e_1671_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16717)) := by
  have c102219 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102220 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102221 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102222 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102223 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102224 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_18
  simp only [← c102219, ← c102220, ← c102221] at e_1671_18
  have e_1671_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_19
  simp only [← c102222, ← c102223, k2, ← c102224] at e_1671_19
  have hr := e_1671_19
  simp only [e_1671_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16718)) := by
  have c102225 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102226 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102227 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102228 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102229 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102230 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_0
  simp only [← c102225, ← c102226, ← c102227] at e_1672_0
  have e_1672_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_1
  simp only [← c102228, ← c102229, k2, ← c102230] at e_1672_1
  have hr := e_1672_1
  simp only [e_1672_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16719)) := by
  have c102231 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102232 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102233 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102234 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102235 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102236 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_2
  simp only [← c102231, ← c102232, ← c102233] at e_1672_2
  have e_1672_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_3
  simp only [← c102234, ← c102235, k2, ← c102236] at e_1672_3
  have hr := e_1672_3
  simp only [e_1672_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16720)) := by
  have c102237 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102238 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102239 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102240 := (publicBatchWrapper86_copies3195 a h).1
  have c102241 := (publicBatchWrapper86_copies3195 a h).2.1
  have c102242 := (publicBatchWrapper86_copies3195 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_4
  simp only [← c102237, ← c102238, ← c102239] at e_1672_4
  have e_1672_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_5
  simp only [← c102240, ← c102241, k2, ← c102242] at e_1672_5
  have hr := e_1672_5
  simp only [e_1672_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16721)) := by
  have c102243 := (publicBatchWrapper86_copies3195 a h).2.2.2.1
  have c102244 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.1
  have c102245 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.1
  have c102246 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.1
  have c102247 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.1
  have c102248 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_6
  simp only [← c102243, ← c102244, ← c102245] at e_1672_6
  have e_1672_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_7
  simp only [← c102246, ← c102247, k2, ← c102248] at e_1672_7
  have hr := e_1672_7
  simp only [e_1672_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16722)) := by
  have c102249 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.1
  have c102250 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102251 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102252 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102253 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102254 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_8
  simp only [← c102249, ← c102250, ← c102251] at e_1672_8
  have e_1672_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_9
  simp only [← c102252, ← c102253, k2, ← c102254] at e_1672_9
  have hr := e_1672_9
  simp only [e_1672_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16723)) := by
  have c102255 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102256 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102257 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102258 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102259 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102260 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_10
  simp only [← c102255, ← c102256, ← c102257] at e_1672_10
  have e_1672_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_11
  simp only [← c102258, ← c102259, k2, ← c102260] at e_1672_11
  have hr := e_1672_11
  simp only [e_1672_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16724)) := by
  have c102261 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102262 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102263 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102264 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102265 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102266 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_12
  simp only [← c102261, ← c102262, ← c102263] at e_1672_12
  have e_1672_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_13
  simp only [← c102264, ← c102265, k2, ← c102266] at e_1672_13
  have hr := e_1672_13
  simp only [e_1672_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16725)) := by
  have c102267 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102268 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102269 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102270 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102271 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102272 := (publicBatchWrapper86_copies3196 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_14
  simp only [← c102267, ← c102268, ← c102269] at e_1672_14
  have e_1672_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_15
  simp only [← c102270, ← c102271, k2, ← c102272] at e_1672_15
  have hr := e_1672_15
  simp only [e_1672_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16726)) := by
  have c102273 := (publicBatchWrapper86_copies3196 a h).2.1
  have c102274 := (publicBatchWrapper86_copies3196 a h).2.2.1
  have c102275 := (publicBatchWrapper86_copies3196 a h).2.2.2.1
  have c102276 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.1
  have c102277 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.1
  have c102278 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_16
  simp only [← c102273, ← c102274, ← c102275] at e_1672_16
  have e_1672_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_17
  simp only [← c102276, ← c102277, k2, ← c102278] at e_1672_17
  have hr := e_1672_17
  simp only [e_1672_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16727)) := by
  have c102279 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.1
  have c102280 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.1
  have c102281 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.1
  have c102282 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102283 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102284 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_18
  simp only [← c102279, ← c102280, ← c102281] at e_1672_18
  have e_1672_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_19
  simp only [← c102282, ← c102283, k2, ← c102284] at e_1672_19
  have hr := e_1672_19
  simp only [e_1672_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16728)) := by
  have c102285 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102286 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102287 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102288 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102289 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102290 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_0
  simp only [← c102285, ← c102286, ← c102287] at e_1673_0
  have e_1673_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_1
  simp only [← c102288, ← c102289, k2, ← c102290] at e_1673_1
  have hr := e_1673_1
  simp only [e_1673_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16729)) := by
  have c102291 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102292 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102293 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102294 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102295 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102296 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_2
  simp only [← c102291, ← c102292, ← c102293] at e_1673_2
  have e_1673_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_3
  simp only [← c102294, ← c102295, k2, ← c102296] at e_1673_3
  have hr := e_1673_3
  simp only [e_1673_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16730)) := by
  have c102297 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102298 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102299 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102300 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102301 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102302 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_4
  simp only [← c102297, ← c102298, ← c102299] at e_1673_4
  have e_1673_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_5
  simp only [← c102300, ← c102301, k2, ← c102302] at e_1673_5
  have hr := e_1673_5
  simp only [e_1673_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16731)) := by
  have c102303 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102304 := (publicBatchWrapper86_copies3197 a h).1
  have c102305 := (publicBatchWrapper86_copies3197 a h).2.1
  have c102306 := (publicBatchWrapper86_copies3197 a h).2.2.1
  have c102307 := (publicBatchWrapper86_copies3197 a h).2.2.2.1
  have c102308 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_6
  simp only [← c102303, ← c102304, ← c102305] at e_1673_6
  have e_1673_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_7
  simp only [← c102306, ← c102307, k2, ← c102308] at e_1673_7
  have hr := e_1673_7
  simp only [e_1673_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16732)) := by
  have c102309 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.1
  have c102310 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.1
  have c102311 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.1
  have c102312 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.1
  have c102313 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.1
  have c102314 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_8
  simp only [← c102309, ← c102310, ← c102311] at e_1673_8
  have e_1673_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_9
  simp only [← c102312, ← c102313, k2, ← c102314] at e_1673_9
  have hr := e_1673_9
  simp only [e_1673_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16733)) := by
  have c102315 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102316 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102317 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102318 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102319 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102320 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_10
  simp only [← c102315, ← c102316, ← c102317] at e_1673_10
  have e_1673_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_11
  simp only [← c102318, ← c102319, k2, ← c102320] at e_1673_11
  have hr := e_1673_11
  simp only [e_1673_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16734)) := by
  have c102321 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102322 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102323 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102324 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102325 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102326 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_12
  simp only [← c102321, ← c102322, ← c102323] at e_1673_12
  have e_1673_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_13
  simp only [← c102324, ← c102325, k2, ← c102326] at e_1673_13
  have hr := e_1673_13
  simp only [e_1673_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16735)) := by
  have c102327 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102328 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102329 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102330 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102331 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102332 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_14
  simp only [← c102327, ← c102328, ← c102329] at e_1673_14
  have e_1673_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_15
  simp only [← c102330, ← c102331, k2, ← c102332] at e_1673_15
  have hr := e_1673_15
  simp only [e_1673_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16736)) := by
  have c102333 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102334 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102335 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102336 := (publicBatchWrapper86_copies3198 a h).1
  have c102337 := (publicBatchWrapper86_copies3198 a h).2.1
  have c102338 := (publicBatchWrapper86_copies3198 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_16
  simp only [← c102333, ← c102334, ← c102335] at e_1673_16
  have e_1673_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_17
  simp only [← c102336, ← c102337, k2, ← c102338] at e_1673_17
  have hr := e_1673_17
  simp only [e_1673_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16737)) := by
  have c102339 := (publicBatchWrapper86_copies3198 a h).2.2.2.1
  have c102340 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.1
  have c102341 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.1
  have c102342 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.1
  have c102343 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.1
  have c102344 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_18
  simp only [← c102339, ← c102340, ← c102341] at e_1673_18
  have e_1673_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_19
  simp only [← c102342, ← c102343, k2, ← c102344] at e_1673_19
  have hr := e_1673_19
  simp only [e_1673_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16738)) := by
  have c102345 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.1
  have c102346 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102347 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102348 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102349 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102350 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_0
  simp only [← c102345, ← c102346, ← c102347] at e_1674_0
  have e_1674_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_1
  simp only [← c102348, ← c102349, k2, ← c102350] at e_1674_1
  have hr := e_1674_1
  simp only [e_1674_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16739)) := by
  have c102351 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102352 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102353 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102354 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102355 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102356 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_2
  simp only [← c102351, ← c102352, ← c102353] at e_1674_2
  have e_1674_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_3
  simp only [← c102354, ← c102355, k2, ← c102356] at e_1674_3
  have hr := e_1674_3
  simp only [e_1674_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16740)) := by
  have c102357 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102358 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102359 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102360 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102361 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102362 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_4
  simp only [← c102357, ← c102358, ← c102359] at e_1674_4
  have e_1674_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_5
  simp only [← c102360, ← c102361, k2, ← c102362] at e_1674_5
  have hr := e_1674_5
  simp only [e_1674_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16741)) := by
  have c102363 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102364 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102365 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102366 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102367 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102368 := (publicBatchWrapper86_copies3199 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_6
  simp only [← c102363, ← c102364, ← c102365] at e_1674_6
  have e_1674_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_7
  simp only [← c102366, ← c102367, k2, ← c102368] at e_1674_7
  have hr := e_1674_7
  simp only [e_1674_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16742)) := by
  have c102369 := (publicBatchWrapper86_copies3199 a h).2.1
  have c102370 := (publicBatchWrapper86_copies3199 a h).2.2.1
  have c102371 := (publicBatchWrapper86_copies3199 a h).2.2.2.1
  have c102372 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.1
  have c102373 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.1
  have c102374 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_8
  simp only [← c102369, ← c102370, ← c102371] at e_1674_8
  have e_1674_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_9
  simp only [← c102372, ← c102373, k2, ← c102374] at e_1674_9
  have hr := e_1674_9
  simp only [e_1674_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16743)) := by
  have c102375 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.1
  have c102376 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.1
  have c102377 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.1
  have c102378 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102379 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102380 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_10
  simp only [← c102375, ← c102376, ← c102377] at e_1674_10
  have e_1674_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_11
  simp only [← c102378, ← c102379, k2, ← c102380] at e_1674_11
  have hr := e_1674_11
  simp only [e_1674_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16744)) := by
  have c102381 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102382 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102383 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102384 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102385 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102386 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_12
  simp only [← c102381, ← c102382, ← c102383] at e_1674_12
  have e_1674_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_13
  simp only [← c102384, ← c102385, k2, ← c102386] at e_1674_13
  have hr := e_1674_13
  simp only [e_1674_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16745)) := by
  have c102387 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102388 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102389 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102390 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102391 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102392 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_14
  simp only [← c102387, ← c102388, ← c102389] at e_1674_14
  have e_1674_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_15
  simp only [← c102390, ← c102391, k2, ← c102392] at e_1674_15
  have hr := e_1674_15
  simp only [e_1674_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16746)) := by
  have c102393 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102394 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102395 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102396 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102397 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102398 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_16
  simp only [← c102393, ← c102394, ← c102395] at e_1674_16
  have e_1674_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_17
  simp only [← c102396, ← c102397, k2, ← c102398] at e_1674_17
  have hr := e_1674_17
  simp only [e_1674_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16747)) := by
  have c102399 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102400 := (publicBatchWrapper86_copies3200 a h).1
  have c102401 := (publicBatchWrapper86_copies3200 a h).2.1
  have c102402 := (publicBatchWrapper86_copies3200 a h).2.2.1
  have c102403 := (publicBatchWrapper86_copies3200 a h).2.2.2.1
  have c102404 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_18
  simp only [← c102399, ← c102400, ← c102401] at e_1674_18
  have e_1674_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_19
  simp only [← c102402, ← c102403, k2, ← c102404] at e_1674_19
  have hr := e_1674_19
  simp only [e_1674_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16748)) := by
  have c102405 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.1
  have c102406 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.1
  have c102407 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.1
  have c102408 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.1
  have c102409 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.1
  have c102410 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_0
  simp only [← c102405, ← c102406, ← c102407] at e_1675_0
  have e_1675_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_1
  simp only [← c102408, ← c102409, k2, ← c102410] at e_1675_1
  have hr := e_1675_1
  simp only [e_1675_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16749)) := by
  have c102411 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102412 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102413 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102414 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102415 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102416 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_2
  simp only [← c102411, ← c102412, ← c102413] at e_1675_2
  have e_1675_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_3
  simp only [← c102414, ← c102415, k2, ← c102416] at e_1675_3
  have hr := e_1675_3
  simp only [e_1675_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16750)) := by
  have c102417 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102418 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102419 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102420 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102421 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102422 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_4
  simp only [← c102417, ← c102418, ← c102419] at e_1675_4
  have e_1675_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_5
  simp only [← c102420, ← c102421, k2, ← c102422] at e_1675_5
  have hr := e_1675_5
  simp only [e_1675_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16751)) := by
  have c102423 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102424 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102425 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102426 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102427 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102428 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_6
  simp only [← c102423, ← c102424, ← c102425] at e_1675_6
  have e_1675_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_7
  simp only [← c102426, ← c102427, k2, ← c102428] at e_1675_7
  have hr := e_1675_7
  simp only [e_1675_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16752)) := by
  have c102429 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102430 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102431 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102432 := (publicBatchWrapper86_copies3201 a h).1
  have c102433 := (publicBatchWrapper86_copies3201 a h).2.1
  have c102434 := (publicBatchWrapper86_copies3201 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_8
  simp only [← c102429, ← c102430, ← c102431] at e_1675_8
  have e_1675_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_9
  simp only [← c102432, ← c102433, k2, ← c102434] at e_1675_9
  have hr := e_1675_9
  simp only [e_1675_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16753)) := by
  have c102435 := (publicBatchWrapper86_copies3201 a h).2.2.2.1
  have c102436 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.1
  have c102437 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.1
  have c102438 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.1
  have c102439 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.1
  have c102440 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_10
  simp only [← c102435, ← c102436, ← c102437] at e_1675_10
  have e_1675_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_11
  simp only [← c102438, ← c102439, k2, ← c102440] at e_1675_11
  have hr := e_1675_11
  simp only [e_1675_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16754)) := by
  have c102441 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.1
  have c102442 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102443 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102444 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102445 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102446 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_12
  simp only [← c102441, ← c102442, ← c102443] at e_1675_12
  have e_1675_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_13
  simp only [← c102444, ← c102445, k2, ← c102446] at e_1675_13
  have hr := e_1675_13
  simp only [e_1675_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16755)) := by
  have c102447 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102448 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102449 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102450 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102451 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102452 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_14
  simp only [← c102447, ← c102448, ← c102449] at e_1675_14
  have e_1675_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_15
  simp only [← c102450, ← c102451, k2, ← c102452] at e_1675_15
  have hr := e_1675_15
  simp only [e_1675_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16756)) := by
  have c102453 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102454 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102455 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102456 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102457 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102458 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_16
  simp only [← c102453, ← c102454, ← c102455] at e_1675_16
  have e_1675_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_17
  simp only [← c102456, ← c102457, k2, ← c102458] at e_1675_17
  have hr := e_1675_17
  simp only [e_1675_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16757)) := by
  have c102459 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102460 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102461 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102462 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102463 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102464 := (publicBatchWrapper86_copies3202 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_18
  simp only [← c102459, ← c102460, ← c102461] at e_1675_18
  have e_1675_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_19
  simp only [← c102462, ← c102463, k2, ← c102464] at e_1675_19
  have hr := e_1675_19
  simp only [e_1675_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16758)) := by
  have c102465 := (publicBatchWrapper86_copies3202 a h).2.1
  have c102466 := (publicBatchWrapper86_copies3202 a h).2.2.1
  have c102467 := (publicBatchWrapper86_copies3202 a h).2.2.2.1
  have c102468 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.1
  have c102469 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.1
  have c102470 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_0
  simp only [← c102465, ← c102466, ← c102467] at e_1676_0
  have e_1676_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_1
  simp only [← c102468, ← c102469, k2, ← c102470] at e_1676_1
  have hr := e_1676_1
  simp only [e_1676_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16759)) := by
  have c102471 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.1
  have c102472 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.1
  have c102473 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.1
  have c102474 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102475 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102476 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_2
  simp only [← c102471, ← c102472, ← c102473] at e_1676_2
  have e_1676_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_3
  simp only [← c102474, ← c102475, k2, ← c102476] at e_1676_3
  have hr := e_1676_3
  simp only [e_1676_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16760)) := by
  have c102477 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102478 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102479 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102480 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102481 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102482 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_4
  simp only [← c102477, ← c102478, ← c102479] at e_1676_4
  have e_1676_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_5
  simp only [← c102480, ← c102481, k2, ← c102482] at e_1676_5
  have hr := e_1676_5
  simp only [e_1676_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16761)) := by
  have c102483 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102484 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102485 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102486 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102487 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102488 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_6
  simp only [← c102483, ← c102484, ← c102485] at e_1676_6
  have e_1676_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_7
  simp only [← c102486, ← c102487, k2, ← c102488] at e_1676_7
  have hr := e_1676_7
  simp only [e_1676_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16762)) := by
  have c102489 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102490 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102491 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102492 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102493 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102494 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_8
  simp only [← c102489, ← c102490, ← c102491] at e_1676_8
  have e_1676_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_9
  simp only [← c102492, ← c102493, k2, ← c102494] at e_1676_9
  have hr := e_1676_9
  simp only [e_1676_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16763)) := by
  have c102495 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102496 := (publicBatchWrapper86_copies3203 a h).1
  have c102497 := (publicBatchWrapper86_copies3203 a h).2.1
  have c102498 := (publicBatchWrapper86_copies3203 a h).2.2.1
  have c102499 := (publicBatchWrapper86_copies3203 a h).2.2.2.1
  have c102500 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_10
  simp only [← c102495, ← c102496, ← c102497] at e_1676_10
  have e_1676_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_11
  simp only [← c102498, ← c102499, k2, ← c102500] at e_1676_11
  have hr := e_1676_11
  simp only [e_1676_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16764)) := by
  have c102501 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.1
  have c102502 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.1
  have c102503 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.1
  have c102504 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.1
  have c102505 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.1
  have c102506 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_12
  simp only [← c102501, ← c102502, ← c102503] at e_1676_12
  have e_1676_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_13
  simp only [← c102504, ← c102505, k2, ← c102506] at e_1676_13
  have hr := e_1676_13
  simp only [e_1676_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16765)) := by
  have c102507 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102508 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102509 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102510 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102511 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102512 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_14
  simp only [← c102507, ← c102508, ← c102509] at e_1676_14
  have e_1676_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_15
  simp only [← c102510, ← c102511, k2, ← c102512] at e_1676_15
  have hr := e_1676_15
  simp only [e_1676_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16766)) := by
  have c102513 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102514 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102515 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102516 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102517 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102518 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_16
  simp only [← c102513, ← c102514, ← c102515] at e_1676_16
  have e_1676_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_17
  simp only [← c102516, ← c102517, k2, ← c102518] at e_1676_17
  have hr := e_1676_17
  simp only [e_1676_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16767)) := by
  have c102519 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102520 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102521 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102522 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102523 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102524 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_18
  simp only [← c102519, ← c102520, ← c102521] at e_1676_18
  have e_1676_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_19
  simp only [← c102522, ← c102523, k2, ← c102524] at e_1676_19
  have hr := e_1676_19
  simp only [e_1676_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16768)) := by
  have c102525 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102526 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102527 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102528 := (publicBatchWrapper86_copies3204 a h).1
  have c102529 := (publicBatchWrapper86_copies3204 a h).2.1
  have c102530 := (publicBatchWrapper86_copies3204 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_0
  simp only [← c102525, ← c102526, ← c102527] at e_1677_0
  have e_1677_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_1
  simp only [← c102528, ← c102529, k2, ← c102530] at e_1677_1
  have hr := e_1677_1
  simp only [e_1677_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16769)) := by
  have c102531 := (publicBatchWrapper86_copies3204 a h).2.2.2.1
  have c102532 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.1
  have c102533 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.1
  have c102534 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.1
  have c102535 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.1
  have c102536 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_2
  simp only [← c102531, ← c102532, ← c102533] at e_1677_2
  have e_1677_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_3
  simp only [← c102534, ← c102535, k2, ← c102536] at e_1677_3
  have hr := e_1677_3
  simp only [e_1677_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16810)) := by
  have c102537 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.1
  have c102538 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102539 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102540 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102541 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102542 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_4
  simp only [← c102537, ← c102538, ← c102539] at e_1677_4
  have e_1677_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_5
  simp only [← c102540, ← c102541, k2, ← c102542] at e_1677_5
  have hr := e_1677_5
  simp only [e_1677_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16811)) := by
  have c102543 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102544 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102545 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102546 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102547 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102548 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_6
  simp only [← c102543, ← c102544, ← c102545] at e_1677_6
  have e_1677_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_7
  simp only [← c102546, ← c102547, k2, ← c102548] at e_1677_7
  have hr := e_1677_7
  simp only [e_1677_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16812)) := by
  have c102549 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102550 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102551 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102552 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102553 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102554 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_8
  simp only [← c102549, ← c102550, ← c102551] at e_1677_8
  have e_1677_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_9
  simp only [← c102552, ← c102553, k2, ← c102554] at e_1677_9
  have hr := e_1677_9
  simp only [e_1677_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16813)) := by
  have c102555 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102556 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102557 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102558 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102559 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102560 := (publicBatchWrapper86_copies3205 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_10
  simp only [← c102555, ← c102556, ← c102557] at e_1677_10
  have e_1677_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_11
  simp only [← c102558, ← c102559, k2, ← c102560] at e_1677_11
  have hr := e_1677_11
  simp only [e_1677_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16814)) := by
  have c102561 := (publicBatchWrapper86_copies3205 a h).2.1
  have c102562 := (publicBatchWrapper86_copies3205 a h).2.2.1
  have c102563 := (publicBatchWrapper86_copies3205 a h).2.2.2.1
  have c102564 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.1
  have c102565 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.1
  have c102566 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_12
  simp only [← c102561, ← c102562, ← c102563] at e_1677_12
  have e_1677_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_13
  simp only [← c102564, ← c102565, k2, ← c102566] at e_1677_13
  have hr := e_1677_13
  simp only [e_1677_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16815)) := by
  have c102567 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.1
  have c102568 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.1
  have c102569 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.1
  have c102570 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102571 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102572 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_14
  simp only [← c102567, ← c102568, ← c102569] at e_1677_14
  have e_1677_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_15
  simp only [← c102570, ← c102571, k2, ← c102572] at e_1677_15
  have hr := e_1677_15
  simp only [e_1677_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16816)) := by
  have c102573 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102574 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102575 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102576 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102577 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102578 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_16
  simp only [← c102573, ← c102574, ← c102575] at e_1677_16
  have e_1677_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_17
  simp only [← c102576, ← c102577, k2, ← c102578] at e_1677_17
  have hr := e_1677_17
  simp only [e_1677_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16817)) := by
  have c102579 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102580 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102581 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102582 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102583 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102584 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_18
  simp only [← c102579, ← c102580, ← c102581] at e_1677_18
  have e_1677_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_19
  simp only [← c102582, ← c102583, k2, ← c102584] at e_1677_19
  have hr := e_1677_19
  simp only [e_1677_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16818)) := by
  have c102585 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102586 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102587 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102588 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102589 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102590 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_0
  simp only [← c102585, ← c102586, ← c102587] at e_1678_0
  have e_1678_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_1
  simp only [← c102588, ← c102589, k2, ← c102590] at e_1678_1
  have hr := e_1678_1
  simp only [e_1678_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16819)) := by
  have c102591 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102592 := (publicBatchWrapper86_copies3206 a h).1
  have c102593 := (publicBatchWrapper86_copies3206 a h).2.1
  have c102594 := (publicBatchWrapper86_copies3206 a h).2.2.1
  have c102595 := (publicBatchWrapper86_copies3206 a h).2.2.2.1
  have c102596 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_2
  simp only [← c102591, ← c102592, ← c102593] at e_1678_2
  have e_1678_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_3
  simp only [← c102594, ← c102595, k2, ← c102596] at e_1678_3
  have hr := e_1678_3
  simp only [e_1678_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16820)) := by
  have c102597 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.1
  have c102598 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.1
  have c102599 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.1
  have c102600 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.1
  have c102601 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.1
  have c102602 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_4
  simp only [← c102597, ← c102598, ← c102599] at e_1678_4
  have e_1678_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_5
  simp only [← c102600, ← c102601, k2, ← c102602] at e_1678_5
  have hr := e_1678_5
  simp only [e_1678_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16821)) := by
  have c102603 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102604 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102605 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102606 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102607 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102608 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_6
  simp only [← c102603, ← c102604, ← c102605] at e_1678_6
  have e_1678_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_7
  simp only [← c102606, ← c102607, k2, ← c102608] at e_1678_7
  have hr := e_1678_7
  simp only [e_1678_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16822)) := by
  have c102609 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102610 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102611 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102612 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102613 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102614 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_8
  simp only [← c102609, ← c102610, ← c102611] at e_1678_8
  have e_1678_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_9
  simp only [← c102612, ← c102613, k2, ← c102614] at e_1678_9
  have hr := e_1678_9
  simp only [e_1678_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16823)) := by
  have c102615 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102616 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102617 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102618 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102619 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102620 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_10
  simp only [← c102615, ← c102616, ← c102617] at e_1678_10
  have e_1678_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_11
  simp only [← c102618, ← c102619, k2, ← c102620] at e_1678_11
  have hr := e_1678_11
  simp only [e_1678_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16824)) := by
  have c102621 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102622 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102623 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102624 := (publicBatchWrapper86_copies3207 a h).1
  have c102625 := (publicBatchWrapper86_copies3207 a h).2.1
  have c102626 := (publicBatchWrapper86_copies3207 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_12
  simp only [← c102621, ← c102622, ← c102623] at e_1678_12
  have e_1678_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_13
  simp only [← c102624, ← c102625, k2, ← c102626] at e_1678_13
  have hr := e_1678_13
  simp only [e_1678_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16825)) := by
  have c102627 := (publicBatchWrapper86_copies3207 a h).2.2.2.1
  have c102628 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.1
  have c102629 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.1
  have c102630 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.1
  have c102631 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.1
  have c102632 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_14
  simp only [← c102627, ← c102628, ← c102629] at e_1678_14
  have e_1678_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_15
  simp only [← c102630, ← c102631, k2, ← c102632] at e_1678_15
  have hr := e_1678_15
  simp only [e_1678_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16826)) := by
  have c102633 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.1
  have c102634 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102635 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102636 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102637 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102638 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_16
  simp only [← c102633, ← c102634, ← c102635] at e_1678_16
  have e_1678_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_17
  simp only [← c102636, ← c102637, k2, ← c102638] at e_1678_17
  have hr := e_1678_17
  simp only [e_1678_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16827)) := by
  have c102639 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102640 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102641 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102642 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102643 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102644 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_18
  simp only [← c102639, ← c102640, ← c102641] at e_1678_18
  have e_1678_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_19
  simp only [← c102642, ← c102643, k2, ← c102644] at e_1678_19
  have hr := e_1678_19
  simp only [e_1678_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16828)) := by
  have c102645 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102646 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102647 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102648 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102649 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102650 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_0
  simp only [← c102645, ← c102646, ← c102647] at e_1679_0
  have e_1679_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_1
  simp only [← c102648, ← c102649, k2, ← c102650] at e_1679_1
  have hr := e_1679_1
  simp only [e_1679_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16829)) := by
  have c102651 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102652 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102653 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102654 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102655 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102656 := (publicBatchWrapper86_copies3208 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_2
  simp only [← c102651, ← c102652, ← c102653] at e_1679_2
  have e_1679_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_3
  simp only [← c102654, ← c102655, k2, ← c102656] at e_1679_3
  have hr := e_1679_3
  simp only [e_1679_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16830)) := by
  have c102657 := (publicBatchWrapper86_copies3208 a h).2.1
  have c102658 := (publicBatchWrapper86_copies3208 a h).2.2.1
  have c102659 := (publicBatchWrapper86_copies3208 a h).2.2.2.1
  have c102660 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.1
  have c102661 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.1
  have c102662 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_4
  simp only [← c102657, ← c102658, ← c102659] at e_1679_4
  have e_1679_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_5
  simp only [← c102660, ← c102661, k2, ← c102662] at e_1679_5
  have hr := e_1679_5
  simp only [e_1679_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16831)) := by
  have c102663 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.1
  have c102664 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.1
  have c102665 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.1
  have c102666 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102667 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102668 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_6
  simp only [← c102663, ← c102664, ← c102665] at e_1679_6
  have e_1679_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_7
  simp only [← c102666, ← c102667, k2, ← c102668] at e_1679_7
  have hr := e_1679_7
  simp only [e_1679_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16832)) := by
  have c102669 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102670 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102671 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102672 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102673 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102674 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_8
  simp only [← c102669, ← c102670, ← c102671] at e_1679_8
  have e_1679_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_9
  simp only [← c102672, ← c102673, k2, ← c102674] at e_1679_9
  have hr := e_1679_9
  simp only [e_1679_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16833)) := by
  have c102675 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102676 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102677 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102678 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102679 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102680 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_10
  simp only [← c102675, ← c102676, ← c102677] at e_1679_10
  have e_1679_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_11
  simp only [← c102678, ← c102679, k2, ← c102680] at e_1679_11
  have hr := e_1679_11
  simp only [e_1679_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16834)) := by
  have c102681 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102682 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102683 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102684 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102685 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102686 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_12
  simp only [← c102681, ← c102682, ← c102683] at e_1679_12
  have e_1679_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_13
  simp only [← c102684, ← c102685, k2, ← c102686] at e_1679_13
  have hr := e_1679_13
  simp only [e_1679_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16835)) := by
  have c102687 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102688 := (publicBatchWrapper86_copies3209 a h).1
  have c102689 := (publicBatchWrapper86_copies3209 a h).2.1
  have c102690 := (publicBatchWrapper86_copies3209 a h).2.2.1
  have c102691 := (publicBatchWrapper86_copies3209 a h).2.2.2.1
  have c102692 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_14
  simp only [← c102687, ← c102688, ← c102689] at e_1679_14
  have e_1679_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_15
  simp only [← c102690, ← c102691, k2, ← c102692] at e_1679_15
  have hr := e_1679_15
  simp only [e_1679_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16836)) := by
  have c102693 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.1
  have c102694 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.1
  have c102695 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.1
  have c102696 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.1
  have c102697 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.1
  have c102698 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_16
  simp only [← c102693, ← c102694, ← c102695] at e_1679_16
  have e_1679_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_17
  simp only [← c102696, ← c102697, k2, ← c102698] at e_1679_17
  have hr := e_1679_17
  simp only [e_1679_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16837)) := by
  have c102699 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102700 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102701 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102702 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102703 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102704 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_18
  simp only [← c102699, ← c102700, ← c102701] at e_1679_18
  have e_1679_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_19
  simp only [← c102702, ← c102703, k2, ← c102704] at e_1679_19
  have hr := e_1679_19
  simp only [e_1679_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16838)) := by
  have c102705 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102706 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102707 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102708 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102709 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102710 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_0
  simp only [← c102705, ← c102706, ← c102707] at e_1680_0
  have e_1680_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_1
  simp only [← c102708, ← c102709, k2, ← c102710] at e_1680_1
  have hr := e_1680_1
  simp only [e_1680_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16839)) := by
  have c102711 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102712 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102713 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102714 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102715 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102716 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_2
  simp only [← c102711, ← c102712, ← c102713] at e_1680_2
  have e_1680_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_3
  simp only [← c102714, ← c102715, k2, ← c102716] at e_1680_3
  have hr := e_1680_3
  simp only [e_1680_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16840)) := by
  have c102717 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102718 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102719 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102720 := (publicBatchWrapper86_copies3210 a h).1
  have c102721 := (publicBatchWrapper86_copies3210 a h).2.1
  have c102722 := (publicBatchWrapper86_copies3210 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_4
  simp only [← c102717, ← c102718, ← c102719] at e_1680_4
  have e_1680_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_5
  simp only [← c102720, ← c102721, k2, ← c102722] at e_1680_5
  have hr := e_1680_5
  simp only [e_1680_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16841)) := by
  have c102723 := (publicBatchWrapper86_copies3210 a h).2.2.2.1
  have c102724 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.1
  have c102725 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.1
  have c102726 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.1
  have c102727 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.1
  have c102728 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_6
  simp only [← c102723, ← c102724, ← c102725] at e_1680_6
  have e_1680_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_7
  simp only [← c102726, ← c102727, k2, ← c102728] at e_1680_7
  have hr := e_1680_7
  simp only [e_1680_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16842)) := by
  have c102729 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.1
  have c102730 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102731 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102732 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102733 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102734 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_8
  simp only [← c102729, ← c102730, ← c102731] at e_1680_8
  have e_1680_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_9
  simp only [← c102732, ← c102733, k2, ← c102734] at e_1680_9
  have hr := e_1680_9
  simp only [e_1680_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16843)) := by
  have c102735 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102736 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102737 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102738 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102739 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102740 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_10
  simp only [← c102735, ← c102736, ← c102737] at e_1680_10
  have e_1680_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_11
  simp only [← c102738, ← c102739, k2, ← c102740] at e_1680_11
  have hr := e_1680_11
  simp only [e_1680_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16844)) := by
  have c102741 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102742 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102743 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102744 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102745 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102746 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_12
  simp only [← c102741, ← c102742, ← c102743] at e_1680_12
  have e_1680_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_13
  simp only [← c102744, ← c102745, k2, ← c102746] at e_1680_13
  have hr := e_1680_13
  simp only [e_1680_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16845)) := by
  have c102747 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102748 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102749 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102750 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102751 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102752 := (publicBatchWrapper86_copies3211 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_14
  simp only [← c102747, ← c102748, ← c102749] at e_1680_14
  have e_1680_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_15
  simp only [← c102750, ← c102751, k2, ← c102752] at e_1680_15
  have hr := e_1680_15
  simp only [e_1680_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16846)) := by
  have c102753 := (publicBatchWrapper86_copies3211 a h).2.1
  have c102754 := (publicBatchWrapper86_copies3211 a h).2.2.1
  have c102755 := (publicBatchWrapper86_copies3211 a h).2.2.2.1
  have c102756 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.1
  have c102757 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.1
  have c102758 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_16
  simp only [← c102753, ← c102754, ← c102755] at e_1680_16
  have e_1680_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_17
  simp only [← c102756, ← c102757, k2, ← c102758] at e_1680_17
  have hr := e_1680_17
  simp only [e_1680_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16847)) := by
  have c102759 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.1
  have c102760 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.1
  have c102761 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.1
  have c102762 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102763 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102764 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_18
  simp only [← c102759, ← c102760, ← c102761] at e_1680_18
  have e_1680_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_19
  simp only [← c102762, ← c102763, k2, ← c102764] at e_1680_19
  have hr := e_1680_19
  simp only [e_1680_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16848)) := by
  have c102765 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102766 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102767 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102768 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102769 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102770 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_0
  simp only [← c102765, ← c102766, ← c102767] at e_1681_0
  have e_1681_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_1
  simp only [← c102768, ← c102769, k2, ← c102770] at e_1681_1
  have hr := e_1681_1
  simp only [e_1681_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16849)) := by
  have c102771 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102772 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102773 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102774 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102775 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102776 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_2
  simp only [← c102771, ← c102772, ← c102773] at e_1681_2
  have e_1681_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_3
  simp only [← c102774, ← c102775, k2, ← c102776] at e_1681_3
  have hr := e_1681_3
  simp only [e_1681_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16850)) := by
  have c102777 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102778 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102779 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102780 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102781 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102782 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_4
  simp only [← c102777, ← c102778, ← c102779] at e_1681_4
  have e_1681_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_5
  simp only [← c102780, ← c102781, k2, ← c102782] at e_1681_5
  have hr := e_1681_5
  simp only [e_1681_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16851)) := by
  have c102783 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102784 := (publicBatchWrapper86_copies3212 a h).1
  have c102785 := (publicBatchWrapper86_copies3212 a h).2.1
  have c102786 := (publicBatchWrapper86_copies3212 a h).2.2.1
  have c102787 := (publicBatchWrapper86_copies3212 a h).2.2.2.1
  have c102788 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_6
  simp only [← c102783, ← c102784, ← c102785] at e_1681_6
  have e_1681_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_7
  simp only [← c102786, ← c102787, k2, ← c102788] at e_1681_7
  have hr := e_1681_7
  simp only [e_1681_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16852)) := by
  have c102789 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.1
  have c102790 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.1
  have c102791 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.1
  have c102792 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.1
  have c102793 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.1
  have c102794 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_8
  simp only [← c102789, ← c102790, ← c102791] at e_1681_8
  have e_1681_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_9
  simp only [← c102792, ← c102793, k2, ← c102794] at e_1681_9
  have hr := e_1681_9
  simp only [e_1681_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16853)) := by
  have c102795 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102796 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102797 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102798 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102799 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102800 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_10
  simp only [← c102795, ← c102796, ← c102797] at e_1681_10
  have e_1681_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_11
  simp only [← c102798, ← c102799, k2, ← c102800] at e_1681_11
  have hr := e_1681_11
  simp only [e_1681_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16854)) := by
  have c102801 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102802 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102803 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102804 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102805 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102806 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_12
  simp only [← c102801, ← c102802, ← c102803] at e_1681_12
  have e_1681_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_13
  simp only [← c102804, ← c102805, k2, ← c102806] at e_1681_13
  have hr := e_1681_13
  simp only [e_1681_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16855)) := by
  have c102807 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102808 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102809 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102810 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102811 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102812 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_14
  simp only [← c102807, ← c102808, ← c102809] at e_1681_14
  have e_1681_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_15
  simp only [← c102810, ← c102811, k2, ← c102812] at e_1681_15
  have hr := e_1681_15
  simp only [e_1681_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16856)) := by
  have c102813 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102814 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102815 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102816 := (publicBatchWrapper86_copies3213 a h).1
  have c102817 := (publicBatchWrapper86_copies3213 a h).2.1
  have c102818 := (publicBatchWrapper86_copies3213 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_16
  simp only [← c102813, ← c102814, ← c102815] at e_1681_16
  have e_1681_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_17
  simp only [← c102816, ← c102817, k2, ← c102818] at e_1681_17
  have hr := e_1681_17
  simp only [e_1681_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16857)) := by
  have c102819 := (publicBatchWrapper86_copies3213 a h).2.2.2.1
  have c102820 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.1
  have c102821 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.1
  have c102822 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.1
  have c102823 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.1
  have c102824 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_18
  simp only [← c102819, ← c102820, ← c102821] at e_1681_18
  have e_1681_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_19
  simp only [← c102822, ← c102823, k2, ← c102824] at e_1681_19
  have hr := e_1681_19
  simp only [e_1681_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16858)) := by
  have c102825 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.1
  have c102826 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102827 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102828 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102829 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102830 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_0
  simp only [← c102825, ← c102826, ← c102827] at e_1682_0
  have e_1682_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_1
  simp only [← c102828, ← c102829, k2, ← c102830] at e_1682_1
  have hr := e_1682_1
  simp only [e_1682_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16859)) := by
  have c102831 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102832 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102833 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102834 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102835 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102836 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_2
  simp only [← c102831, ← c102832, ← c102833] at e_1682_2
  have e_1682_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_3
  simp only [← c102834, ← c102835, k2, ← c102836] at e_1682_3
  have hr := e_1682_3
  simp only [e_1682_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16860)) := by
  have c102837 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102838 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102839 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102840 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102841 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102842 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_4
  simp only [← c102837, ← c102838, ← c102839] at e_1682_4
  have e_1682_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_5
  simp only [← c102840, ← c102841, k2, ← c102842] at e_1682_5
  have hr := e_1682_5
  simp only [e_1682_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16861)) := by
  have c102843 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102844 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102845 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102846 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102847 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102848 := (publicBatchWrapper86_copies3214 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_6
  simp only [← c102843, ← c102844, ← c102845] at e_1682_6
  have e_1682_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_7
  simp only [← c102846, ← c102847, k2, ← c102848] at e_1682_7
  have hr := e_1682_7
  simp only [e_1682_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16862)) := by
  have c102849 := (publicBatchWrapper86_copies3214 a h).2.1
  have c102850 := (publicBatchWrapper86_copies3214 a h).2.2.1
  have c102851 := (publicBatchWrapper86_copies3214 a h).2.2.2.1
  have c102852 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.1
  have c102853 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.1
  have c102854 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_8
  simp only [← c102849, ← c102850, ← c102851] at e_1682_8
  have e_1682_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_9
  simp only [← c102852, ← c102853, k2, ← c102854] at e_1682_9
  have hr := e_1682_9
  simp only [e_1682_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16863)) := by
  have c102855 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.1
  have c102856 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.1
  have c102857 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.1
  have c102858 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102859 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102860 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_10
  simp only [← c102855, ← c102856, ← c102857] at e_1682_10
  have e_1682_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_11
  simp only [← c102858, ← c102859, k2, ← c102860] at e_1682_11
  have hr := e_1682_11
  simp only [e_1682_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16864)) := by
  have c102861 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102862 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102863 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102864 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102865 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102866 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_12
  simp only [← c102861, ← c102862, ← c102863] at e_1682_12
  have e_1682_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_13
  simp only [← c102864, ← c102865, k2, ← c102866] at e_1682_13
  have hr := e_1682_13
  simp only [e_1682_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16865)) := by
  have c102867 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102868 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102869 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102870 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102871 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102872 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_14
  simp only [← c102867, ← c102868, ← c102869] at e_1682_14
  have e_1682_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_15
  simp only [← c102870, ← c102871, k2, ← c102872] at e_1682_15
  have hr := e_1682_15
  simp only [e_1682_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16866)) := by
  have c102873 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102874 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102875 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102876 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102877 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102878 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_16
  simp only [← c102873, ← c102874, ← c102875] at e_1682_16
  have e_1682_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_17
  simp only [← c102876, ← c102877, k2, ← c102878] at e_1682_17
  have hr := e_1682_17
  simp only [e_1682_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16867)) := by
  have c102879 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102880 := (publicBatchWrapper86_copies3215 a h).1
  have c102881 := (publicBatchWrapper86_copies3215 a h).2.1
  have c102882 := (publicBatchWrapper86_copies3215 a h).2.2.1
  have c102883 := (publicBatchWrapper86_copies3215 a h).2.2.2.1
  have c102884 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_18
  simp only [← c102879, ← c102880, ← c102881] at e_1682_18
  have e_1682_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_19
  simp only [← c102882, ← c102883, k2, ← c102884] at e_1682_19
  have hr := e_1682_19
  simp only [e_1682_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16868)) := by
  have c102885 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.1
  have c102886 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.1
  have c102887 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.1
  have c102888 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.1
  have c102889 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.1
  have c102890 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_0
  simp only [← c102885, ← c102886, ← c102887] at e_1683_0
  have e_1683_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_1
  simp only [← c102888, ← c102889, k2, ← c102890] at e_1683_1
  have hr := e_1683_1
  simp only [e_1683_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16869)) := by
  have c102891 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102892 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102893 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102894 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102895 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102896 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_2
  simp only [← c102891, ← c102892, ← c102893] at e_1683_2
  have e_1683_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_3
  simp only [← c102894, ← c102895, k2, ← c102896] at e_1683_3
  have hr := e_1683_3
  simp only [e_1683_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16870)) := by
  have c102897 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102898 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102899 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102900 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102901 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102902 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_4
  simp only [← c102897, ← c102898, ← c102899] at e_1683_4
  have e_1683_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_5
  simp only [← c102900, ← c102901, k2, ← c102902] at e_1683_5
  have hr := e_1683_5
  simp only [e_1683_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16871)) := by
  have c102903 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102904 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102905 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102906 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102907 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102908 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_6
  simp only [← c102903, ← c102904, ← c102905] at e_1683_6
  have e_1683_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_7
  simp only [← c102906, ← c102907, k2, ← c102908] at e_1683_7
  have hr := e_1683_7
  simp only [e_1683_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16872)) := by
  have c102909 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102910 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102911 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102912 := (publicBatchWrapper86_copies3216 a h).1
  have c102913 := (publicBatchWrapper86_copies3216 a h).2.1
  have c102914 := (publicBatchWrapper86_copies3216 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_8
  simp only [← c102909, ← c102910, ← c102911] at e_1683_8
  have e_1683_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_9
  simp only [← c102912, ← c102913, k2, ← c102914] at e_1683_9
  have hr := e_1683_9
  simp only [e_1683_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16873)) := by
  have c102915 := (publicBatchWrapper86_copies3216 a h).2.2.2.1
  have c102916 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.1
  have c102917 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.1
  have c102918 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.1
  have c102919 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.1
  have c102920 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_10
  simp only [← c102915, ← c102916, ← c102917] at e_1683_10
  have e_1683_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_11
  simp only [← c102918, ← c102919, k2, ← c102920] at e_1683_11
  have hr := e_1683_11
  simp only [e_1683_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16874)) := by
  have c102921 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.1
  have c102922 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102923 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102924 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102925 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102926 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_12
  simp only [← c102921, ← c102922, ← c102923] at e_1683_12
  have e_1683_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_13
  simp only [← c102924, ← c102925, k2, ← c102926] at e_1683_13
  have hr := e_1683_13
  simp only [e_1683_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16875)) := by
  have c102927 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102928 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102929 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102930 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102931 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102932 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_14
  simp only [← c102927, ← c102928, ← c102929] at e_1683_14
  have e_1683_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_15
  simp only [← c102930, ← c102931, k2, ← c102932] at e_1683_15
  have hr := e_1683_15
  simp only [e_1683_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16876)) := by
  have c102933 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102934 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102935 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102936 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102937 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102938 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_16
  simp only [← c102933, ← c102934, ← c102935] at e_1683_16
  have e_1683_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_17
  simp only [← c102936, ← c102937, k2, ← c102938] at e_1683_17
  have hr := e_1683_17
  simp only [e_1683_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16877)) := by
  have c102939 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102940 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102941 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102942 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102943 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102944 := (publicBatchWrapper86_copies3217 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_18
  simp only [← c102939, ← c102940, ← c102941] at e_1683_18
  have e_1683_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_19
  simp only [← c102942, ← c102943, k2, ← c102944] at e_1683_19
  have hr := e_1683_19
  simp only [e_1683_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16878)) := by
  have c102945 := (publicBatchWrapper86_copies3217 a h).2.1
  have c102946 := (publicBatchWrapper86_copies3217 a h).2.2.1
  have c102947 := (publicBatchWrapper86_copies3217 a h).2.2.2.1
  have c102948 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.1
  have c102949 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.1
  have c102950 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_0
  simp only [← c102945, ← c102946, ← c102947] at e_1684_0
  have e_1684_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_1
  simp only [← c102948, ← c102949, k2, ← c102950] at e_1684_1
  have hr := e_1684_1
  simp only [e_1684_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16879)) := by
  have c102951 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.1
  have c102952 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.1
  have c102953 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.1
  have c102954 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102955 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102956 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_2
  simp only [← c102951, ← c102952, ← c102953] at e_1684_2
  have e_1684_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_3
  simp only [← c102954, ← c102955, k2, ← c102956] at e_1684_3
  have hr := e_1684_3
  simp only [e_1684_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16880)) := by
  have c102957 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102958 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102959 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102960 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102961 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102962 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_4
  simp only [← c102957, ← c102958, ← c102959] at e_1684_4
  have e_1684_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_5
  simp only [← c102960, ← c102961, k2, ← c102962] at e_1684_5
  have hr := e_1684_5
  simp only [e_1684_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16881)) := by
  have c102963 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102964 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102965 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102966 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102967 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102968 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_6
  simp only [← c102963, ← c102964, ← c102965] at e_1684_6
  have e_1684_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_7
  simp only [← c102966, ← c102967, k2, ← c102968] at e_1684_7
  have hr := e_1684_7
  simp only [e_1684_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16882)) := by
  have c102969 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102970 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102971 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102972 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102973 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102974 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_8
  simp only [← c102969, ← c102970, ← c102971] at e_1684_8
  have e_1684_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_9
  simp only [← c102972, ← c102973, k2, ← c102974] at e_1684_9
  have hr := e_1684_9
  simp only [e_1684_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16883)) := by
  have c102975 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102976 := (publicBatchWrapper86_copies3218 a h).1
  have c102977 := (publicBatchWrapper86_copies3218 a h).2.1
  have c102978 := (publicBatchWrapper86_copies3218 a h).2.2.1
  have c102979 := (publicBatchWrapper86_copies3218 a h).2.2.2.1
  have c102980 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_10
  simp only [← c102975, ← c102976, ← c102977] at e_1684_10
  have e_1684_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_11
  simp only [← c102978, ← c102979, k2, ← c102980] at e_1684_11
  have hr := e_1684_11
  simp only [e_1684_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16884)) := by
  have c102981 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.1
  have c102982 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.1
  have c102983 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.1
  have c102984 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.1
  have c102985 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.1
  have c102986 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_12
  simp only [← c102981, ← c102982, ← c102983] at e_1684_12
  have e_1684_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_13
  simp only [← c102984, ← c102985, k2, ← c102986] at e_1684_13
  have hr := e_1684_13
  simp only [e_1684_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16885)) := by
  have c102987 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102988 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102989 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102990 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102991 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102992 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_14
  simp only [← c102987, ← c102988, ← c102989] at e_1684_14
  have e_1684_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_15
  simp only [← c102990, ← c102991, k2, ← c102992] at e_1684_15
  have hr := e_1684_15
  simp only [e_1684_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16886)) := by
  have c102993 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102994 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102995 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102996 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102997 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102998 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_16
  simp only [← c102993, ← c102994, ← c102995] at e_1684_16
  have e_1684_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_17
  simp only [← c102996, ← c102997, k2, ← c102998] at e_1684_17
  have hr := e_1684_17
  simp only [e_1684_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16887)) := by
  have c102999 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103000 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103001 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103002 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103003 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103004 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_18
  simp only [← c102999, ← c103000, ← c103001] at e_1684_18
  have e_1684_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_19
  simp only [← c103002, ← c103003, k2, ← c103004] at e_1684_19
  have hr := e_1684_19
  simp only [e_1684_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16888)) := by
  have c103005 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103006 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103007 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103008 := (publicBatchWrapper86_copies3219 a h).1
  have c103009 := (publicBatchWrapper86_copies3219 a h).2.1
  have c103010 := (publicBatchWrapper86_copies3219 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_0
  simp only [← c103005, ← c103006, ← c103007] at e_1685_0
  have e_1685_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_1
  simp only [← c103008, ← c103009, k2, ← c103010] at e_1685_1
  have hr := e_1685_1
  simp only [e_1685_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16889)) := by
  have c103011 := (publicBatchWrapper86_copies3219 a h).2.2.2.1
  have c103012 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.1
  have c103013 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.1
  have c103014 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.1
  have c103015 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.1
  have c103016 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_2
  simp only [← c103011, ← c103012, ← c103013] at e_1685_2
  have e_1685_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_3
  simp only [← c103014, ← c103015, k2, ← c103016] at e_1685_3
  have hr := e_1685_3
  simp only [e_1685_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16890)) := by
  have c103017 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.1
  have c103018 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103019 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103020 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103021 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103022 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_4
  simp only [← c103017, ← c103018, ← c103019] at e_1685_4
  have e_1685_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_5
  simp only [← c103020, ← c103021, k2, ← c103022] at e_1685_5
  have hr := e_1685_5
  simp only [e_1685_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16891)) := by
  have c103023 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103024 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103025 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103026 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103027 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103028 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_6
  simp only [← c103023, ← c103024, ← c103025] at e_1685_6
  have e_1685_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_7
  simp only [← c103026, ← c103027, k2, ← c103028] at e_1685_7
  have hr := e_1685_7
  simp only [e_1685_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16892)) := by
  have c103029 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103030 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103031 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103032 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103033 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103034 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_8
  simp only [← c103029, ← c103030, ← c103031] at e_1685_8
  have e_1685_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_9
  simp only [← c103032, ← c103033, k2, ← c103034] at e_1685_9
  have hr := e_1685_9
  simp only [e_1685_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16893)) := by
  have c103035 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103036 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103037 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103038 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103039 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103040 := (publicBatchWrapper86_copies3220 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_10
  simp only [← c103035, ← c103036, ← c103037] at e_1685_10
  have e_1685_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_11
  simp only [← c103038, ← c103039, k2, ← c103040] at e_1685_11
  have hr := e_1685_11
  simp only [e_1685_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16894)) := by
  have c103041 := (publicBatchWrapper86_copies3220 a h).2.1
  have c103042 := (publicBatchWrapper86_copies3220 a h).2.2.1
  have c103043 := (publicBatchWrapper86_copies3220 a h).2.2.2.1
  have c103044 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.1
  have c103045 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.1
  have c103046 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_12
  simp only [← c103041, ← c103042, ← c103043] at e_1685_12
  have e_1685_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_13
  simp only [← c103044, ← c103045, k2, ← c103046] at e_1685_13
  have hr := e_1685_13
  simp only [e_1685_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16895)) := by
  have c103047 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.1
  have c103048 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.1
  have c103049 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.1
  have c103050 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103051 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103052 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_14
  simp only [← c103047, ← c103048, ← c103049] at e_1685_14
  have e_1685_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_15
  simp only [← c103050, ← c103051, k2, ← c103052] at e_1685_15
  have hr := e_1685_15
  simp only [e_1685_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16896)) := by
  have c103053 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103054 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103055 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103056 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103057 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103058 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_16
  simp only [← c103053, ← c103054, ← c103055] at e_1685_16
  have e_1685_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_17
  simp only [← c103056, ← c103057, k2, ← c103058] at e_1685_17
  have hr := e_1685_17
  simp only [e_1685_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16897)) := by
  have c103059 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103060 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103061 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103062 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103063 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103064 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_18
  simp only [← c103059, ← c103060, ← c103061] at e_1685_18
  have e_1685_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_19
  simp only [← c103062, ← c103063, k2, ← c103064] at e_1685_19
  have hr := e_1685_19
  simp only [e_1685_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16898)) := by
  have c103065 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103066 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103067 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103068 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103069 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103070 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_0
  simp only [← c103065, ← c103066, ← c103067] at e_1686_0
  have e_1686_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_1
  simp only [← c103068, ← c103069, k2, ← c103070] at e_1686_1
  have hr := e_1686_1
  simp only [e_1686_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16899)) := by
  have c103071 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103072 := (publicBatchWrapper86_copies3221 a h).1
  have c103073 := (publicBatchWrapper86_copies3221 a h).2.1
  have c103074 := (publicBatchWrapper86_copies3221 a h).2.2.1
  have c103075 := (publicBatchWrapper86_copies3221 a h).2.2.2.1
  have c103076 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_2
  simp only [← c103071, ← c103072, ← c103073] at e_1686_2
  have e_1686_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_3
  simp only [← c103074, ← c103075, k2, ← c103076] at e_1686_3
  have hr := e_1686_3
  simp only [e_1686_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16900)) := by
  have c103077 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.1
  have c103078 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.1
  have c103079 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.1
  have c103080 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.1
  have c103081 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.1
  have c103082 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_4
  simp only [← c103077, ← c103078, ← c103079] at e_1686_4
  have e_1686_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_5
  simp only [← c103080, ← c103081, k2, ← c103082] at e_1686_5
  have hr := e_1686_5
  simp only [e_1686_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16901)) := by
  have c103083 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103084 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103085 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103086 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103087 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103088 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_6
  simp only [← c103083, ← c103084, ← c103085] at e_1686_6
  have e_1686_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_7
  simp only [← c103086, ← c103087, k2, ← c103088] at e_1686_7
  have hr := e_1686_7
  simp only [e_1686_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16902)) := by
  have c103089 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103090 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103091 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103092 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103093 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103094 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_8
  simp only [← c103089, ← c103090, ← c103091] at e_1686_8
  have e_1686_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_9
  simp only [← c103092, ← c103093, k2, ← c103094] at e_1686_9
  have hr := e_1686_9
  simp only [e_1686_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16903)) := by
  have c103095 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103096 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103097 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103098 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103099 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103100 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_10
  simp only [← c103095, ← c103096, ← c103097] at e_1686_10
  have e_1686_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_11
  simp only [← c103098, ← c103099, k2, ← c103100] at e_1686_11
  have hr := e_1686_11
  simp only [e_1686_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16904)) := by
  have c103101 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103102 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103103 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103104 := (publicBatchWrapper86_copies3222 a h).1
  have c103105 := (publicBatchWrapper86_copies3222 a h).2.1
  have c103106 := (publicBatchWrapper86_copies3222 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_12
  simp only [← c103101, ← c103102, ← c103103] at e_1686_12
  have e_1686_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_13
  simp only [← c103104, ← c103105, k2, ← c103106] at e_1686_13
  have hr := e_1686_13
  simp only [e_1686_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16905)) := by
  have c103107 := (publicBatchWrapper86_copies3222 a h).2.2.2.1
  have c103108 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.1
  have c103109 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.1
  have c103110 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.1
  have c103111 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.1
  have c103112 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_14
  simp only [← c103107, ← c103108, ← c103109] at e_1686_14
  have e_1686_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_15
  simp only [← c103110, ← c103111, k2, ← c103112] at e_1686_15
  have hr := e_1686_15
  simp only [e_1686_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16906)) := by
  have c103113 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.1
  have c103114 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103115 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103116 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103117 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103118 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_16
  simp only [← c103113, ← c103114, ← c103115] at e_1686_16
  have e_1686_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_17
  simp only [← c103116, ← c103117, k2, ← c103118] at e_1686_17
  have hr := e_1686_17
  simp only [e_1686_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16907)) := by
  have c103119 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103120 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103121 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103122 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103123 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103124 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_18
  simp only [← c103119, ← c103120, ← c103121] at e_1686_18
  have e_1686_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_19
  simp only [← c103122, ← c103123, k2, ← c103124] at e_1686_19
  have hr := e_1686_19
  simp only [e_1686_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16908)) := by
  have c103125 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103126 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103127 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103128 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103129 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103130 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_0
  simp only [← c103125, ← c103126, ← c103127] at e_1687_0
  have e_1687_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_1
  simp only [← c103128, ← c103129, k2, ← c103130] at e_1687_1
  have hr := e_1687_1
  simp only [e_1687_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16909)) := by
  have c103131 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103132 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103133 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103134 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103135 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103136 := (publicBatchWrapper86_copies3223 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_2
  simp only [← c103131, ← c103132, ← c103133] at e_1687_2
  have e_1687_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_3
  simp only [← c103134, ← c103135, k2, ← c103136] at e_1687_3
  have hr := e_1687_3
  simp only [e_1687_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16910)) := by
  have c103137 := (publicBatchWrapper86_copies3223 a h).2.1
  have c103138 := (publicBatchWrapper86_copies3223 a h).2.2.1
  have c103139 := (publicBatchWrapper86_copies3223 a h).2.2.2.1
  have c103140 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.1
  have c103141 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.1
  have c103142 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_4
  simp only [← c103137, ← c103138, ← c103139] at e_1687_4
  have e_1687_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_5
  simp only [← c103140, ← c103141, k2, ← c103142] at e_1687_5
  have hr := e_1687_5
  simp only [e_1687_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16911)) := by
  have c103143 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.1
  have c103144 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.1
  have c103145 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.1
  have c103146 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103147 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103148 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_6
  simp only [← c103143, ← c103144, ← c103145] at e_1687_6
  have e_1687_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_7
  simp only [← c103146, ← c103147, k2, ← c103148] at e_1687_7
  have hr := e_1687_7
  simp only [e_1687_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16912)) := by
  have c103149 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103150 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103151 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103152 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103153 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103154 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_8
  simp only [← c103149, ← c103150, ← c103151] at e_1687_8
  have e_1687_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_9
  simp only [← c103152, ← c103153, k2, ← c103154] at e_1687_9
  have hr := e_1687_9
  simp only [e_1687_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16913)) := by
  have c103155 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103156 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103157 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103158 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103159 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103160 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_10
  simp only [← c103155, ← c103156, ← c103157] at e_1687_10
  have e_1687_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_11
  simp only [← c103158, ← c103159, k2, ← c103160] at e_1687_11
  have hr := e_1687_11
  simp only [e_1687_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16914)) := by
  have c103161 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103162 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103163 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103164 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103165 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103166 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_12
  simp only [← c103161, ← c103162, ← c103163] at e_1687_12
  have e_1687_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_13
  simp only [← c103164, ← c103165, k2, ← c103166] at e_1687_13
  have hr := e_1687_13
  simp only [e_1687_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16915)) := by
  have c103167 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103168 := (publicBatchWrapper86_copies3224 a h).1
  have c103169 := (publicBatchWrapper86_copies3224 a h).2.1
  have c103170 := (publicBatchWrapper86_copies3224 a h).2.2.1
  have c103171 := (publicBatchWrapper86_copies3224 a h).2.2.2.1
  have c103172 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_14
  simp only [← c103167, ← c103168, ← c103169] at e_1687_14
  have e_1687_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_15
  simp only [← c103170, ← c103171, k2, ← c103172] at e_1687_15
  have hr := e_1687_15
  simp only [e_1687_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16916)) := by
  have c103173 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.1
  have c103174 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.1
  have c103175 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.1
  have c103176 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.1
  have c103177 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.1
  have c103178 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_16
  simp only [← c103173, ← c103174, ← c103175] at e_1687_16
  have e_1687_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_17
  simp only [← c103176, ← c103177, k2, ← c103178] at e_1687_17
  have hr := e_1687_17
  simp only [e_1687_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16917)) := by
  have c103179 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103180 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103181 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103182 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103183 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103184 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_18
  simp only [← c103179, ← c103180, ← c103181] at e_1687_18
  have e_1687_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_19
  simp only [← c103182, ← c103183, k2, ← c103184] at e_1687_19
  have hr := e_1687_19
  simp only [e_1687_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16918)) := by
  have c103185 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103186 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103187 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103188 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103189 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103190 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_0
  simp only [← c103185, ← c103186, ← c103187] at e_1688_0
  have e_1688_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_1
  simp only [← c103188, ← c103189, k2, ← c103190] at e_1688_1
  have hr := e_1688_1
  simp only [e_1688_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16919)) := by
  have c103191 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103192 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103193 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103194 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103195 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103196 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_2
  simp only [← c103191, ← c103192, ← c103193] at e_1688_2
  have e_1688_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_3
  simp only [← c103194, ← c103195, k2, ← c103196] at e_1688_3
  have hr := e_1688_3
  simp only [e_1688_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16920)) := by
  have c103197 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103198 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103199 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103200 := (publicBatchWrapper86_copies3225 a h).1
  have c103201 := (publicBatchWrapper86_copies3225 a h).2.1
  have c103202 := (publicBatchWrapper86_copies3225 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_4
  simp only [← c103197, ← c103198, ← c103199] at e_1688_4
  have e_1688_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_5
  simp only [← c103200, ← c103201, k2, ← c103202] at e_1688_5
  have hr := e_1688_5
  simp only [e_1688_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16921)) := by
  have c103203 := (publicBatchWrapper86_copies3225 a h).2.2.2.1
  have c103204 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.1
  have c103205 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.1
  have c103206 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.1
  have c103207 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.1
  have c103208 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_6
  simp only [← c103203, ← c103204, ← c103205] at e_1688_6
  have e_1688_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_7
  simp only [← c103206, ← c103207, k2, ← c103208] at e_1688_7
  have hr := e_1688_7
  simp only [e_1688_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16922)) := by
  have c103209 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.1
  have c103210 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103211 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103212 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103213 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103214 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_8
  simp only [← c103209, ← c103210, ← c103211] at e_1688_8
  have e_1688_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_9
  simp only [← c103212, ← c103213, k2, ← c103214] at e_1688_9
  have hr := e_1688_9
  simp only [e_1688_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16923)) := by
  have c103215 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103216 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103217 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103218 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103219 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103220 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_10
  simp only [← c103215, ← c103216, ← c103217] at e_1688_10
  have e_1688_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_11
  simp only [← c103218, ← c103219, k2, ← c103220] at e_1688_11
  have hr := e_1688_11
  simp only [e_1688_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16924)) := by
  have c103221 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103222 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103223 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103224 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103225 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103226 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_12
  simp only [← c103221, ← c103222, ← c103223] at e_1688_12
  have e_1688_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_13
  simp only [← c103224, ← c103225, k2, ← c103226] at e_1688_13
  have hr := e_1688_13
  simp only [e_1688_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16925)) := by
  have c103227 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103228 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103229 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103230 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103231 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103232 := (publicBatchWrapper86_copies3226 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_14
  simp only [← c103227, ← c103228, ← c103229] at e_1688_14
  have e_1688_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_15
  simp only [← c103230, ← c103231, k2, ← c103232] at e_1688_15
  have hr := e_1688_15
  simp only [e_1688_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16926)) := by
  have c103233 := (publicBatchWrapper86_copies3226 a h).2.1
  have c103234 := (publicBatchWrapper86_copies3226 a h).2.2.1
  have c103235 := (publicBatchWrapper86_copies3226 a h).2.2.2.1
  have c103236 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.1
  have c103237 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.1
  have c103238 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_16
  simp only [← c103233, ← c103234, ← c103235] at e_1688_16
  have e_1688_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_17
  simp only [← c103236, ← c103237, k2, ← c103238] at e_1688_17
  have hr := e_1688_17
  simp only [e_1688_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16927)) := by
  have c103239 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.1
  have c103240 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.1
  have c103241 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.1
  have c103242 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103243 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103244 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_18
  simp only [← c103239, ← c103240, ← c103241] at e_1688_18
  have e_1688_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_19
  simp only [← c103242, ← c103243, k2, ← c103244] at e_1688_19
  have hr := e_1688_19
  simp only [e_1688_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16928)) := by
  have c103245 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103246 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103247 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103248 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103249 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103250 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_0
  simp only [← c103245, ← c103246, ← c103247] at e_1689_0
  have e_1689_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_1
  simp only [← c103248, ← c103249, k2, ← c103250] at e_1689_1
  have hr := e_1689_1
  simp only [e_1689_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16929)) := by
  have c103251 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103252 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103253 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103254 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103255 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103256 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_2
  simp only [← c103251, ← c103252, ← c103253] at e_1689_2
  have e_1689_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_3
  simp only [← c103254, ← c103255, k2, ← c103256] at e_1689_3
  have hr := e_1689_3
  simp only [e_1689_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16930)) := by
  have c103257 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103258 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103259 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103260 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103261 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103262 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_4
  simp only [← c103257, ← c103258, ← c103259] at e_1689_4
  have e_1689_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_5
  simp only [← c103260, ← c103261, k2, ← c103262] at e_1689_5
  have hr := e_1689_5
  simp only [e_1689_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16931)) := by
  have c103263 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103264 := (publicBatchWrapper86_copies3227 a h).1
  have c103265 := (publicBatchWrapper86_copies3227 a h).2.1
  have c103266 := (publicBatchWrapper86_copies3227 a h).2.2.1
  have c103267 := (publicBatchWrapper86_copies3227 a h).2.2.2.1
  have c103268 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_6
  simp only [← c103263, ← c103264, ← c103265] at e_1689_6
  have e_1689_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_7
  simp only [← c103266, ← c103267, k2, ← c103268] at e_1689_7
  have hr := e_1689_7
  simp only [e_1689_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16932)) := by
  have c103269 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.1
  have c103270 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.1
  have c103271 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.1
  have c103272 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.1
  have c103273 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.1
  have c103274 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_8
  simp only [← c103269, ← c103270, ← c103271] at e_1689_8
  have e_1689_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_9
  simp only [← c103272, ← c103273, k2, ← c103274] at e_1689_9
  have hr := e_1689_9
  simp only [e_1689_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16933)) := by
  have c103275 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103276 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103277 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103278 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103279 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103280 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_10
  simp only [← c103275, ← c103276, ← c103277] at e_1689_10
  have e_1689_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_11
  simp only [← c103278, ← c103279, k2, ← c103280] at e_1689_11
  have hr := e_1689_11
  simp only [e_1689_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16934)) := by
  have c103281 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103282 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103283 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103284 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103285 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103286 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_12
  simp only [← c103281, ← c103282, ← c103283] at e_1689_12
  have e_1689_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_13
  simp only [← c103284, ← c103285, k2, ← c103286] at e_1689_13
  have hr := e_1689_13
  simp only [e_1689_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16935)) := by
  have c103287 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103288 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103289 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103290 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103291 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103292 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_14
  simp only [← c103287, ← c103288, ← c103289] at e_1689_14
  have e_1689_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_15
  simp only [← c103290, ← c103291, k2, ← c103292] at e_1689_15
  have hr := e_1689_15
  simp only [e_1689_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16936)) := by
  have c103293 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103294 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103295 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103296 := (publicBatchWrapper86_copies3228 a h).1
  have c103297 := (publicBatchWrapper86_copies3228 a h).2.1
  have c103298 := (publicBatchWrapper86_copies3228 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_16
  simp only [← c103293, ← c103294, ← c103295] at e_1689_16
  have e_1689_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_17
  simp only [← c103296, ← c103297, k2, ← c103298] at e_1689_17
  have hr := e_1689_17
  simp only [e_1689_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16937)) := by
  have c103299 := (publicBatchWrapper86_copies3228 a h).2.2.2.1
  have c103300 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.1
  have c103301 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.1
  have c103302 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.1
  have c103303 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.1
  have c103304 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_18
  simp only [← c103299, ← c103300, ← c103301] at e_1689_18
  have e_1689_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_19
  simp only [← c103302, ← c103303, k2, ← c103304] at e_1689_19
  have hr := e_1689_19
  simp only [e_1689_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16938)) := by
  have c103305 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.1
  have c103306 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103307 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103308 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103309 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103310 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_0
  simp only [← c103305, ← c103306, ← c103307] at e_1690_0
  have e_1690_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_1
  simp only [← c103308, ← c103309, k2, ← c103310] at e_1690_1
  have hr := e_1690_1
  simp only [e_1690_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16939)) := by
  have c103311 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103312 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103313 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103314 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103315 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103316 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_2
  simp only [← c103311, ← c103312, ← c103313] at e_1690_2
  have e_1690_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_3
  simp only [← c103314, ← c103315, k2, ← c103316] at e_1690_3
  have hr := e_1690_3
  simp only [e_1690_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16940)) := by
  have c103317 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103318 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103319 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103320 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103321 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103322 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_4
  simp only [← c103317, ← c103318, ← c103319] at e_1690_4
  have e_1690_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_5
  simp only [← c103320, ← c103321, k2, ← c103322] at e_1690_5
  have hr := e_1690_5
  simp only [e_1690_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16941)) := by
  have c103323 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103324 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103325 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103326 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103327 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103328 := (publicBatchWrapper86_copies3229 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_6
  simp only [← c103323, ← c103324, ← c103325] at e_1690_6
  have e_1690_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_7
  simp only [← c103326, ← c103327, k2, ← c103328] at e_1690_7
  have hr := e_1690_7
  simp only [e_1690_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16942)) := by
  have c103329 := (publicBatchWrapper86_copies3229 a h).2.1
  have c103330 := (publicBatchWrapper86_copies3229 a h).2.2.1
  have c103331 := (publicBatchWrapper86_copies3229 a h).2.2.2.1
  have c103332 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.1
  have c103333 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.1
  have c103334 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_8
  simp only [← c103329, ← c103330, ← c103331] at e_1690_8
  have e_1690_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_9
  simp only [← c103332, ← c103333, k2, ← c103334] at e_1690_9
  have hr := e_1690_9
  simp only [e_1690_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16943)) := by
  have c103335 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.1
  have c103336 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.1
  have c103337 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.1
  have c103338 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103339 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103340 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_10
  simp only [← c103335, ← c103336, ← c103337] at e_1690_10
  have e_1690_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_11
  simp only [← c103338, ← c103339, k2, ← c103340] at e_1690_11
  have hr := e_1690_11
  simp only [e_1690_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16944)) := by
  have c103341 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103342 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103343 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103344 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103345 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103346 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_12
  simp only [← c103341, ← c103342, ← c103343] at e_1690_12
  have e_1690_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_13
  simp only [← c103344, ← c103345, k2, ← c103346] at e_1690_13
  have hr := e_1690_13
  simp only [e_1690_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16945)) := by
  have c103347 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103348 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103349 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103350 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103351 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103352 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_14
  simp only [← c103347, ← c103348, ← c103349] at e_1690_14
  have e_1690_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_15
  simp only [← c103350, ← c103351, k2, ← c103352] at e_1690_15
  have hr := e_1690_15
  simp only [e_1690_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16946)) := by
  have c103353 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103354 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103355 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103356 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103357 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103358 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_16
  simp only [← c103353, ← c103354, ← c103355] at e_1690_16
  have e_1690_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_17
  simp only [← c103356, ← c103357, k2, ← c103358] at e_1690_17
  have hr := e_1690_17
  simp only [e_1690_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16947)) := by
  have c103359 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103360 := (publicBatchWrapper86_copies3230 a h).1
  have c103361 := (publicBatchWrapper86_copies3230 a h).2.1
  have c103362 := (publicBatchWrapper86_copies3230 a h).2.2.1
  have c103363 := (publicBatchWrapper86_copies3230 a h).2.2.2.1
  have c103364 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_18
  simp only [← c103359, ← c103360, ← c103361] at e_1690_18
  have e_1690_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_19
  simp only [← c103362, ← c103363, k2, ← c103364] at e_1690_19
  have hr := e_1690_19
  simp only [e_1690_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16948)) := by
  have c103365 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.1
  have c103366 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.1
  have c103367 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.1
  have c103368 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.1
  have c103369 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.1
  have c103370 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_0
  simp only [← c103365, ← c103366, ← c103367] at e_1691_0
  have e_1691_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_1
  simp only [← c103368, ← c103369, k2, ← c103370] at e_1691_1
  have hr := e_1691_1
  simp only [e_1691_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16949)) := by
  have c103371 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103372 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103373 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103374 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103375 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103376 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_2
  simp only [← c103371, ← c103372, ← c103373] at e_1691_2
  have e_1691_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_3
  simp only [← c103374, ← c103375, k2, ← c103376] at e_1691_3
  have hr := e_1691_3
  simp only [e_1691_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16950)) := by
  have c103377 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103378 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103379 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103380 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103381 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103382 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_4
  simp only [← c103377, ← c103378, ← c103379] at e_1691_4
  have e_1691_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_5
  simp only [← c103380, ← c103381, k2, ← c103382] at e_1691_5
  have hr := e_1691_5
  simp only [e_1691_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16951)) := by
  have c103383 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103384 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103385 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103386 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103387 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103388 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_6
  simp only [← c103383, ← c103384, ← c103385] at e_1691_6
  have e_1691_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_7
  simp only [← c103386, ← c103387, k2, ← c103388] at e_1691_7
  have hr := e_1691_7
  simp only [e_1691_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16952)) := by
  have c103389 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103390 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103391 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103392 := (publicBatchWrapper86_copies3231 a h).1
  have c103393 := (publicBatchWrapper86_copies3231 a h).2.1
  have c103394 := (publicBatchWrapper86_copies3231 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_8
  simp only [← c103389, ← c103390, ← c103391] at e_1691_8
  have e_1691_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_9
  simp only [← c103392, ← c103393, k2, ← c103394] at e_1691_9
  have hr := e_1691_9
  simp only [e_1691_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16953)) := by
  have c103395 := (publicBatchWrapper86_copies3231 a h).2.2.2.1
  have c103396 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.1
  have c103397 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.1
  have c103398 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.1
  have c103399 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.1
  have c103400 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_10
  simp only [← c103395, ← c103396, ← c103397] at e_1691_10
  have e_1691_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_11
  simp only [← c103398, ← c103399, k2, ← c103400] at e_1691_11
  have hr := e_1691_11
  simp only [e_1691_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16954)) := by
  have c103401 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.1
  have c103402 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103403 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103404 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103405 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103406 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_12
  simp only [← c103401, ← c103402, ← c103403] at e_1691_12
  have e_1691_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_13
  simp only [← c103404, ← c103405, k2, ← c103406] at e_1691_13
  have hr := e_1691_13
  simp only [e_1691_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16955)) := by
  have c103407 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103408 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103409 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103410 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103411 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103412 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_14
  simp only [← c103407, ← c103408, ← c103409] at e_1691_14
  have e_1691_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_15
  simp only [← c103410, ← c103411, k2, ← c103412] at e_1691_15
  have hr := e_1691_15
  simp only [e_1691_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16956)) := by
  have c103413 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103414 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103415 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103416 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103417 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103418 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_16
  simp only [← c103413, ← c103414, ← c103415] at e_1691_16
  have e_1691_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_17
  simp only [← c103416, ← c103417, k2, ← c103418] at e_1691_17
  have hr := e_1691_17
  simp only [e_1691_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16957)) := by
  have c103419 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103420 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103421 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103422 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103423 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103424 := (publicBatchWrapper86_copies3232 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_18
  simp only [← c103419, ← c103420, ← c103421] at e_1691_18
  have e_1691_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_19
  simp only [← c103422, ← c103423, k2, ← c103424] at e_1691_19
  have hr := e_1691_19
  simp only [e_1691_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16958)) := by
  have c103425 := (publicBatchWrapper86_copies3232 a h).2.1
  have c103426 := (publicBatchWrapper86_copies3232 a h).2.2.1
  have c103427 := (publicBatchWrapper86_copies3232 a h).2.2.2.1
  have c103428 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.1
  have c103429 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.1
  have c103430 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_0
  simp only [← c103425, ← c103426, ← c103427] at e_1692_0
  have e_1692_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_1
  simp only [← c103428, ← c103429, k2, ← c103430] at e_1692_1
  have hr := e_1692_1
  simp only [e_1692_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16959)) := by
  have c103431 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.1
  have c103432 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.1
  have c103433 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.1
  have c103434 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103435 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103436 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_2
  simp only [← c103431, ← c103432, ← c103433] at e_1692_2
  have e_1692_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_3
  simp only [← c103434, ← c103435, k2, ← c103436] at e_1692_3
  have hr := e_1692_3
  simp only [e_1692_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16960)) := by
  have c103437 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103438 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103439 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103440 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103441 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103442 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_4
  simp only [← c103437, ← c103438, ← c103439] at e_1692_4
  have e_1692_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_5
  simp only [← c103440, ← c103441, k2, ← c103442] at e_1692_5
  have hr := e_1692_5
  simp only [e_1692_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16961)) := by
  have c103443 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103444 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103445 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103446 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103447 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103448 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_6
  simp only [← c103443, ← c103444, ← c103445] at e_1692_6
  have e_1692_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_7
  simp only [← c103446, ← c103447, k2, ← c103448] at e_1692_7
  have hr := e_1692_7
  simp only [e_1692_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16962)) := by
  have c103449 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103450 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103451 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103452 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103453 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103454 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_8
  simp only [← c103449, ← c103450, ← c103451] at e_1692_8
  have e_1692_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_9
  simp only [← c103452, ← c103453, k2, ← c103454] at e_1692_9
  have hr := e_1692_9
  simp only [e_1692_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16963)) := by
  have c103455 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103456 := (publicBatchWrapper86_copies3233 a h).1
  have c103457 := (publicBatchWrapper86_copies3233 a h).2.1
  have c103458 := (publicBatchWrapper86_copies3233 a h).2.2.1
  have c103459 := (publicBatchWrapper86_copies3233 a h).2.2.2.1
  have c103460 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_10
  simp only [← c103455, ← c103456, ← c103457] at e_1692_10
  have e_1692_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_11
  simp only [← c103458, ← c103459, k2, ← c103460] at e_1692_11
  have hr := e_1692_11
  simp only [e_1692_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16964)) := by
  have c103461 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.1
  have c103462 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.1
  have c103463 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.1
  have c103464 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.1
  have c103465 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.1
  have c103466 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_12
  simp only [← c103461, ← c103462, ← c103463] at e_1692_12
  have e_1692_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_13
  simp only [← c103464, ← c103465, k2, ← c103466] at e_1692_13
  have hr := e_1692_13
  simp only [e_1692_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16965)) := by
  have c103467 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103468 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103469 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103470 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103471 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103472 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_14
  simp only [← c103467, ← c103468, ← c103469] at e_1692_14
  have e_1692_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_15
  simp only [← c103470, ← c103471, k2, ← c103472] at e_1692_15
  have hr := e_1692_15
  simp only [e_1692_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16966)) := by
  have c103473 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103474 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103475 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103476 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103477 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103478 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_16
  simp only [← c103473, ← c103474, ← c103475] at e_1692_16
  have e_1692_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_17
  simp only [← c103476, ← c103477, k2, ← c103478] at e_1692_17
  have hr := e_1692_17
  simp only [e_1692_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16967)) := by
  have c103479 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103480 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103481 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103482 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103483 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103484 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_18
  simp only [← c103479, ← c103480, ← c103481] at e_1692_18
  have e_1692_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_19
  simp only [← c103482, ← c103483, k2, ← c103484] at e_1692_19
  have hr := e_1692_19
  simp only [e_1692_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16968)) := by
  have c103485 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103486 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103487 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103488 := (publicBatchWrapper86_copies3234 a h).1
  have c103489 := (publicBatchWrapper86_copies3234 a h).2.1
  have c103490 := (publicBatchWrapper86_copies3234 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_0
  simp only [← c103485, ← c103486, ← c103487] at e_1693_0
  have e_1693_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_1
  simp only [← c103488, ← c103489, k2, ← c103490] at e_1693_1
  have hr := e_1693_1
  simp only [e_1693_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16969)) := by
  have c103491 := (publicBatchWrapper86_copies3234 a h).2.2.2.1
  have c103492 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.1
  have c103493 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.1
  have c103494 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.1
  have c103495 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.1
  have c103496 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_2
  simp only [← c103491, ← c103492, ← c103493] at e_1693_2
  have e_1693_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_3
  simp only [← c103494, ← c103495, k2, ← c103496] at e_1693_3
  have hr := e_1693_3
  simp only [e_1693_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17010)) := by
  have c103497 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.1
  have c103498 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103499 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103500 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103501 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103502 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_4
  simp only [← c103497, ← c103498, ← c103499] at e_1693_4
  have e_1693_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_5
  simp only [← c103500, ← c103501, k2, ← c103502] at e_1693_5
  have hr := e_1693_5
  simp only [e_1693_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17011)) := by
  have c103503 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103504 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103505 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103506 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103507 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103508 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_6
  simp only [← c103503, ← c103504, ← c103505] at e_1693_6
  have e_1693_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_7
  simp only [← c103506, ← c103507, k2, ← c103508] at e_1693_7
  have hr := e_1693_7
  simp only [e_1693_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17012)) := by
  have c103509 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103510 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103511 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103512 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103513 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103514 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_8
  simp only [← c103509, ← c103510, ← c103511] at e_1693_8
  have e_1693_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_9
  simp only [← c103512, ← c103513, k2, ← c103514] at e_1693_9
  have hr := e_1693_9
  simp only [e_1693_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17013)) := by
  have c103515 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103516 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103517 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103518 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103519 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103520 := (publicBatchWrapper86_copies3235 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_10
  simp only [← c103515, ← c103516, ← c103517] at e_1693_10
  have e_1693_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_11
  simp only [← c103518, ← c103519, k2, ← c103520] at e_1693_11
  have hr := e_1693_11
  simp only [e_1693_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17014)) := by
  have c103521 := (publicBatchWrapper86_copies3235 a h).2.1
  have c103522 := (publicBatchWrapper86_copies3235 a h).2.2.1
  have c103523 := (publicBatchWrapper86_copies3235 a h).2.2.2.1
  have c103524 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.1
  have c103525 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.1
  have c103526 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_12
  simp only [← c103521, ← c103522, ← c103523] at e_1693_12
  have e_1693_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_13
  simp only [← c103524, ← c103525, k2, ← c103526] at e_1693_13
  have hr := e_1693_13
  simp only [e_1693_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17015)) := by
  have c103527 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.1
  have c103528 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.1
  have c103529 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.1
  have c103530 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103531 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103532 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_14
  simp only [← c103527, ← c103528, ← c103529] at e_1693_14
  have e_1693_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_15
  simp only [← c103530, ← c103531, k2, ← c103532] at e_1693_15
  have hr := e_1693_15
  simp only [e_1693_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17016)) := by
  have c103533 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103534 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103535 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103536 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103537 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103538 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_16
  simp only [← c103533, ← c103534, ← c103535] at e_1693_16
  have e_1693_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_17
  simp only [← c103536, ← c103537, k2, ← c103538] at e_1693_17
  have hr := e_1693_17
  simp only [e_1693_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17017)) := by
  have c103539 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103540 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103541 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103542 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103543 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103544 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_18
  simp only [← c103539, ← c103540, ← c103541] at e_1693_18
  have e_1693_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_19
  simp only [← c103542, ← c103543, k2, ← c103544] at e_1693_19
  have hr := e_1693_19
  simp only [e_1693_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17018)) := by
  have c103545 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103546 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103547 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103548 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103549 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103550 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_0
  simp only [← c103545, ← c103546, ← c103547] at e_1694_0
  have e_1694_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_1
  simp only [← c103548, ← c103549, k2, ← c103550] at e_1694_1
  have hr := e_1694_1
  simp only [e_1694_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17019)) := by
  have c103551 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103552 := (publicBatchWrapper86_copies3236 a h).1
  have c103553 := (publicBatchWrapper86_copies3236 a h).2.1
  have c103554 := (publicBatchWrapper86_copies3236 a h).2.2.1
  have c103555 := (publicBatchWrapper86_copies3236 a h).2.2.2.1
  have c103556 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_2
  simp only [← c103551, ← c103552, ← c103553] at e_1694_2
  have e_1694_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_3
  simp only [← c103554, ← c103555, k2, ← c103556] at e_1694_3
  have hr := e_1694_3
  simp only [e_1694_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17020)) := by
  have c103557 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.1
  have c103558 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.1
  have c103559 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.1
  have c103560 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.1
  have c103561 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.1
  have c103562 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_4
  simp only [← c103557, ← c103558, ← c103559] at e_1694_4
  have e_1694_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_5
  simp only [← c103560, ← c103561, k2, ← c103562] at e_1694_5
  have hr := e_1694_5
  simp only [e_1694_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17021)) := by
  have c103563 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103564 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103565 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103566 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103567 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103568 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_6
  simp only [← c103563, ← c103564, ← c103565] at e_1694_6
  have e_1694_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_7
  simp only [← c103566, ← c103567, k2, ← c103568] at e_1694_7
  have hr := e_1694_7
  simp only [e_1694_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17022)) := by
  have c103569 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103570 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103571 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103572 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103573 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103574 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_8
  simp only [← c103569, ← c103570, ← c103571] at e_1694_8
  have e_1694_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_9
  simp only [← c103572, ← c103573, k2, ← c103574] at e_1694_9
  have hr := e_1694_9
  simp only [e_1694_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17023)) := by
  have c103575 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103576 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103577 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103578 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103579 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103580 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_10
  simp only [← c103575, ← c103576, ← c103577] at e_1694_10
  have e_1694_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_11
  simp only [← c103578, ← c103579, k2, ← c103580] at e_1694_11
  have hr := e_1694_11
  simp only [e_1694_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17024)) := by
  have c103581 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103582 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103583 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103584 := (publicBatchWrapper86_copies3237 a h).1
  have c103585 := (publicBatchWrapper86_copies3237 a h).2.1
  have c103586 := (publicBatchWrapper86_copies3237 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_12
  simp only [← c103581, ← c103582, ← c103583] at e_1694_12
  have e_1694_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_13
  simp only [← c103584, ← c103585, k2, ← c103586] at e_1694_13
  have hr := e_1694_13
  simp only [e_1694_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17025)) := by
  have c103587 := (publicBatchWrapper86_copies3237 a h).2.2.2.1
  have c103588 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.1
  have c103589 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.1
  have c103590 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.1
  have c103591 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.1
  have c103592 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_14
  simp only [← c103587, ← c103588, ← c103589] at e_1694_14
  have e_1694_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_15
  simp only [← c103590, ← c103591, k2, ← c103592] at e_1694_15
  have hr := e_1694_15
  simp only [e_1694_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17026)) := by
  have c103593 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.1
  have c103594 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103595 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103596 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103597 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103598 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_16
  simp only [← c103593, ← c103594, ← c103595] at e_1694_16
  have e_1694_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_17
  simp only [← c103596, ← c103597, k2, ← c103598] at e_1694_17
  have hr := e_1694_17
  simp only [e_1694_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17027)) := by
  have c103599 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103600 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103601 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103602 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103603 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103604 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_18
  simp only [← c103599, ← c103600, ← c103601] at e_1694_18
  have e_1694_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_19
  simp only [← c103602, ← c103603, k2, ← c103604] at e_1694_19
  have hr := e_1694_19
  simp only [e_1694_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17028)) := by
  have c103605 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103606 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103607 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103608 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103609 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103610 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_0
  simp only [← c103605, ← c103606, ← c103607] at e_1695_0
  have e_1695_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_1
  simp only [← c103608, ← c103609, k2, ← c103610] at e_1695_1
  have hr := e_1695_1
  simp only [e_1695_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17029)) := by
  have c103611 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103612 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103613 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103614 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103615 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103616 := (publicBatchWrapper86_copies3238 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_2
  simp only [← c103611, ← c103612, ← c103613] at e_1695_2
  have e_1695_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_3
  simp only [← c103614, ← c103615, k2, ← c103616] at e_1695_3
  have hr := e_1695_3
  simp only [e_1695_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17030)) := by
  have c103617 := (publicBatchWrapper86_copies3238 a h).2.1
  have c103618 := (publicBatchWrapper86_copies3238 a h).2.2.1
  have c103619 := (publicBatchWrapper86_copies3238 a h).2.2.2.1
  have c103620 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.1
  have c103621 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.1
  have c103622 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_4
  simp only [← c103617, ← c103618, ← c103619] at e_1695_4
  have e_1695_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_5
  simp only [← c103620, ← c103621, k2, ← c103622] at e_1695_5
  have hr := e_1695_5
  simp only [e_1695_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17031)) := by
  have c103623 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.1
  have c103624 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.1
  have c103625 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.1
  have c103626 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103627 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103628 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_6
  simp only [← c103623, ← c103624, ← c103625] at e_1695_6
  have e_1695_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_7
  simp only [← c103626, ← c103627, k2, ← c103628] at e_1695_7
  have hr := e_1695_7
  simp only [e_1695_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17032)) := by
  have c103629 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103630 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103631 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103632 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103633 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103634 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_8
  simp only [← c103629, ← c103630, ← c103631] at e_1695_8
  have e_1695_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_9
  simp only [← c103632, ← c103633, k2, ← c103634] at e_1695_9
  have hr := e_1695_9
  simp only [e_1695_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17033)) := by
  have c103635 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103636 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103637 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103638 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103639 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103640 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_10
  simp only [← c103635, ← c103636, ← c103637] at e_1695_10
  have e_1695_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_11
  simp only [← c103638, ← c103639, k2, ← c103640] at e_1695_11
  have hr := e_1695_11
  simp only [e_1695_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17034)) := by
  have c103641 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103642 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103643 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103644 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103645 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103646 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_12
  simp only [← c103641, ← c103642, ← c103643] at e_1695_12
  have e_1695_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_13
  simp only [← c103644, ← c103645, k2, ← c103646] at e_1695_13
  have hr := e_1695_13
  simp only [e_1695_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17035)) := by
  have c103647 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103648 := (publicBatchWrapper86_copies3239 a h).1
  have c103649 := (publicBatchWrapper86_copies3239 a h).2.1
  have c103650 := (publicBatchWrapper86_copies3239 a h).2.2.1
  have c103651 := (publicBatchWrapper86_copies3239 a h).2.2.2.1
  have c103652 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_14
  simp only [← c103647, ← c103648, ← c103649] at e_1695_14
  have e_1695_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_15
  simp only [← c103650, ← c103651, k2, ← c103652] at e_1695_15
  have hr := e_1695_15
  simp only [e_1695_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17036)) := by
  have c103653 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.1
  have c103654 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.1
  have c103655 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.1
  have c103656 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.1
  have c103657 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.1
  have c103658 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_16
  simp only [← c103653, ← c103654, ← c103655] at e_1695_16
  have e_1695_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_17
  simp only [← c103656, ← c103657, k2, ← c103658] at e_1695_17
  have hr := e_1695_17
  simp only [e_1695_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17037)) := by
  have c103659 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103660 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103661 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103662 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103663 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103664 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_18
  simp only [← c103659, ← c103660, ← c103661] at e_1695_18
  have e_1695_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_19
  simp only [← c103662, ← c103663, k2, ← c103664] at e_1695_19
  have hr := e_1695_19
  simp only [e_1695_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17038)) := by
  have c103665 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103666 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103667 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103668 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103669 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103670 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_0
  simp only [← c103665, ← c103666, ← c103667] at e_1696_0
  have e_1696_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_1
  simp only [← c103668, ← c103669, k2, ← c103670] at e_1696_1
  have hr := e_1696_1
  simp only [e_1696_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17039)) := by
  have c103671 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103672 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103673 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103674 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103675 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103676 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_2
  simp only [← c103671, ← c103672, ← c103673] at e_1696_2
  have e_1696_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_3
  simp only [← c103674, ← c103675, k2, ← c103676] at e_1696_3
  have hr := e_1696_3
  simp only [e_1696_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17040)) := by
  have c103677 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103678 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103679 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103680 := (publicBatchWrapper86_copies3240 a h).1
  have c103681 := (publicBatchWrapper86_copies3240 a h).2.1
  have c103682 := (publicBatchWrapper86_copies3240 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_4
  simp only [← c103677, ← c103678, ← c103679] at e_1696_4
  have e_1696_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_5
  simp only [← c103680, ← c103681, k2, ← c103682] at e_1696_5
  have hr := e_1696_5
  simp only [e_1696_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17041)) := by
  have c103683 := (publicBatchWrapper86_copies3240 a h).2.2.2.1
  have c103684 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.1
  have c103685 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.1
  have c103686 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.1
  have c103687 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.1
  have c103688 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_6
  simp only [← c103683, ← c103684, ← c103685] at e_1696_6
  have e_1696_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_7
  simp only [← c103686, ← c103687, k2, ← c103688] at e_1696_7
  have hr := e_1696_7
  simp only [e_1696_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17042)) := by
  have c103689 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.1
  have c103690 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103691 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103692 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103693 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103694 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_8
  simp only [← c103689, ← c103690, ← c103691] at e_1696_8
  have e_1696_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_9
  simp only [← c103692, ← c103693, k2, ← c103694] at e_1696_9
  have hr := e_1696_9
  simp only [e_1696_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17043)) := by
  have c103695 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103696 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103697 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103698 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103699 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103700 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_10
  simp only [← c103695, ← c103696, ← c103697] at e_1696_10
  have e_1696_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_11
  simp only [← c103698, ← c103699, k2, ← c103700] at e_1696_11
  have hr := e_1696_11
  simp only [e_1696_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17044)) := by
  have c103701 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103702 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103703 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103704 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103705 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103706 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_12
  simp only [← c103701, ← c103702, ← c103703] at e_1696_12
  have e_1696_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_13
  simp only [← c103704, ← c103705, k2, ← c103706] at e_1696_13
  have hr := e_1696_13
  simp only [e_1696_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17045)) := by
  have c103707 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103708 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103709 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103710 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103711 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103712 := (publicBatchWrapper86_copies3241 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_14
  simp only [← c103707, ← c103708, ← c103709] at e_1696_14
  have e_1696_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_15
  simp only [← c103710, ← c103711, k2, ← c103712] at e_1696_15
  have hr := e_1696_15
  simp only [e_1696_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17046)) := by
  have c103713 := (publicBatchWrapper86_copies3241 a h).2.1
  have c103714 := (publicBatchWrapper86_copies3241 a h).2.2.1
  have c103715 := (publicBatchWrapper86_copies3241 a h).2.2.2.1
  have c103716 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.1
  have c103717 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.1
  have c103718 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_16
  simp only [← c103713, ← c103714, ← c103715] at e_1696_16
  have e_1696_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_17
  simp only [← c103716, ← c103717, k2, ← c103718] at e_1696_17
  have hr := e_1696_17
  simp only [e_1696_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17047)) := by
  have c103719 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.1
  have c103720 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.1
  have c103721 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.1
  have c103722 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103723 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103724 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_18
  simp only [← c103719, ← c103720, ← c103721] at e_1696_18
  have e_1696_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_19
  simp only [← c103722, ← c103723, k2, ← c103724] at e_1696_19
  have hr := e_1696_19
  simp only [e_1696_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17048)) := by
  have c103725 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103726 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103727 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103728 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103729 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103730 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_0
  simp only [← c103725, ← c103726, ← c103727] at e_1697_0
  have e_1697_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_1
  simp only [← c103728, ← c103729, k2, ← c103730] at e_1697_1
  have hr := e_1697_1
  simp only [e_1697_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17049)) := by
  have c103731 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103732 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103733 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103734 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103735 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103736 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_2
  simp only [← c103731, ← c103732, ← c103733] at e_1697_2
  have e_1697_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_3
  simp only [← c103734, ← c103735, k2, ← c103736] at e_1697_3
  have hr := e_1697_3
  simp only [e_1697_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17050)) := by
  have c103737 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103738 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103739 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103740 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103741 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103742 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_4
  simp only [← c103737, ← c103738, ← c103739] at e_1697_4
  have e_1697_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_5
  simp only [← c103740, ← c103741, k2, ← c103742] at e_1697_5
  have hr := e_1697_5
  simp only [e_1697_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17051)) := by
  have c103743 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103744 := (publicBatchWrapper86_copies3242 a h).1
  have c103745 := (publicBatchWrapper86_copies3242 a h).2.1
  have c103746 := (publicBatchWrapper86_copies3242 a h).2.2.1
  have c103747 := (publicBatchWrapper86_copies3242 a h).2.2.2.1
  have c103748 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_6
  simp only [← c103743, ← c103744, ← c103745] at e_1697_6
  have e_1697_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_7
  simp only [← c103746, ← c103747, k2, ← c103748] at e_1697_7
  have hr := e_1697_7
  simp only [e_1697_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17052)) := by
  have c103749 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.1
  have c103750 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.1
  have c103751 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.1
  have c103752 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.1
  have c103753 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.1
  have c103754 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_8
  simp only [← c103749, ← c103750, ← c103751] at e_1697_8
  have e_1697_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_9
  simp only [← c103752, ← c103753, k2, ← c103754] at e_1697_9
  have hr := e_1697_9
  simp only [e_1697_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17053)) := by
  have c103755 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103756 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103757 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103758 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103759 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103760 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_10
  simp only [← c103755, ← c103756, ← c103757] at e_1697_10
  have e_1697_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_11
  simp only [← c103758, ← c103759, k2, ← c103760] at e_1697_11
  have hr := e_1697_11
  simp only [e_1697_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17054)) := by
  have c103761 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103762 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103763 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103764 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103765 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103766 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_12
  simp only [← c103761, ← c103762, ← c103763] at e_1697_12
  have e_1697_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_13
  simp only [← c103764, ← c103765, k2, ← c103766] at e_1697_13
  have hr := e_1697_13
  simp only [e_1697_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17055)) := by
  have c103767 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103768 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103769 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103770 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103771 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103772 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_14
  simp only [← c103767, ← c103768, ← c103769] at e_1697_14
  have e_1697_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_15
  simp only [← c103770, ← c103771, k2, ← c103772] at e_1697_15
  have hr := e_1697_15
  simp only [e_1697_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17056)) := by
  have c103773 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103774 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103775 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103776 := (publicBatchWrapper86_copies3243 a h).1
  have c103777 := (publicBatchWrapper86_copies3243 a h).2.1
  have c103778 := (publicBatchWrapper86_copies3243 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_16
  simp only [← c103773, ← c103774, ← c103775] at e_1697_16
  have e_1697_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_17
  simp only [← c103776, ← c103777, k2, ← c103778] at e_1697_17
  have hr := e_1697_17
  simp only [e_1697_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17057)) := by
  have c103779 := (publicBatchWrapper86_copies3243 a h).2.2.2.1
  have c103780 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.1
  have c103781 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.1
  have c103782 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.1
  have c103783 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.1
  have c103784 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_18
  simp only [← c103779, ← c103780, ← c103781] at e_1697_18
  have e_1697_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_19
  simp only [← c103782, ← c103783, k2, ← c103784] at e_1697_19
  have hr := e_1697_19
  simp only [e_1697_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17058)) := by
  have c103785 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.1
  have c103786 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103787 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103788 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103789 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103790 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_0
  simp only [← c103785, ← c103786, ← c103787] at e_1698_0
  have e_1698_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_1
  simp only [← c103788, ← c103789, k2, ← c103790] at e_1698_1
  have hr := e_1698_1
  simp only [e_1698_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17059)) := by
  have c103791 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103792 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103793 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103794 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103795 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103796 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_2
  simp only [← c103791, ← c103792, ← c103793] at e_1698_2
  have e_1698_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_3
  simp only [← c103794, ← c103795, k2, ← c103796] at e_1698_3
  have hr := e_1698_3
  simp only [e_1698_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17060)) := by
  have c103797 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103798 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103799 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103800 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103801 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103802 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_4
  simp only [← c103797, ← c103798, ← c103799] at e_1698_4
  have e_1698_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_5
  simp only [← c103800, ← c103801, k2, ← c103802] at e_1698_5
  have hr := e_1698_5
  simp only [e_1698_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17061)) := by
  have c103803 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103804 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103805 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103806 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103807 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103808 := (publicBatchWrapper86_copies3244 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_6
  simp only [← c103803, ← c103804, ← c103805] at e_1698_6
  have e_1698_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_7
  simp only [← c103806, ← c103807, k2, ← c103808] at e_1698_7
  have hr := e_1698_7
  simp only [e_1698_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17062)) := by
  have c103809 := (publicBatchWrapper86_copies3244 a h).2.1
  have c103810 := (publicBatchWrapper86_copies3244 a h).2.2.1
  have c103811 := (publicBatchWrapper86_copies3244 a h).2.2.2.1
  have c103812 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.1
  have c103813 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.1
  have c103814 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_8
  simp only [← c103809, ← c103810, ← c103811] at e_1698_8
  have e_1698_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_9
  simp only [← c103812, ← c103813, k2, ← c103814] at e_1698_9
  have hr := e_1698_9
  simp only [e_1698_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17063)) := by
  have c103815 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.1
  have c103816 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.1
  have c103817 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.1
  have c103818 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103819 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103820 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_10
  simp only [← c103815, ← c103816, ← c103817] at e_1698_10
  have e_1698_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_11
  simp only [← c103818, ← c103819, k2, ← c103820] at e_1698_11
  have hr := e_1698_11
  simp only [e_1698_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17064)) := by
  have c103821 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103822 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103823 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103824 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103825 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103826 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_12
  simp only [← c103821, ← c103822, ← c103823] at e_1698_12
  have e_1698_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_13
  simp only [← c103824, ← c103825, k2, ← c103826] at e_1698_13
  have hr := e_1698_13
  simp only [e_1698_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17065)) := by
  have c103827 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103828 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103829 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103830 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103831 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103832 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_14
  simp only [← c103827, ← c103828, ← c103829] at e_1698_14
  have e_1698_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_15
  simp only [← c103830, ← c103831, k2, ← c103832] at e_1698_15
  have hr := e_1698_15
  simp only [e_1698_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17066)) := by
  have c103833 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103834 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103835 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103836 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103837 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103838 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_16
  simp only [← c103833, ← c103834, ← c103835] at e_1698_16
  have e_1698_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_17
  simp only [← c103836, ← c103837, k2, ← c103838] at e_1698_17
  have hr := e_1698_17
  simp only [e_1698_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17067)) := by
  have c103839 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103840 := (publicBatchWrapper86_copies3245 a h).1
  have c103841 := (publicBatchWrapper86_copies3245 a h).2.1
  have c103842 := (publicBatchWrapper86_copies3245 a h).2.2.1
  have c103843 := (publicBatchWrapper86_copies3245 a h).2.2.2.1
  have c103844 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_18
  simp only [← c103839, ← c103840, ← c103841] at e_1698_18
  have e_1698_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_19
  simp only [← c103842, ← c103843, k2, ← c103844] at e_1698_19
  have hr := e_1698_19
  simp only [e_1698_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17068)) := by
  have c103845 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.1
  have c103846 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.1
  have c103847 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.1
  have c103848 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.1
  have c103849 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.1
  have c103850 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_0
  simp only [← c103845, ← c103846, ← c103847] at e_1699_0
  have e_1699_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_1
  simp only [← c103848, ← c103849, k2, ← c103850] at e_1699_1
  have hr := e_1699_1
  simp only [e_1699_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17069)) := by
  have c103851 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103852 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103853 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103854 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103855 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103856 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_2
  simp only [← c103851, ← c103852, ← c103853] at e_1699_2
  have e_1699_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_3
  simp only [← c103854, ← c103855, k2, ← c103856] at e_1699_3
  have hr := e_1699_3
  simp only [e_1699_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17070)) := by
  have c103857 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103858 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103859 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103860 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103861 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103862 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_4
  simp only [← c103857, ← c103858, ← c103859] at e_1699_4
  have e_1699_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_5
  simp only [← c103860, ← c103861, k2, ← c103862] at e_1699_5
  have hr := e_1699_5
  simp only [e_1699_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17071)) := by
  have c103863 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103864 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103865 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103866 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103867 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103868 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_6
  simp only [← c103863, ← c103864, ← c103865] at e_1699_6
  have e_1699_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_7
  simp only [← c103866, ← c103867, k2, ← c103868] at e_1699_7
  have hr := e_1699_7
  simp only [e_1699_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17072)) := by
  have c103869 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103870 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103871 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103872 := (publicBatchWrapper86_copies3246 a h).1
  have c103873 := (publicBatchWrapper86_copies3246 a h).2.1
  have c103874 := (publicBatchWrapper86_copies3246 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_8
  simp only [← c103869, ← c103870, ← c103871] at e_1699_8
  have e_1699_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_9
  simp only [← c103872, ← c103873, k2, ← c103874] at e_1699_9
  have hr := e_1699_9
  simp only [e_1699_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17073)) := by
  have c103875 := (publicBatchWrapper86_copies3246 a h).2.2.2.1
  have c103876 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.1
  have c103877 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.1
  have c103878 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.1
  have c103879 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.1
  have c103880 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_10
  simp only [← c103875, ← c103876, ← c103877] at e_1699_10
  have e_1699_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_11
  simp only [← c103878, ← c103879, k2, ← c103880] at e_1699_11
  have hr := e_1699_11
  simp only [e_1699_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17074)) := by
  have c103881 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.1
  have c103882 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103883 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103884 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103885 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103886 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_12
  simp only [← c103881, ← c103882, ← c103883] at e_1699_12
  have e_1699_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_13
  simp only [← c103884, ← c103885, k2, ← c103886] at e_1699_13
  have hr := e_1699_13
  simp only [e_1699_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17075)) := by
  have c103887 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103888 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103889 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103890 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103891 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103892 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_14
  simp only [← c103887, ← c103888, ← c103889] at e_1699_14
  have e_1699_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_15
  simp only [← c103890, ← c103891, k2, ← c103892] at e_1699_15
  have hr := e_1699_15
  simp only [e_1699_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17076)) := by
  have c103893 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103894 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103895 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103896 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103897 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103898 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_16
  simp only [← c103893, ← c103894, ← c103895] at e_1699_16
  have e_1699_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_17
  simp only [← c103896, ← c103897, k2, ← c103898] at e_1699_17
  have hr := e_1699_17
  simp only [e_1699_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17077)) := by
  have c103899 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103900 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103901 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103902 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103903 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103904 := (publicBatchWrapper86_copies3247 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_18
  simp only [← c103899, ← c103900, ← c103901] at e_1699_18
  have e_1699_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_19
  simp only [← c103902, ← c103903, k2, ← c103904] at e_1699_19
  have hr := e_1699_19
  simp only [e_1699_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17078)) := by
  have c103905 := (publicBatchWrapper86_copies3247 a h).2.1
  have c103906 := (publicBatchWrapper86_copies3247 a h).2.2.1
  have c103907 := (publicBatchWrapper86_copies3247 a h).2.2.2.1
  have c103908 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.1
  have c103909 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.1
  have c103910 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_0
  simp only [← c103905, ← c103906, ← c103907] at e_1700_0
  have e_1700_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_1
  simp only [← c103908, ← c103909, k2, ← c103910] at e_1700_1
  have hr := e_1700_1
  simp only [e_1700_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17079)) := by
  have c103911 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.1
  have c103912 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.1
  have c103913 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.1
  have c103914 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103915 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103916 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_2
  simp only [← c103911, ← c103912, ← c103913] at e_1700_2
  have e_1700_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_3
  simp only [← c103914, ← c103915, k2, ← c103916] at e_1700_3
  have hr := e_1700_3
  simp only [e_1700_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17080)) := by
  have c103917 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103918 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103919 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103920 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103921 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103922 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_4
  simp only [← c103917, ← c103918, ← c103919] at e_1700_4
  have e_1700_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_5
  simp only [← c103920, ← c103921, k2, ← c103922] at e_1700_5
  have hr := e_1700_5
  simp only [e_1700_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17081)) := by
  have c103923 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103924 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103925 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103926 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103927 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103928 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_6
  simp only [← c103923, ← c103924, ← c103925] at e_1700_6
  have e_1700_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_7
  simp only [← c103926, ← c103927, k2, ← c103928] at e_1700_7
  have hr := e_1700_7
  simp only [e_1700_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17082)) := by
  have c103929 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103930 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103931 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103932 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103933 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103934 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_8
  simp only [← c103929, ← c103930, ← c103931] at e_1700_8
  have e_1700_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_9
  simp only [← c103932, ← c103933, k2, ← c103934] at e_1700_9
  have hr := e_1700_9
  simp only [e_1700_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17083)) := by
  have c103935 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103936 := (publicBatchWrapper86_copies3248 a h).1
  have c103937 := (publicBatchWrapper86_copies3248 a h).2.1
  have c103938 := (publicBatchWrapper86_copies3248 a h).2.2.1
  have c103939 := (publicBatchWrapper86_copies3248 a h).2.2.2.1
  have c103940 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_10
  simp only [← c103935, ← c103936, ← c103937] at e_1700_10
  have e_1700_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_11
  simp only [← c103938, ← c103939, k2, ← c103940] at e_1700_11
  have hr := e_1700_11
  simp only [e_1700_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17084)) := by
  have c103941 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.1
  have c103942 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.1
  have c103943 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.1
  have c103944 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.1
  have c103945 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.1
  have c103946 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_12
  simp only [← c103941, ← c103942, ← c103943] at e_1700_12
  have e_1700_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_13
  simp only [← c103944, ← c103945, k2, ← c103946] at e_1700_13
  have hr := e_1700_13
  simp only [e_1700_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17085)) := by
  have c103947 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103948 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103949 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103950 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103951 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103952 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_14
  simp only [← c103947, ← c103948, ← c103949] at e_1700_14
  have e_1700_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_15
  simp only [← c103950, ← c103951, k2, ← c103952] at e_1700_15
  have hr := e_1700_15
  simp only [e_1700_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17086)) := by
  have c103953 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103954 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103955 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103956 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103957 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103958 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_16
  simp only [← c103953, ← c103954, ← c103955] at e_1700_16
  have e_1700_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_17
  simp only [← c103956, ← c103957, k2, ← c103958] at e_1700_17
  have hr := e_1700_17
  simp only [e_1700_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17087)) := by
  have c103959 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103960 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103961 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103962 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103963 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103964 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_18
  simp only [← c103959, ← c103960, ← c103961] at e_1700_18
  have e_1700_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_19
  simp only [← c103962, ← c103963, k2, ← c103964] at e_1700_19
  have hr := e_1700_19
  simp only [e_1700_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17088)) := by
  have c103965 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103966 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103967 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103968 := (publicBatchWrapper86_copies3249 a h).1
  have c103969 := (publicBatchWrapper86_copies3249 a h).2.1
  have c103970 := (publicBatchWrapper86_copies3249 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_0
  simp only [← c103965, ← c103966, ← c103967] at e_1701_0
  have e_1701_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_1
  simp only [← c103968, ← c103969, k2, ← c103970] at e_1701_1
  have hr := e_1701_1
  simp only [e_1701_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17089)) := by
  have c103971 := (publicBatchWrapper86_copies3249 a h).2.2.2.1
  have c103972 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.1
  have c103973 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.1
  have c103974 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.1
  have c103975 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.1
  have c103976 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_2
  simp only [← c103971, ← c103972, ← c103973] at e_1701_2
  have e_1701_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_3
  simp only [← c103974, ← c103975, k2, ← c103976] at e_1701_3
  have hr := e_1701_3
  simp only [e_1701_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17090)) := by
  have c103977 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.1
  have c103978 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103979 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103980 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103981 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103982 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_4
  simp only [← c103977, ← c103978, ← c103979] at e_1701_4
  have e_1701_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_5
  simp only [← c103980, ← c103981, k2, ← c103982] at e_1701_5
  have hr := e_1701_5
  simp only [e_1701_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17091)) := by
  have c103983 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103984 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103985 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103986 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103987 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103988 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_6
  simp only [← c103983, ← c103984, ← c103985] at e_1701_6
  have e_1701_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_7
  simp only [← c103986, ← c103987, k2, ← c103988] at e_1701_7
  have hr := e_1701_7
  simp only [e_1701_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17092)) := by
  have c103989 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103990 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103991 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103992 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103993 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103994 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_8
  simp only [← c103989, ← c103990, ← c103991] at e_1701_8
  have e_1701_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_9
  simp only [← c103992, ← c103993, k2, ← c103994] at e_1701_9
  have hr := e_1701_9
  simp only [e_1701_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17093)) := by
  have c103995 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103996 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103997 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103998 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103999 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104000 := (publicBatchWrapper86_copies3250 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_10
  simp only [← c103995, ← c103996, ← c103997] at e_1701_10
  have e_1701_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_11
  simp only [← c103998, ← c103999, k2, ← c104000] at e_1701_11
  have hr := e_1701_11
  simp only [e_1701_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17094)) := by
  have c104001 := (publicBatchWrapper86_copies3250 a h).2.1
  have c104002 := (publicBatchWrapper86_copies3250 a h).2.2.1
  have c104003 := (publicBatchWrapper86_copies3250 a h).2.2.2.1
  have c104004 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.1
  have c104005 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.1
  have c104006 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_12
  simp only [← c104001, ← c104002, ← c104003] at e_1701_12
  have e_1701_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_13
  simp only [← c104004, ← c104005, k2, ← c104006] at e_1701_13
  have hr := e_1701_13
  simp only [e_1701_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17095)) := by
  have c104007 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.1
  have c104008 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.1
  have c104009 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.1
  have c104010 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104011 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104012 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_14
  simp only [← c104007, ← c104008, ← c104009] at e_1701_14
  have e_1701_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_15
  simp only [← c104010, ← c104011, k2, ← c104012] at e_1701_15
  have hr := e_1701_15
  simp only [e_1701_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17096)) := by
  have c104013 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104014 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104015 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104016 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104017 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104018 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_16
  simp only [← c104013, ← c104014, ← c104015] at e_1701_16
  have e_1701_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_17
  simp only [← c104016, ← c104017, k2, ← c104018] at e_1701_17
  have hr := e_1701_17
  simp only [e_1701_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17097)) := by
  have c104019 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104020 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104021 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104022 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104023 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104024 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_18
  simp only [← c104019, ← c104020, ← c104021] at e_1701_18
  have e_1701_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_19
  simp only [← c104022, ← c104023, k2, ← c104024] at e_1701_19
  have hr := e_1701_19
  simp only [e_1701_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17098)) := by
  have c104025 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104026 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104027 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104028 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104029 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104030 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_0
  simp only [← c104025, ← c104026, ← c104027] at e_1702_0
  have e_1702_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_1
  simp only [← c104028, ← c104029, k2, ← c104030] at e_1702_1
  have hr := e_1702_1
  simp only [e_1702_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17099)) := by
  have c104031 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104032 := (publicBatchWrapper86_copies3251 a h).1
  have c104033 := (publicBatchWrapper86_copies3251 a h).2.1
  have c104034 := (publicBatchWrapper86_copies3251 a h).2.2.1
  have c104035 := (publicBatchWrapper86_copies3251 a h).2.2.2.1
  have c104036 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_2
  simp only [← c104031, ← c104032, ← c104033] at e_1702_2
  have e_1702_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_3
  simp only [← c104034, ← c104035, k2, ← c104036] at e_1702_3
  have hr := e_1702_3
  simp only [e_1702_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17100)) := by
  have c104037 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.1
  have c104038 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.1
  have c104039 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.1
  have c104040 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.1
  have c104041 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.1
  have c104042 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_4
  simp only [← c104037, ← c104038, ← c104039] at e_1702_4
  have e_1702_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_5
  simp only [← c104040, ← c104041, k2, ← c104042] at e_1702_5
  have hr := e_1702_5
  simp only [e_1702_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17101)) := by
  have c104043 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104044 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104045 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104046 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104047 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104048 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_6
  simp only [← c104043, ← c104044, ← c104045] at e_1702_6
  have e_1702_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_7
  simp only [← c104046, ← c104047, k2, ← c104048] at e_1702_7
  have hr := e_1702_7
  simp only [e_1702_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17102)) := by
  have c104049 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104050 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104051 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104052 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104053 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104054 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_8
  simp only [← c104049, ← c104050, ← c104051] at e_1702_8
  have e_1702_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_9
  simp only [← c104052, ← c104053, k2, ← c104054] at e_1702_9
  have hr := e_1702_9
  simp only [e_1702_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17103)) := by
  have c104055 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104056 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104057 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104058 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104059 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104060 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_10
  simp only [← c104055, ← c104056, ← c104057] at e_1702_10
  have e_1702_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_11
  simp only [← c104058, ← c104059, k2, ← c104060] at e_1702_11
  have hr := e_1702_11
  simp only [e_1702_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17104)) := by
  have c104061 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104062 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104063 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104064 := (publicBatchWrapper86_copies3252 a h).1
  have c104065 := (publicBatchWrapper86_copies3252 a h).2.1
  have c104066 := (publicBatchWrapper86_copies3252 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_12
  simp only [← c104061, ← c104062, ← c104063] at e_1702_12
  have e_1702_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_13
  simp only [← c104064, ← c104065, k2, ← c104066] at e_1702_13
  have hr := e_1702_13
  simp only [e_1702_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17105)) := by
  have c104067 := (publicBatchWrapper86_copies3252 a h).2.2.2.1
  have c104068 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.1
  have c104069 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.1
  have c104070 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.1
  have c104071 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.1
  have c104072 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_14
  simp only [← c104067, ← c104068, ← c104069] at e_1702_14
  have e_1702_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_15
  simp only [← c104070, ← c104071, k2, ← c104072] at e_1702_15
  have hr := e_1702_15
  simp only [e_1702_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17106)) := by
  have c104073 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.1
  have c104074 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104075 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104076 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104077 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104078 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_16
  simp only [← c104073, ← c104074, ← c104075] at e_1702_16
  have e_1702_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_17
  simp only [← c104076, ← c104077, k2, ← c104078] at e_1702_17
  have hr := e_1702_17
  simp only [e_1702_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17107)) := by
  have c104079 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104080 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104081 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104082 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104083 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104084 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_18
  simp only [← c104079, ← c104080, ← c104081] at e_1702_18
  have e_1702_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_19
  simp only [← c104082, ← c104083, k2, ← c104084] at e_1702_19
  have hr := e_1702_19
  simp only [e_1702_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17108)) := by
  have c104085 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104086 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104087 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104088 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104089 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104090 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_0
  simp only [← c104085, ← c104086, ← c104087] at e_1703_0
  have e_1703_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_1
  simp only [← c104088, ← c104089, k2, ← c104090] at e_1703_1
  have hr := e_1703_1
  simp only [e_1703_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17109)) := by
  have c104091 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104092 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104093 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104094 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104095 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104096 := (publicBatchWrapper86_copies3253 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_2
  simp only [← c104091, ← c104092, ← c104093] at e_1703_2
  have e_1703_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_3
  simp only [← c104094, ← c104095, k2, ← c104096] at e_1703_3
  have hr := e_1703_3
  simp only [e_1703_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17110)) := by
  have c104097 := (publicBatchWrapper86_copies3253 a h).2.1
  have c104098 := (publicBatchWrapper86_copies3253 a h).2.2.1
  have c104099 := (publicBatchWrapper86_copies3253 a h).2.2.2.1
  have c104100 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.1
  have c104101 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.1
  have c104102 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_4
  simp only [← c104097, ← c104098, ← c104099] at e_1703_4
  have e_1703_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_5
  simp only [← c104100, ← c104101, k2, ← c104102] at e_1703_5
  have hr := e_1703_5
  simp only [e_1703_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17111)) := by
  have c104103 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.1
  have c104104 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.1
  have c104105 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.1
  have c104106 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104107 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104108 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_6
  simp only [← c104103, ← c104104, ← c104105] at e_1703_6
  have e_1703_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_7
  simp only [← c104106, ← c104107, k2, ← c104108] at e_1703_7
  have hr := e_1703_7
  simp only [e_1703_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17112)) := by
  have c104109 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104110 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104111 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104112 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104113 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104114 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_8
  simp only [← c104109, ← c104110, ← c104111] at e_1703_8
  have e_1703_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_9
  simp only [← c104112, ← c104113, k2, ← c104114] at e_1703_9
  have hr := e_1703_9
  simp only [e_1703_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17113)) := by
  have c104115 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104116 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104117 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104118 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104119 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104120 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_10
  simp only [← c104115, ← c104116, ← c104117] at e_1703_10
  have e_1703_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_11
  simp only [← c104118, ← c104119, k2, ← c104120] at e_1703_11
  have hr := e_1703_11
  simp only [e_1703_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17114)) := by
  have c104121 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104122 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104123 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104124 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104125 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104126 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_12
  simp only [← c104121, ← c104122, ← c104123] at e_1703_12
  have e_1703_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_13
  simp only [← c104124, ← c104125, k2, ← c104126] at e_1703_13
  have hr := e_1703_13
  simp only [e_1703_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17115)) := by
  have c104127 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104128 := (publicBatchWrapper86_copies3254 a h).1
  have c104129 := (publicBatchWrapper86_copies3254 a h).2.1
  have c104130 := (publicBatchWrapper86_copies3254 a h).2.2.1
  have c104131 := (publicBatchWrapper86_copies3254 a h).2.2.2.1
  have c104132 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_14
  simp only [← c104127, ← c104128, ← c104129] at e_1703_14
  have e_1703_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_15
  simp only [← c104130, ← c104131, k2, ← c104132] at e_1703_15
  have hr := e_1703_15
  simp only [e_1703_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17116)) := by
  have c104133 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.1
  have c104134 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.1
  have c104135 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.1
  have c104136 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.1
  have c104137 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.1
  have c104138 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_16
  simp only [← c104133, ← c104134, ← c104135] at e_1703_16
  have e_1703_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_17
  simp only [← c104136, ← c104137, k2, ← c104138] at e_1703_17
  have hr := e_1703_17
  simp only [e_1703_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17117)) := by
  have c104139 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104140 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104141 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104142 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104143 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104144 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_18
  simp only [← c104139, ← c104140, ← c104141] at e_1703_18
  have e_1703_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_19
  simp only [← c104142, ← c104143, k2, ← c104144] at e_1703_19
  have hr := e_1703_19
  simp only [e_1703_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17118)) := by
  have c104145 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104146 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104147 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104148 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104149 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104150 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_0
  simp only [← c104145, ← c104146, ← c104147] at e_1704_0
  have e_1704_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_1
  simp only [← c104148, ← c104149, k2, ← c104150] at e_1704_1
  have hr := e_1704_1
  simp only [e_1704_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17119)) := by
  have c104151 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104152 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104153 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104154 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104155 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104156 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_2
  simp only [← c104151, ← c104152, ← c104153] at e_1704_2
  have e_1704_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_3
  simp only [← c104154, ← c104155, k2, ← c104156] at e_1704_3
  have hr := e_1704_3
  simp only [e_1704_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17120)) := by
  have c104157 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104158 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104159 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104160 := (publicBatchWrapper86_copies3255 a h).1
  have c104161 := (publicBatchWrapper86_copies3255 a h).2.1
  have c104162 := (publicBatchWrapper86_copies3255 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_4
  simp only [← c104157, ← c104158, ← c104159] at e_1704_4
  have e_1704_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_5
  simp only [← c104160, ← c104161, k2, ← c104162] at e_1704_5
  have hr := e_1704_5
  simp only [e_1704_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17121)) := by
  have c104163 := (publicBatchWrapper86_copies3255 a h).2.2.2.1
  have c104164 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.1
  have c104165 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.1
  have c104166 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.1
  have c104167 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.1
  have c104168 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_6
  simp only [← c104163, ← c104164, ← c104165] at e_1704_6
  have e_1704_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_7
  simp only [← c104166, ← c104167, k2, ← c104168] at e_1704_7
  have hr := e_1704_7
  simp only [e_1704_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17122)) := by
  have c104169 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.1
  have c104170 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104171 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104172 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104173 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104174 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_8
  simp only [← c104169, ← c104170, ← c104171] at e_1704_8
  have e_1704_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_9
  simp only [← c104172, ← c104173, k2, ← c104174] at e_1704_9
  have hr := e_1704_9
  simp only [e_1704_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17123)) := by
  have c104175 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104176 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104177 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104178 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104179 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104180 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_10
  simp only [← c104175, ← c104176, ← c104177] at e_1704_10
  have e_1704_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_11
  simp only [← c104178, ← c104179, k2, ← c104180] at e_1704_11
  have hr := e_1704_11
  simp only [e_1704_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17124)) := by
  have c104181 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104182 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104183 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104184 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104185 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104186 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_12
  simp only [← c104181, ← c104182, ← c104183] at e_1704_12
  have e_1704_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_13
  simp only [← c104184, ← c104185, k2, ← c104186] at e_1704_13
  have hr := e_1704_13
  simp only [e_1704_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17125)) := by
  have c104187 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104188 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104189 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104190 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104191 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104192 := (publicBatchWrapper86_copies3256 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_14
  simp only [← c104187, ← c104188, ← c104189] at e_1704_14
  have e_1704_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_15
  simp only [← c104190, ← c104191, k2, ← c104192] at e_1704_15
  have hr := e_1704_15
  simp only [e_1704_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17126)) := by
  have c104193 := (publicBatchWrapper86_copies3256 a h).2.1
  have c104194 := (publicBatchWrapper86_copies3256 a h).2.2.1
  have c104195 := (publicBatchWrapper86_copies3256 a h).2.2.2.1
  have c104196 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.1
  have c104197 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.1
  have c104198 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_16
  simp only [← c104193, ← c104194, ← c104195] at e_1704_16
  have e_1704_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_17
  simp only [← c104196, ← c104197, k2, ← c104198] at e_1704_17
  have hr := e_1704_17
  simp only [e_1704_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17127)) := by
  have c104199 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.1
  have c104200 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.1
  have c104201 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.1
  have c104202 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104203 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104204 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_18
  simp only [← c104199, ← c104200, ← c104201] at e_1704_18
  have e_1704_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_19
  simp only [← c104202, ← c104203, k2, ← c104204] at e_1704_19
  have hr := e_1704_19
  simp only [e_1704_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17128)) := by
  have c104205 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104206 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104207 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104208 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104209 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104210 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_0
  simp only [← c104205, ← c104206, ← c104207] at e_1705_0
  have e_1705_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_1
  simp only [← c104208, ← c104209, k2, ← c104210] at e_1705_1
  have hr := e_1705_1
  simp only [e_1705_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17129)) := by
  have c104211 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104212 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104213 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104214 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104215 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104216 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_2
  simp only [← c104211, ← c104212, ← c104213] at e_1705_2
  have e_1705_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_3
  simp only [← c104214, ← c104215, k2, ← c104216] at e_1705_3
  have hr := e_1705_3
  simp only [e_1705_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17130)) := by
  have c104217 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104218 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104219 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104220 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104221 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104222 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_4
  simp only [← c104217, ← c104218, ← c104219] at e_1705_4
  have e_1705_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_5
  simp only [← c104220, ← c104221, k2, ← c104222] at e_1705_5
  have hr := e_1705_5
  simp only [e_1705_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17131)) := by
  have c104223 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104224 := (publicBatchWrapper86_copies3257 a h).1
  have c104225 := (publicBatchWrapper86_copies3257 a h).2.1
  have c104226 := (publicBatchWrapper86_copies3257 a h).2.2.1
  have c104227 := (publicBatchWrapper86_copies3257 a h).2.2.2.1
  have c104228 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_6
  simp only [← c104223, ← c104224, ← c104225] at e_1705_6
  have e_1705_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_7
  simp only [← c104226, ← c104227, k2, ← c104228] at e_1705_7
  have hr := e_1705_7
  simp only [e_1705_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17132)) := by
  have c104229 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.1
  have c104230 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.1
  have c104231 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.1
  have c104232 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.1
  have c104233 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.1
  have c104234 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_8
  simp only [← c104229, ← c104230, ← c104231] at e_1705_8
  have e_1705_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_9
  simp only [← c104232, ← c104233, k2, ← c104234] at e_1705_9
  have hr := e_1705_9
  simp only [e_1705_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17133)) := by
  have c104235 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104236 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104237 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104238 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104239 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104240 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_10
  simp only [← c104235, ← c104236, ← c104237] at e_1705_10
  have e_1705_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_11
  simp only [← c104238, ← c104239, k2, ← c104240] at e_1705_11
  have hr := e_1705_11
  simp only [e_1705_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17134)) := by
  have c104241 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104242 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104243 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104244 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104245 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104246 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_12
  simp only [← c104241, ← c104242, ← c104243] at e_1705_12
  have e_1705_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_13
  simp only [← c104244, ← c104245, k2, ← c104246] at e_1705_13
  have hr := e_1705_13
  simp only [e_1705_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17135)) := by
  have c104247 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104248 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104249 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104250 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104251 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104252 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_14
  simp only [← c104247, ← c104248, ← c104249] at e_1705_14
  have e_1705_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_15
  simp only [← c104250, ← c104251, k2, ← c104252] at e_1705_15
  have hr := e_1705_15
  simp only [e_1705_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17136)) := by
  have c104253 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104254 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104255 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104256 := (publicBatchWrapper86_copies3258 a h).1
  have c104257 := (publicBatchWrapper86_copies3258 a h).2.1
  have c104258 := (publicBatchWrapper86_copies3258 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_16
  simp only [← c104253, ← c104254, ← c104255] at e_1705_16
  have e_1705_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_17
  simp only [← c104256, ← c104257, k2, ← c104258] at e_1705_17
  have hr := e_1705_17
  simp only [e_1705_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17137)) := by
  have c104259 := (publicBatchWrapper86_copies3258 a h).2.2.2.1
  have c104260 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.1
  have c104261 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.1
  have c104262 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.1
  have c104263 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.1
  have c104264 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_18
  simp only [← c104259, ← c104260, ← c104261] at e_1705_18
  have e_1705_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_19
  simp only [← c104262, ← c104263, k2, ← c104264] at e_1705_19
  have hr := e_1705_19
  simp only [e_1705_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17138)) := by
  have c104265 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.1
  have c104266 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104267 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104268 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104269 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104270 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_0
  simp only [← c104265, ← c104266, ← c104267] at e_1706_0
  have e_1706_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_1
  simp only [← c104268, ← c104269, k2, ← c104270] at e_1706_1
  have hr := e_1706_1
  simp only [e_1706_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17139)) := by
  have c104271 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104272 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104273 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104274 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104275 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104276 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_2
  simp only [← c104271, ← c104272, ← c104273] at e_1706_2
  have e_1706_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_3
  simp only [← c104274, ← c104275, k2, ← c104276] at e_1706_3
  have hr := e_1706_3
  simp only [e_1706_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17140)) := by
  have c104277 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104278 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104279 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104280 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104281 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104282 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_4
  simp only [← c104277, ← c104278, ← c104279] at e_1706_4
  have e_1706_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_5
  simp only [← c104280, ← c104281, k2, ← c104282] at e_1706_5
  have hr := e_1706_5
  simp only [e_1706_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17141)) := by
  have c104283 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104284 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104285 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104286 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104287 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104288 := (publicBatchWrapper86_copies3259 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_6
  simp only [← c104283, ← c104284, ← c104285] at e_1706_6
  have e_1706_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_7
  simp only [← c104286, ← c104287, k2, ← c104288] at e_1706_7
  have hr := e_1706_7
  simp only [e_1706_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17142)) := by
  have c104289 := (publicBatchWrapper86_copies3259 a h).2.1
  have c104290 := (publicBatchWrapper86_copies3259 a h).2.2.1
  have c104291 := (publicBatchWrapper86_copies3259 a h).2.2.2.1
  have c104292 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.1
  have c104293 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.1
  have c104294 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_8
  simp only [← c104289, ← c104290, ← c104291] at e_1706_8
  have e_1706_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_9
  simp only [← c104292, ← c104293, k2, ← c104294] at e_1706_9
  have hr := e_1706_9
  simp only [e_1706_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17143)) := by
  have c104295 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.1
  have c104296 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.1
  have c104297 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.1
  have c104298 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104299 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104300 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_10
  simp only [← c104295, ← c104296, ← c104297] at e_1706_10
  have e_1706_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_11
  simp only [← c104298, ← c104299, k2, ← c104300] at e_1706_11
  have hr := e_1706_11
  simp only [e_1706_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17144)) := by
  have c104301 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104302 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104303 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104304 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104305 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104306 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_12
  simp only [← c104301, ← c104302, ← c104303] at e_1706_12
  have e_1706_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_13
  simp only [← c104304, ← c104305, k2, ← c104306] at e_1706_13
  have hr := e_1706_13
  simp only [e_1706_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17145)) := by
  have c104307 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104308 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104309 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104310 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104311 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104312 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_14
  simp only [← c104307, ← c104308, ← c104309] at e_1706_14
  have e_1706_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_15
  simp only [← c104310, ← c104311, k2, ← c104312] at e_1706_15
  have hr := e_1706_15
  simp only [e_1706_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17146)) := by
  have c104313 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104314 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104315 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104316 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104317 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104318 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_16
  simp only [← c104313, ← c104314, ← c104315] at e_1706_16
  have e_1706_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_17
  simp only [← c104316, ← c104317, k2, ← c104318] at e_1706_17
  have hr := e_1706_17
  simp only [e_1706_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17147)) := by
  have c104319 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104320 := (publicBatchWrapper86_copies3260 a h).1
  have c104321 := (publicBatchWrapper86_copies3260 a h).2.1
  have c104322 := (publicBatchWrapper86_copies3260 a h).2.2.1
  have c104323 := (publicBatchWrapper86_copies3260 a h).2.2.2.1
  have c104324 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_18
  simp only [← c104319, ← c104320, ← c104321] at e_1706_18
  have e_1706_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_19
  simp only [← c104322, ← c104323, k2, ← c104324] at e_1706_19
  have hr := e_1706_19
  simp only [e_1706_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17148)) := by
  have c104325 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.1
  have c104326 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.1
  have c104327 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.1
  have c104328 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.1
  have c104329 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.1
  have c104330 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_0
  simp only [← c104325, ← c104326, ← c104327] at e_1707_0
  have e_1707_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_1
  simp only [← c104328, ← c104329, k2, ← c104330] at e_1707_1
  have hr := e_1707_1
  simp only [e_1707_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17149)) := by
  have c104331 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104332 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104333 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104334 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104335 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104336 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_2
  simp only [← c104331, ← c104332, ← c104333] at e_1707_2
  have e_1707_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_3
  simp only [← c104334, ← c104335, k2, ← c104336] at e_1707_3
  have hr := e_1707_3
  simp only [e_1707_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17150)) := by
  have c104337 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104338 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104339 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104340 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104341 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104342 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_4
  simp only [← c104337, ← c104338, ← c104339] at e_1707_4
  have e_1707_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_5
  simp only [← c104340, ← c104341, k2, ← c104342] at e_1707_5
  have hr := e_1707_5
  simp only [e_1707_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17151)) := by
  have c104343 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104344 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104345 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104346 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104347 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104348 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_6
  simp only [← c104343, ← c104344, ← c104345] at e_1707_6
  have e_1707_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_7
  simp only [← c104346, ← c104347, k2, ← c104348] at e_1707_7
  have hr := e_1707_7
  simp only [e_1707_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17152)) := by
  have c104349 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104350 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104351 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104352 := (publicBatchWrapper86_copies3261 a h).1
  have c104353 := (publicBatchWrapper86_copies3261 a h).2.1
  have c104354 := (publicBatchWrapper86_copies3261 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_8
  simp only [← c104349, ← c104350, ← c104351] at e_1707_8
  have e_1707_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_9
  simp only [← c104352, ← c104353, k2, ← c104354] at e_1707_9
  have hr := e_1707_9
  simp only [e_1707_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17153)) := by
  have c104355 := (publicBatchWrapper86_copies3261 a h).2.2.2.1
  have c104356 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.1
  have c104357 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.1
  have c104358 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.1
  have c104359 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.1
  have c104360 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_10
  simp only [← c104355, ← c104356, ← c104357] at e_1707_10
  have e_1707_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_11
  simp only [← c104358, ← c104359, k2, ← c104360] at e_1707_11
  have hr := e_1707_11
  simp only [e_1707_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17154)) := by
  have c104361 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.1
  have c104362 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104363 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104364 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104365 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104366 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_12
  simp only [← c104361, ← c104362, ← c104363] at e_1707_12
  have e_1707_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_13
  simp only [← c104364, ← c104365, k2, ← c104366] at e_1707_13
  have hr := e_1707_13
  simp only [e_1707_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17155)) := by
  have c104367 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104368 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104369 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104370 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104371 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104372 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_14
  simp only [← c104367, ← c104368, ← c104369] at e_1707_14
  have e_1707_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_15
  simp only [← c104370, ← c104371, k2, ← c104372] at e_1707_15
  have hr := e_1707_15
  simp only [e_1707_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17156)) := by
  have c104373 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104374 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104375 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104376 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104377 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104378 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_16
  simp only [← c104373, ← c104374, ← c104375] at e_1707_16
  have e_1707_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_17
  simp only [← c104376, ← c104377, k2, ← c104378] at e_1707_17
  have hr := e_1707_17
  simp only [e_1707_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17157)) := by
  have c104379 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104380 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104381 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104382 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104383 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104384 := (publicBatchWrapper86_copies3262 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_18
  simp only [← c104379, ← c104380, ← c104381] at e_1707_18
  have e_1707_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_19
  simp only [← c104382, ← c104383, k2, ← c104384] at e_1707_19
  have hr := e_1707_19
  simp only [e_1707_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17158)) := by
  have c104385 := (publicBatchWrapper86_copies3262 a h).2.1
  have c104386 := (publicBatchWrapper86_copies3262 a h).2.2.1
  have c104387 := (publicBatchWrapper86_copies3262 a h).2.2.2.1
  have c104388 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.1
  have c104389 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.1
  have c104390 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_0
  simp only [← c104385, ← c104386, ← c104387] at e_1708_0
  have e_1708_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_1
  simp only [← c104388, ← c104389, k2, ← c104390] at e_1708_1
  have hr := e_1708_1
  simp only [e_1708_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17159)) := by
  have c104391 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.1
  have c104392 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.1
  have c104393 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.1
  have c104394 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104395 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104396 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_2
  simp only [← c104391, ← c104392, ← c104393] at e_1708_2
  have e_1708_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_3
  simp only [← c104394, ← c104395, k2, ← c104396] at e_1708_3
  have hr := e_1708_3
  simp only [e_1708_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17160)) := by
  have c104397 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104398 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104399 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104400 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104401 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104402 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_4
  simp only [← c104397, ← c104398, ← c104399] at e_1708_4
  have e_1708_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_5
  simp only [← c104400, ← c104401, k2, ← c104402] at e_1708_5
  have hr := e_1708_5
  simp only [e_1708_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17161)) := by
  have c104403 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104404 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104405 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104406 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104407 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104408 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_6
  simp only [← c104403, ← c104404, ← c104405] at e_1708_6
  have e_1708_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_7
  simp only [← c104406, ← c104407, k2, ← c104408] at e_1708_7
  have hr := e_1708_7
  simp only [e_1708_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17162)) := by
  have c104409 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104410 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104411 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104412 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104413 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104414 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_8
  simp only [← c104409, ← c104410, ← c104411] at e_1708_8
  have e_1708_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_9
  simp only [← c104412, ← c104413, k2, ← c104414] at e_1708_9
  have hr := e_1708_9
  simp only [e_1708_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17163)) := by
  have c104415 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104416 := (publicBatchWrapper86_copies3263 a h).1
  have c104417 := (publicBatchWrapper86_copies3263 a h).2.1
  have c104418 := (publicBatchWrapper86_copies3263 a h).2.2.1
  have c104419 := (publicBatchWrapper86_copies3263 a h).2.2.2.1
  have c104420 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_10
  simp only [← c104415, ← c104416, ← c104417] at e_1708_10
  have e_1708_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_11
  simp only [← c104418, ← c104419, k2, ← c104420] at e_1708_11
  have hr := e_1708_11
  simp only [e_1708_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17164)) := by
  have c104421 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.1
  have c104422 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.1
  have c104423 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.1
  have c104424 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.1
  have c104425 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.1
  have c104426 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_12
  simp only [← c104421, ← c104422, ← c104423] at e_1708_12
  have e_1708_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_13
  simp only [← c104424, ← c104425, k2, ← c104426] at e_1708_13
  have hr := e_1708_13
  simp only [e_1708_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17165)) := by
  have c104427 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104428 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104429 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104430 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104431 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104432 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_14
  simp only [← c104427, ← c104428, ← c104429] at e_1708_14
  have e_1708_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_15
  simp only [← c104430, ← c104431, k2, ← c104432] at e_1708_15
  have hr := e_1708_15
  simp only [e_1708_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17166)) := by
  have c104433 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104434 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104435 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104436 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104437 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104438 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_16
  simp only [← c104433, ← c104434, ← c104435] at e_1708_16
  have e_1708_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_17
  simp only [← c104436, ← c104437, k2, ← c104438] at e_1708_17
  have hr := e_1708_17
  simp only [e_1708_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17167)) := by
  have c104439 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104440 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104441 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104442 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104443 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104444 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_18
  simp only [← c104439, ← c104440, ← c104441] at e_1708_18
  have e_1708_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_19
  simp only [← c104442, ← c104443, k2, ← c104444] at e_1708_19
  have hr := e_1708_19
  simp only [e_1708_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17168)) := by
  have c104445 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104446 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104447 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104448 := (publicBatchWrapper86_copies3264 a h).1
  have c104449 := (publicBatchWrapper86_copies3264 a h).2.1
  have c104450 := (publicBatchWrapper86_copies3264 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_0
  simp only [← c104445, ← c104446, ← c104447] at e_1709_0
  have e_1709_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_1
  simp only [← c104448, ← c104449, k2, ← c104450] at e_1709_1
  have hr := e_1709_1
  simp only [e_1709_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17169)) := by
  have c104451 := (publicBatchWrapper86_copies3264 a h).2.2.2.1
  have c104452 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.1
  have c104453 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.1
  have c104454 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.1
  have c104455 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.1
  have c104456 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_2
  simp only [← c104451, ← c104452, ← c104453] at e_1709_2
  have e_1709_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_3
  simp only [← c104454, ← c104455, k2, ← c104456] at e_1709_3
  have hr := e_1709_3
  simp only [e_1709_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 168)) := by
  have c104457 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.1
  have c104458 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104459 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104460 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104461 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104462 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_4
  simp only [← c104457, ← c104458, ← c104459] at e_1709_4
  have e_1709_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_5
  simp only [← c104460, ← c104461, k2, ← c104462] at e_1709_5
  have hr := e_1709_5
  simp only [e_1709_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 169)) := by
  have c104463 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104464 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104465 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104466 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104467 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104468 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_6
  simp only [← c104463, ← c104464, ← c104465] at e_1709_6
  have e_1709_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_7
  simp only [← c104466, ← c104467, k2, ← c104468] at e_1709_7
  have hr := e_1709_7
  simp only [e_1709_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 170)) := by
  have c104469 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104470 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104471 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104472 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104473 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104474 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_8
  simp only [← c104469, ← c104470, ← c104471] at e_1709_8
  have e_1709_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_9
  simp only [← c104472, ← c104473, k2, ← c104474] at e_1709_9
  have hr := e_1709_9
  simp only [e_1709_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 171)) := by
  have c104475 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104476 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104477 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104478 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104479 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104480 := (publicBatchWrapper86_copies3265 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_10
  simp only [← c104475, ← c104476, ← c104477] at e_1709_10
  have e_1709_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_11
  simp only [← c104478, ← c104479, k2, ← c104480] at e_1709_11
  have hr := e_1709_11
  simp only [e_1709_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 172)) := by
  have c104481 := (publicBatchWrapper86_copies3265 a h).2.1
  have c104482 := (publicBatchWrapper86_copies3265 a h).2.2.1
  have c104483 := (publicBatchWrapper86_copies3265 a h).2.2.2.1
  have c104484 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.1
  have c104485 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.1
  have c104486 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_12
  simp only [← c104481, ← c104482, ← c104483] at e_1709_12
  have e_1709_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_13
  simp only [← c104484, ← c104485, k2, ← c104486] at e_1709_13
  have hr := e_1709_13
  simp only [e_1709_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 173)) := by
  have c104487 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.1
  have c104488 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.1
  have c104489 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.1
  have c104490 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104491 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104492 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_14
  simp only [← c104487, ← c104488, ← c104489] at e_1709_14
  have e_1709_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_15
  simp only [← c104490, ← c104491, k2, ← c104492] at e_1709_15
  have hr := e_1709_15
  simp only [e_1709_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 174)) := by
  have c104493 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104494 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104495 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104496 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104497 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104498 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_16
  simp only [← c104493, ← c104494, ← c104495] at e_1709_16
  have e_1709_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_17
  simp only [← c104496, ← c104497, k2, ← c104498] at e_1709_17
  have hr := e_1709_17
  simp only [e_1709_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 175)) := by
  have c104499 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104500 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104501 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104502 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104503 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104504 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_18
  simp only [← c104499, ← c104500, ← c104501] at e_1709_18
  have e_1709_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_19
  simp only [← c104502, ← c104503, k2, ← c104504] at e_1709_19
  have hr := e_1709_19
  simp only [e_1709_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 176)) := by
  have c104505 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104506 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104507 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104508 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104509 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104510 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_0
  simp only [← c104505, ← c104506, ← c104507] at e_1710_0
  have e_1710_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_1
  simp only [← c104508, ← c104509, k2, ← c104510] at e_1710_1
  have hr := e_1710_1
  simp only [e_1710_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 177)) := by
  have c104511 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104512 := (publicBatchWrapper86_copies3266 a h).1
  have c104513 := (publicBatchWrapper86_copies3266 a h).2.1
  have c104514 := (publicBatchWrapper86_copies3266 a h).2.2.1
  have c104515 := (publicBatchWrapper86_copies3266 a h).2.2.2.1
  have c104516 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_2
  simp only [← c104511, ← c104512, ← c104513] at e_1710_2
  have e_1710_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_3
  simp only [← c104514, ← c104515, k2, ← c104516] at e_1710_3
  have hr := e_1710_3
  simp only [e_1710_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 178)) := by
  have c104517 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.1
  have c104518 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.1
  have c104519 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.1
  have c104520 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.1
  have c104521 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.1
  have c104522 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_4
  simp only [← c104517, ← c104518, ← c104519] at e_1710_4
  have e_1710_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_5
  simp only [← c104520, ← c104521, k2, ← c104522] at e_1710_5
  have hr := e_1710_5
  simp only [e_1710_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 179)) := by
  have c104523 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104524 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104525 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104526 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104527 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104528 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_6
  simp only [← c104523, ← c104524, ← c104525] at e_1710_6
  have e_1710_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_7
  simp only [← c104526, ← c104527, k2, ← c104528] at e_1710_7
  have hr := e_1710_7
  simp only [e_1710_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 180)) := by
  have c104529 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104530 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104531 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104532 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104533 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104534 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_8
  simp only [← c104529, ← c104530, ← c104531] at e_1710_8
  have e_1710_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_9
  simp only [← c104532, ← c104533, k2, ← c104534] at e_1710_9
  have hr := e_1710_9
  simp only [e_1710_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 181)) := by
  have c104535 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104536 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104537 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104538 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104539 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104540 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_10
  simp only [← c104535, ← c104536, ← c104537] at e_1710_10
  have e_1710_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_11
  simp only [← c104538, ← c104539, k2, ← c104540] at e_1710_11
  have hr := e_1710_11
  simp only [e_1710_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 182)) := by
  have c104541 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104542 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104543 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104544 := (publicBatchWrapper86_copies3267 a h).1
  have c104545 := (publicBatchWrapper86_copies3267 a h).2.1
  have c104546 := (publicBatchWrapper86_copies3267 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_12
  simp only [← c104541, ← c104542, ← c104543] at e_1710_12
  have e_1710_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_13
  simp only [← c104544, ← c104545, k2, ← c104546] at e_1710_13
  have hr := e_1710_13
  simp only [e_1710_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 183)) := by
  have c104547 := (publicBatchWrapper86_copies3267 a h).2.2.2.1
  have c104548 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.1
  have c104549 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.1
  have c104550 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.1
  have c104551 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.1
  have c104552 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_14
  simp only [← c104547, ← c104548, ← c104549] at e_1710_14
  have e_1710_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_15
  simp only [← c104550, ← c104551, k2, ← c104552] at e_1710_15
  have hr := e_1710_15
  simp only [e_1710_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 184)) := by
  have c104553 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.1
  have c104554 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104555 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104556 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104557 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104558 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_16
  simp only [← c104553, ← c104554, ← c104555] at e_1710_16
  have e_1710_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_17
  simp only [← c104556, ← c104557, k2, ← c104558] at e_1710_17
  have hr := e_1710_17
  simp only [e_1710_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 185)) := by
  have c104559 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104560 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104561 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104562 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104563 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104564 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_18
  simp only [← c104559, ← c104560, ← c104561] at e_1710_18
  have e_1710_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_19
  simp only [← c104562, ← c104563, k2, ← c104564] at e_1710_19
  have hr := e_1710_19
  simp only [e_1710_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 186)) := by
  have c104565 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104566 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104567 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104568 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104569 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104570 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_0
  simp only [← c104565, ← c104566, ← c104567] at e_1711_0
  have e_1711_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_1
  simp only [← c104568, ← c104569, k2, ← c104570] at e_1711_1
  have hr := e_1711_1
  simp only [e_1711_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 187)) := by
  have c104571 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104572 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104573 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104574 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104575 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104576 := (publicBatchWrapper86_copies3268 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_2
  simp only [← c104571, ← c104572, ← c104573] at e_1711_2
  have e_1711_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_3
  simp only [← c104574, ← c104575, k2, ← c104576] at e_1711_3
  have hr := e_1711_3
  simp only [e_1711_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 188)) := by
  have c104577 := (publicBatchWrapper86_copies3268 a h).2.1
  have c104578 := (publicBatchWrapper86_copies3268 a h).2.2.1
  have c104579 := (publicBatchWrapper86_copies3268 a h).2.2.2.1
  have c104580 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.1
  have c104581 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.1
  have c104582 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_4
  simp only [← c104577, ← c104578, ← c104579] at e_1711_4
  have e_1711_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_5
  simp only [← c104580, ← c104581, k2, ← c104582] at e_1711_5
  have hr := e_1711_5
  simp only [e_1711_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 189)) := by
  have c104583 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.1
  have c104584 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.1
  have c104585 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.1
  have c104586 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104587 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104588 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_6
  simp only [← c104583, ← c104584, ← c104585] at e_1711_6
  have e_1711_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_7
  simp only [← c104586, ← c104587, k2, ← c104588] at e_1711_7
  have hr := e_1711_7
  simp only [e_1711_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 190)) := by
  have c104589 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104590 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104591 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104592 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104593 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104594 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_8
  simp only [← c104589, ← c104590, ← c104591] at e_1711_8
  have e_1711_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_9
  simp only [← c104592, ← c104593, k2, ← c104594] at e_1711_9
  have hr := e_1711_9
  simp only [e_1711_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 191)) := by
  have c104595 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104596 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104597 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104598 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104599 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104600 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_10
  simp only [← c104595, ← c104596, ← c104597] at e_1711_10
  have e_1711_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_11
  simp only [← c104598, ← c104599, k2, ← c104600] at e_1711_11
  have hr := e_1711_11
  simp only [e_1711_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 192)) := by
  have c104601 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104602 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104603 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104604 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104605 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104606 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_12
  simp only [← c104601, ← c104602, ← c104603] at e_1711_12
  have e_1711_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_13
  simp only [← c104604, ← c104605, k2, ← c104606] at e_1711_13
  have hr := e_1711_13
  simp only [e_1711_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 193)) := by
  have c104607 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104608 := (publicBatchWrapper86_copies3269 a h).1
  have c104609 := (publicBatchWrapper86_copies3269 a h).2.1
  have c104610 := (publicBatchWrapper86_copies3269 a h).2.2.1
  have c104611 := (publicBatchWrapper86_copies3269 a h).2.2.2.1
  have c104612 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_14
  simp only [← c104607, ← c104608, ← c104609] at e_1711_14
  have e_1711_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_15
  simp only [← c104610, ← c104611, k2, ← c104612] at e_1711_15
  have hr := e_1711_15
  simp only [e_1711_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 194)) := by
  have c104613 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.1
  have c104614 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.1
  have c104615 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.1
  have c104616 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.1
  have c104617 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.1
  have c104618 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_16
  simp only [← c104613, ← c104614, ← c104615] at e_1711_16
  have e_1711_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_17
  simp only [← c104616, ← c104617, k2, ← c104618] at e_1711_17
  have hr := e_1711_17
  simp only [e_1711_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 195)) := by
  have c104619 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104620 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104621 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104622 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104623 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104624 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_18
  simp only [← c104619, ← c104620, ← c104621] at e_1711_18
  have e_1711_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_19
  simp only [← c104622, ← c104623, k2, ← c104624] at e_1711_19
  have hr := e_1711_19
  simp only [e_1711_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1712 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 196)) := by
  have c104625 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104626 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104627 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104628 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104629 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104630 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1712_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_0
  simp only [← c104625, ← c104626, ← c104627] at e_1712_0
  have e_1712_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_1
  simp only [← c104628, ← c104629, k2, ← c104630] at e_1712_1
  have hr := e_1712_1
  simp only [e_1712_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1712 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 197)) := by
  have c104631 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104632 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104633 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104634 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104635 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104636 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1712_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_2
  simp only [← c104631, ← c104632, ← c104633] at e_1712_2
  have e_1712_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_3
  simp only [← c104634, ← c104635, k2, ← c104636] at e_1712_3
  have hr := e_1712_3
  simp only [e_1712_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1712 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 198)) := by
  have c104637 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104638 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104639 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104640 := (publicBatchWrapper86_copies3270 a h).1
  have c104641 := (publicBatchWrapper86_copies3270 a h).2.1
  have c104642 := (publicBatchWrapper86_copies3270 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1712_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_4
  simp only [← c104637, ← c104638, ← c104639] at e_1712_4
  have e_1712_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_5
  simp only [← c104640, ← c104641, k2, ← c104642] at e_1712_5
  have hr := e_1712_5
  simp only [e_1712_4] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
