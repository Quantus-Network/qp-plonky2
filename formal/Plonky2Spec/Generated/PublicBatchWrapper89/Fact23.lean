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

theorem publicBatchWrapper89_f17767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3191)) := by
  have c111108 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.1
  have c111109 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.1
  have c111110 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.1
  have c111111 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.1
  have c111112 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.1
  have c111113 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_12 := arithEq_of_rows h (row := 1818) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_12
  simp only [← c111108, ← c111109, ← c111110] at e_1818_12
  have e_1818_13 := arithEq_of_rows h (row := 1818) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_13
  simp only [← c111111, ← c111112, k2, ← c111113] at e_1818_13
  have hr := e_1818_13
  simp only [e_1818_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3192)) := by
  have c111114 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111115 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111116 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111117 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111118 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111119 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_14 := arithEq_of_rows h (row := 1818) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_14
  simp only [← c111114, ← c111115, ← c111116] at e_1818_14
  have e_1818_15 := arithEq_of_rows h (row := 1818) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_15
  simp only [← c111117, ← c111118, k2, ← c111119] at e_1818_15
  have hr := e_1818_15
  simp only [e_1818_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3193)) := by
  have c111120 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111121 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111122 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111123 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111124 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111125 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_16 := arithEq_of_rows h (row := 1818) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_16
  simp only [← c111120, ← c111121, ← c111122] at e_1818_16
  have e_1818_17 := arithEq_of_rows h (row := 1818) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_17
  simp only [← c111123, ← c111124, k2, ← c111125] at e_1818_17
  have hr := e_1818_17
  simp only [e_1818_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3194)) := by
  have c111126 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111127 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111128 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111129 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111130 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111131 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_18 := arithEq_of_rows h (row := 1818) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_18
  simp only [← c111126, ← c111127, ← c111128] at e_1818_18
  have e_1818_19 := arithEq_of_rows h (row := 1818) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_19
  simp only [← c111129, ← c111130, k2, ← c111131] at e_1818_19
  have hr := e_1818_19
  simp only [e_1818_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3195)) := by
  have c111132 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111133 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111134 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111135 := (publicBatchWrapper89_copies3472 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111136 := (publicBatchWrapper89_copies3473 a h).1
  have c111137 := (publicBatchWrapper89_copies3473 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_0 := arithEq_of_rows h (row := 1819) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_0
  simp only [← c111132, ← c111133, ← c111134] at e_1819_0
  have e_1819_1 := arithEq_of_rows h (row := 1819) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_1
  simp only [← c111135, ← c111136, k2, ← c111137] at e_1819_1
  have hr := e_1819_1
  simp only [e_1819_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3196)) := by
  have c111138 := (publicBatchWrapper89_copies3473 a h).2.2.1
  have c111139 := (publicBatchWrapper89_copies3473 a h).2.2.2.1
  have c111140 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.1
  have c111141 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.1
  have c111142 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.1
  have c111143 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_2 := arithEq_of_rows h (row := 1819) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_2
  simp only [← c111138, ← c111139, ← c111140] at e_1819_2
  have e_1819_3 := arithEq_of_rows h (row := 1819) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_3
  simp only [← c111141, ← c111142, k2, ← c111143] at e_1819_3
  have hr := e_1819_3
  simp only [e_1819_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3197)) := by
  have c111144 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.1
  have c111145 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.1
  have c111146 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111147 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111148 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111149 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_4 := arithEq_of_rows h (row := 1819) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_4
  simp only [← c111144, ← c111145, ← c111146] at e_1819_4
  have e_1819_5 := arithEq_of_rows h (row := 1819) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_5
  simp only [← c111147, ← c111148, k2, ← c111149] at e_1819_5
  have hr := e_1819_5
  simp only [e_1819_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3198)) := by
  have c111150 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111151 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111152 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111153 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111154 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111155 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_6 := arithEq_of_rows h (row := 1819) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_6
  simp only [← c111150, ← c111151, ← c111152] at e_1819_6
  have e_1819_7 := arithEq_of_rows h (row := 1819) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_7
  simp only [← c111153, ← c111154, k2, ← c111155] at e_1819_7
  have hr := e_1819_7
  simp only [e_1819_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3199)) := by
  have c111156 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111157 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111158 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111159 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111160 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111161 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_8 := arithEq_of_rows h (row := 1819) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_8
  simp only [← c111156, ← c111157, ← c111158] at e_1819_8
  have e_1819_9 := arithEq_of_rows h (row := 1819) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_9
  simp only [← c111159, ← c111160, k2, ← c111161] at e_1819_9
  have hr := e_1819_9
  simp only [e_1819_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3200)) := by
  have c111162 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111163 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111164 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111165 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111166 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111167 := (publicBatchWrapper89_copies3473 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_10 := arithEq_of_rows h (row := 1819) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_10
  simp only [← c111162, ← c111163, ← c111164] at e_1819_10
  have e_1819_11 := arithEq_of_rows h (row := 1819) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_11
  simp only [← c111165, ← c111166, k2, ← c111167] at e_1819_11
  have hr := e_1819_11
  simp only [e_1819_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3201)) := by
  have c111168 := (publicBatchWrapper89_copies3474 a h).1
  have c111169 := (publicBatchWrapper89_copies3474 a h).2.1
  have c111170 := (publicBatchWrapper89_copies3474 a h).2.2.1
  have c111171 := (publicBatchWrapper89_copies3474 a h).2.2.2.1
  have c111172 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.1
  have c111173 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_12 := arithEq_of_rows h (row := 1819) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_12
  simp only [← c111168, ← c111169, ← c111170] at e_1819_12
  have e_1819_13 := arithEq_of_rows h (row := 1819) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_13
  simp only [← c111171, ← c111172, k2, ← c111173] at e_1819_13
  have hr := e_1819_13
  simp only [e_1819_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3370)) := by
  have c111174 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.1
  have c111175 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.1
  have c111176 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.1
  have c111177 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.1
  have c111178 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111179 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_14 := arithEq_of_rows h (row := 1819) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_14
  simp only [← c111174, ← c111175, ← c111176] at e_1819_14
  have e_1819_15 := arithEq_of_rows h (row := 1819) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_15
  simp only [← c111177, ← c111178, k2, ← c111179] at e_1819_15
  have hr := e_1819_15
  simp only [e_1819_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3371)) := by
  have c111180 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111181 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111182 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111183 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111184 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111185 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_16 := arithEq_of_rows h (row := 1819) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_16
  simp only [← c111180, ← c111181, ← c111182] at e_1819_16
  have e_1819_17 := arithEq_of_rows h (row := 1819) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_17
  simp only [← c111183, ← c111184, k2, ← c111185] at e_1819_17
  have hr := e_1819_17
  simp only [e_1819_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1819 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3372)) := by
  have c111186 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111187 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111188 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111189 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111190 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111191 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1819_18 := arithEq_of_rows h (row := 1819) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_18
  simp only [← c111186, ← c111187, ← c111188] at e_1819_18
  have e_1819_19 := arithEq_of_rows h (row := 1819) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_19
  simp only [← c111189, ← c111190, k2, ← c111191] at e_1819_19
  have hr := e_1819_19
  simp only [e_1819_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3373)) := by
  have c111192 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111193 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111194 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111195 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111196 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111197 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_0 := arithEq_of_rows h (row := 1820) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_0
  simp only [← c111192, ← c111193, ← c111194] at e_1820_0
  have e_1820_1 := arithEq_of_rows h (row := 1820) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_1
  simp only [← c111195, ← c111196, k2, ← c111197] at e_1820_1
  have hr := e_1820_1
  simp only [e_1820_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3374)) := by
  have c111198 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111199 := (publicBatchWrapper89_copies3474 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111200 := (publicBatchWrapper89_copies3475 a h).1
  have c111201 := (publicBatchWrapper89_copies3475 a h).2.1
  have c111202 := (publicBatchWrapper89_copies3475 a h).2.2.1
  have c111203 := (publicBatchWrapper89_copies3475 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_2 := arithEq_of_rows h (row := 1820) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_2
  simp only [← c111198, ← c111199, ← c111200] at e_1820_2
  have e_1820_3 := arithEq_of_rows h (row := 1820) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_3
  simp only [← c111201, ← c111202, k2, ← c111203] at e_1820_3
  have hr := e_1820_3
  simp only [e_1820_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3375)) := by
  have c111204 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.1
  have c111205 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.1
  have c111206 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.1
  have c111207 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.1
  have c111208 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.1
  have c111209 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_4 := arithEq_of_rows h (row := 1820) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_4
  simp only [← c111204, ← c111205, ← c111206] at e_1820_4
  have e_1820_5 := arithEq_of_rows h (row := 1820) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_5
  simp only [← c111207, ← c111208, k2, ← c111209] at e_1820_5
  have hr := e_1820_5
  simp only [e_1820_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3376)) := by
  have c111210 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111211 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111212 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111213 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111214 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111215 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_6 := arithEq_of_rows h (row := 1820) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_6
  simp only [← c111210, ← c111211, ← c111212] at e_1820_6
  have e_1820_7 := arithEq_of_rows h (row := 1820) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_7
  simp only [← c111213, ← c111214, k2, ← c111215] at e_1820_7
  have hr := e_1820_7
  simp only [e_1820_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3377)) := by
  have c111216 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111217 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111218 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111219 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111220 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111221 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_8 := arithEq_of_rows h (row := 1820) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_8
  simp only [← c111216, ← c111217, ← c111218] at e_1820_8
  have e_1820_9 := arithEq_of_rows h (row := 1820) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_9
  simp only [← c111219, ← c111220, k2, ← c111221] at e_1820_9
  have hr := e_1820_9
  simp only [e_1820_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3378)) := by
  have c111222 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111223 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111224 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111225 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111226 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111227 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_10 := arithEq_of_rows h (row := 1820) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_10
  simp only [← c111222, ← c111223, ← c111224] at e_1820_10
  have e_1820_11 := arithEq_of_rows h (row := 1820) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_11
  simp only [← c111225, ← c111226, k2, ← c111227] at e_1820_11
  have hr := e_1820_11
  simp only [e_1820_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3379)) := by
  have c111228 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111229 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111230 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111231 := (publicBatchWrapper89_copies3475 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111232 := (publicBatchWrapper89_copies3476 a h).1
  have c111233 := (publicBatchWrapper89_copies3476 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_12 := arithEq_of_rows h (row := 1820) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_12
  simp only [← c111228, ← c111229, ← c111230] at e_1820_12
  have e_1820_13 := arithEq_of_rows h (row := 1820) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_13
  simp only [← c111231, ← c111232, k2, ← c111233] at e_1820_13
  have hr := e_1820_13
  simp only [e_1820_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3380)) := by
  have c111234 := (publicBatchWrapper89_copies3476 a h).2.2.1
  have c111235 := (publicBatchWrapper89_copies3476 a h).2.2.2.1
  have c111236 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.1
  have c111237 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.1
  have c111238 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.1
  have c111239 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_14 := arithEq_of_rows h (row := 1820) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_14
  simp only [← c111234, ← c111235, ← c111236] at e_1820_14
  have e_1820_15 := arithEq_of_rows h (row := 1820) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_15
  simp only [← c111237, ← c111238, k2, ← c111239] at e_1820_15
  have hr := e_1820_15
  simp only [e_1820_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3381)) := by
  have c111240 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.1
  have c111241 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.1
  have c111242 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111243 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111244 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111245 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_16 := arithEq_of_rows h (row := 1820) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_16
  simp only [← c111240, ← c111241, ← c111242] at e_1820_16
  have e_1820_17 := arithEq_of_rows h (row := 1820) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_17
  simp only [← c111243, ← c111244, k2, ← c111245] at e_1820_17
  have hr := e_1820_17
  simp only [e_1820_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1820 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3382)) := by
  have c111246 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111247 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111248 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111249 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111250 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111251 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1820_18 := arithEq_of_rows h (row := 1820) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_18
  simp only [← c111246, ← c111247, ← c111248] at e_1820_18
  have e_1820_19 := arithEq_of_rows h (row := 1820) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_19
  simp only [← c111249, ← c111250, k2, ← c111251] at e_1820_19
  have hr := e_1820_19
  simp only [e_1820_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3383)) := by
  have c111252 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111253 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111254 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111255 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111256 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111257 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_0 := arithEq_of_rows h (row := 1821) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_0
  simp only [← c111252, ← c111253, ← c111254] at e_1821_0
  have e_1821_1 := arithEq_of_rows h (row := 1821) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_1
  simp only [← c111255, ← c111256, k2, ← c111257] at e_1821_1
  have hr := e_1821_1
  simp only [e_1821_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3384)) := by
  have c111258 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111259 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111260 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111261 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111262 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111263 := (publicBatchWrapper89_copies3476 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_2 := arithEq_of_rows h (row := 1821) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_2
  simp only [← c111258, ← c111259, ← c111260] at e_1821_2
  have e_1821_3 := arithEq_of_rows h (row := 1821) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_3
  simp only [← c111261, ← c111262, k2, ← c111263] at e_1821_3
  have hr := e_1821_3
  simp only [e_1821_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3385)) := by
  have c111264 := (publicBatchWrapper89_copies3477 a h).1
  have c111265 := (publicBatchWrapper89_copies3477 a h).2.1
  have c111266 := (publicBatchWrapper89_copies3477 a h).2.2.1
  have c111267 := (publicBatchWrapper89_copies3477 a h).2.2.2.1
  have c111268 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.1
  have c111269 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_4 := arithEq_of_rows h (row := 1821) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_4
  simp only [← c111264, ← c111265, ← c111266] at e_1821_4
  have e_1821_5 := arithEq_of_rows h (row := 1821) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_5
  simp only [← c111267, ← c111268, k2, ← c111269] at e_1821_5
  have hr := e_1821_5
  simp only [e_1821_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3386)) := by
  have c111270 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.1
  have c111271 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.1
  have c111272 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.1
  have c111273 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.1
  have c111274 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111275 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_6 := arithEq_of_rows h (row := 1821) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_6
  simp only [← c111270, ← c111271, ← c111272] at e_1821_6
  have e_1821_7 := arithEq_of_rows h (row := 1821) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_7
  simp only [← c111273, ← c111274, k2, ← c111275] at e_1821_7
  have hr := e_1821_7
  simp only [e_1821_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3387)) := by
  have c111276 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111277 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111278 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111279 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111280 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111281 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_8 := arithEq_of_rows h (row := 1821) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_8
  simp only [← c111276, ← c111277, ← c111278] at e_1821_8
  have e_1821_9 := arithEq_of_rows h (row := 1821) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_9
  simp only [← c111279, ← c111280, k2, ← c111281] at e_1821_9
  have hr := e_1821_9
  simp only [e_1821_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3388)) := by
  have c111282 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111283 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111284 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111285 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111286 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111287 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_10 := arithEq_of_rows h (row := 1821) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_10
  simp only [← c111282, ← c111283, ← c111284] at e_1821_10
  have e_1821_11 := arithEq_of_rows h (row := 1821) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_11
  simp only [← c111285, ← c111286, k2, ← c111287] at e_1821_11
  have hr := e_1821_11
  simp only [e_1821_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3389)) := by
  have c111288 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111289 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111290 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111291 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111292 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111293 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_12 := arithEq_of_rows h (row := 1821) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_12
  simp only [← c111288, ← c111289, ← c111290] at e_1821_12
  have e_1821_13 := arithEq_of_rows h (row := 1821) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_13
  simp only [← c111291, ← c111292, k2, ← c111293] at e_1821_13
  have hr := e_1821_13
  simp only [e_1821_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3390)) := by
  have c111294 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111295 := (publicBatchWrapper89_copies3477 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111296 := (publicBatchWrapper89_copies3478 a h).1
  have c111297 := (publicBatchWrapper89_copies3478 a h).2.1
  have c111298 := (publicBatchWrapper89_copies3478 a h).2.2.1
  have c111299 := (publicBatchWrapper89_copies3478 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_14 := arithEq_of_rows h (row := 1821) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_14
  simp only [← c111294, ← c111295, ← c111296] at e_1821_14
  have e_1821_15 := arithEq_of_rows h (row := 1821) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_15
  simp only [← c111297, ← c111298, k2, ← c111299] at e_1821_15
  have hr := e_1821_15
  simp only [e_1821_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3391)) := by
  have c111300 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.1
  have c111301 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.1
  have c111302 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.1
  have c111303 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.1
  have c111304 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.1
  have c111305 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_16 := arithEq_of_rows h (row := 1821) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_16
  simp only [← c111300, ← c111301, ← c111302] at e_1821_16
  have e_1821_17 := arithEq_of_rows h (row := 1821) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_17
  simp only [← c111303, ← c111304, k2, ← c111305] at e_1821_17
  have hr := e_1821_17
  simp only [e_1821_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1821 79) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3392)) := by
  have c111306 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111307 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111308 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111309 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111310 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111311 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1821_18 := arithEq_of_rows h (row := 1821) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_18
  simp only [← c111306, ← c111307, ← c111308] at e_1821_18
  have e_1821_19 := arithEq_of_rows h (row := 1821) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_19
  simp only [← c111309, ← c111310, k2, ← c111311] at e_1821_19
  have hr := e_1821_19
  simp only [e_1821_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 7) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3393)) := by
  have c111312 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111313 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111314 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111315 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111316 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111317 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_0 := arithEq_of_rows h (row := 1822) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_0
  simp only [← c111312, ← c111313, ← c111314] at e_1822_0
  have e_1822_1 := arithEq_of_rows h (row := 1822) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_1
  simp only [← c111315, ← c111316, k2, ← c111317] at e_1822_1
  have hr := e_1822_1
  simp only [e_1822_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 15) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3394)) := by
  have c111318 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111319 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111320 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111321 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111322 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111323 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_2 := arithEq_of_rows h (row := 1822) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_2
  simp only [← c111318, ← c111319, ← c111320] at e_1822_2
  have e_1822_3 := arithEq_of_rows h (row := 1822) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_3
  simp only [← c111321, ← c111322, k2, ← c111323] at e_1822_3
  have hr := e_1822_3
  simp only [e_1822_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 23) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3395)) := by
  have c111324 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111325 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111326 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111327 := (publicBatchWrapper89_copies3478 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111328 := (publicBatchWrapper89_copies3479 a h).1
  have c111329 := (publicBatchWrapper89_copies3479 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_4 := arithEq_of_rows h (row := 1822) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_4
  simp only [← c111324, ← c111325, ← c111326] at e_1822_4
  have e_1822_5 := arithEq_of_rows h (row := 1822) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_5
  simp only [← c111327, ← c111328, k2, ← c111329] at e_1822_5
  have hr := e_1822_5
  simp only [e_1822_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 31) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3396)) := by
  have c111330 := (publicBatchWrapper89_copies3479 a h).2.2.1
  have c111331 := (publicBatchWrapper89_copies3479 a h).2.2.2.1
  have c111332 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.1
  have c111333 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.1
  have c111334 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.1
  have c111335 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_6 := arithEq_of_rows h (row := 1822) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_6
  simp only [← c111330, ← c111331, ← c111332] at e_1822_6
  have e_1822_7 := arithEq_of_rows h (row := 1822) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_7
  simp only [← c111333, ← c111334, k2, ← c111335] at e_1822_7
  have hr := e_1822_7
  simp only [e_1822_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 39) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3397)) := by
  have c111336 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.1
  have c111337 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.1
  have c111338 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111339 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111340 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111341 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_8 := arithEq_of_rows h (row := 1822) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_8
  simp only [← c111336, ← c111337, ← c111338] at e_1822_8
  have e_1822_9 := arithEq_of_rows h (row := 1822) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_9
  simp only [← c111339, ← c111340, k2, ← c111341] at e_1822_9
  have hr := e_1822_9
  simp only [e_1822_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 47) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3398)) := by
  have c111342 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111343 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111344 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111345 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111346 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111347 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_10 := arithEq_of_rows h (row := 1822) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_10
  simp only [← c111342, ← c111343, ← c111344] at e_1822_10
  have e_1822_11 := arithEq_of_rows h (row := 1822) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_11
  simp only [← c111345, ← c111346, k2, ← c111347] at e_1822_11
  have hr := e_1822_11
  simp only [e_1822_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 55) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3399)) := by
  have c111348 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111349 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111350 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111351 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111352 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111353 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_12 := arithEq_of_rows h (row := 1822) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_12
  simp only [← c111348, ← c111349, ← c111350] at e_1822_12
  have e_1822_13 := arithEq_of_rows h (row := 1822) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_13
  simp only [← c111351, ← c111352, k2, ← c111353] at e_1822_13
  have hr := e_1822_13
  simp only [e_1822_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 63) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3400)) := by
  have c111354 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111355 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111356 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111357 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111358 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111359 := (publicBatchWrapper89_copies3479 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_14 := arithEq_of_rows h (row := 1822) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_14
  simp only [← c111354, ← c111355, ← c111356] at e_1822_14
  have e_1822_15 := arithEq_of_rows h (row := 1822) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_15
  simp only [← c111357, ← c111358, k2, ← c111359] at e_1822_15
  have hr := e_1822_15
  simp only [e_1822_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 71) = bselect (a (.wire 17 43)) (a (.virt 200)) (a (.virt 3401)) := by
  have c111360 := (publicBatchWrapper89_copies3480 a h).1
  have c111361 := (publicBatchWrapper89_copies3480 a h).2.1
  have c111362 := (publicBatchWrapper89_copies3480 a h).2.2.1
  have c111363 := (publicBatchWrapper89_copies3480 a h).2.2.2.1
  have c111364 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.1
  have c111365 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_16 := arithEq_of_rows h (row := 1822) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_16
  simp only [← c111360, ← c111361, ← c111362] at e_1822_16
  have e_1822_17 := arithEq_of_rows h (row := 1822) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_17
  simp only [← c111363, ← c111364, k2, ← c111365] at e_1822_17
  have hr := e_1822_17
  simp only [e_1822_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1822 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3570)) := by
  have c111366 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.1
  have c111367 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.1
  have c111368 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.1
  have c111369 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.1
  have c111370 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111371 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1822_18 := arithEq_of_rows h (row := 1822) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_18
  simp only [← c111366, ← c111367, ← c111368] at e_1822_18
  have e_1822_19 := arithEq_of_rows h (row := 1822) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_19
  simp only [← c111369, ← c111370, k2, ← c111371] at e_1822_19
  have hr := e_1822_19
  simp only [e_1822_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3571)) := by
  have c111372 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111373 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111374 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111375 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111376 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111377 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_0 := arithEq_of_rows h (row := 1823) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_0
  simp only [← c111372, ← c111373, ← c111374] at e_1823_0
  have e_1823_1 := arithEq_of_rows h (row := 1823) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_1
  simp only [← c111375, ← c111376, k2, ← c111377] at e_1823_1
  have hr := e_1823_1
  simp only [e_1823_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3572)) := by
  have c111378 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111379 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111380 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111381 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111382 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111383 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_2 := arithEq_of_rows h (row := 1823) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_2
  simp only [← c111378, ← c111379, ← c111380] at e_1823_2
  have e_1823_3 := arithEq_of_rows h (row := 1823) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_3
  simp only [← c111381, ← c111382, k2, ← c111383] at e_1823_3
  have hr := e_1823_3
  simp only [e_1823_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3573)) := by
  have c111384 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111385 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111386 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111387 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111388 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111389 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_4 := arithEq_of_rows h (row := 1823) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_4
  simp only [← c111384, ← c111385, ← c111386] at e_1823_4
  have e_1823_5 := arithEq_of_rows h (row := 1823) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_5
  simp only [← c111387, ← c111388, k2, ← c111389] at e_1823_5
  have hr := e_1823_5
  simp only [e_1823_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3574)) := by
  have c111390 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111391 := (publicBatchWrapper89_copies3480 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111392 := (publicBatchWrapper89_copies3481 a h).1
  have c111393 := (publicBatchWrapper89_copies3481 a h).2.1
  have c111394 := (publicBatchWrapper89_copies3481 a h).2.2.1
  have c111395 := (publicBatchWrapper89_copies3481 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_6 := arithEq_of_rows h (row := 1823) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_6
  simp only [← c111390, ← c111391, ← c111392] at e_1823_6
  have e_1823_7 := arithEq_of_rows h (row := 1823) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_7
  simp only [← c111393, ← c111394, k2, ← c111395] at e_1823_7
  have hr := e_1823_7
  simp only [e_1823_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3575)) := by
  have c111396 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.1
  have c111397 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.1
  have c111398 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.1
  have c111399 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.1
  have c111400 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.1
  have c111401 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_8 := arithEq_of_rows h (row := 1823) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_8
  simp only [← c111396, ← c111397, ← c111398] at e_1823_8
  have e_1823_9 := arithEq_of_rows h (row := 1823) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_9
  simp only [← c111399, ← c111400, k2, ← c111401] at e_1823_9
  have hr := e_1823_9
  simp only [e_1823_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3576)) := by
  have c111402 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111403 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111404 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111405 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111406 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111407 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_10 := arithEq_of_rows h (row := 1823) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_10
  simp only [← c111402, ← c111403, ← c111404] at e_1823_10
  have e_1823_11 := arithEq_of_rows h (row := 1823) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_11
  simp only [← c111405, ← c111406, k2, ← c111407] at e_1823_11
  have hr := e_1823_11
  simp only [e_1823_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3577)) := by
  have c111408 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111409 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111410 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111411 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111412 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111413 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_12 := arithEq_of_rows h (row := 1823) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_12
  simp only [← c111408, ← c111409, ← c111410] at e_1823_12
  have e_1823_13 := arithEq_of_rows h (row := 1823) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_13
  simp only [← c111411, ← c111412, k2, ← c111413] at e_1823_13
  have hr := e_1823_13
  simp only [e_1823_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3578)) := by
  have c111414 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111415 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111416 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111417 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111418 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111419 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_14 := arithEq_of_rows h (row := 1823) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_14
  simp only [← c111414, ← c111415, ← c111416] at e_1823_14
  have e_1823_15 := arithEq_of_rows h (row := 1823) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_15
  simp only [← c111417, ← c111418, k2, ← c111419] at e_1823_15
  have hr := e_1823_15
  simp only [e_1823_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3579)) := by
  have c111420 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111421 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111422 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111423 := (publicBatchWrapper89_copies3481 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111424 := (publicBatchWrapper89_copies3482 a h).1
  have c111425 := (publicBatchWrapper89_copies3482 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_16 := arithEq_of_rows h (row := 1823) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_16
  simp only [← c111420, ← c111421, ← c111422] at e_1823_16
  have e_1823_17 := arithEq_of_rows h (row := 1823) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_17
  simp only [← c111423, ← c111424, k2, ← c111425] at e_1823_17
  have hr := e_1823_17
  simp only [e_1823_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1823 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3580)) := by
  have c111426 := (publicBatchWrapper89_copies3482 a h).2.2.1
  have c111427 := (publicBatchWrapper89_copies3482 a h).2.2.2.1
  have c111428 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.1
  have c111429 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.1
  have c111430 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.1
  have c111431 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1823_18 := arithEq_of_rows h (row := 1823) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_18
  simp only [← c111426, ← c111427, ← c111428] at e_1823_18
  have e_1823_19 := arithEq_of_rows h (row := 1823) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_19
  simp only [← c111429, ← c111430, k2, ← c111431] at e_1823_19
  have hr := e_1823_19
  simp only [e_1823_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3581)) := by
  have c111432 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.1
  have c111433 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.1
  have c111434 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111435 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111436 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111437 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_0 := arithEq_of_rows h (row := 1824) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_0
  simp only [← c111432, ← c111433, ← c111434] at e_1824_0
  have e_1824_1 := arithEq_of_rows h (row := 1824) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_1
  simp only [← c111435, ← c111436, k2, ← c111437] at e_1824_1
  have hr := e_1824_1
  simp only [e_1824_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3582)) := by
  have c111438 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111439 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111440 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111441 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111442 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111443 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_2 := arithEq_of_rows h (row := 1824) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_2
  simp only [← c111438, ← c111439, ← c111440] at e_1824_2
  have e_1824_3 := arithEq_of_rows h (row := 1824) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_3
  simp only [← c111441, ← c111442, k2, ← c111443] at e_1824_3
  have hr := e_1824_3
  simp only [e_1824_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3583)) := by
  have c111444 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111445 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111446 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111447 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111448 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111449 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_4 := arithEq_of_rows h (row := 1824) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_4
  simp only [← c111444, ← c111445, ← c111446] at e_1824_4
  have e_1824_5 := arithEq_of_rows h (row := 1824) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_5
  simp only [← c111447, ← c111448, k2, ← c111449] at e_1824_5
  have hr := e_1824_5
  simp only [e_1824_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3584)) := by
  have c111450 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111451 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111452 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111453 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111454 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111455 := (publicBatchWrapper89_copies3482 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_6 := arithEq_of_rows h (row := 1824) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_6
  simp only [← c111450, ← c111451, ← c111452] at e_1824_6
  have e_1824_7 := arithEq_of_rows h (row := 1824) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_7
  simp only [← c111453, ← c111454, k2, ← c111455] at e_1824_7
  have hr := e_1824_7
  simp only [e_1824_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3585)) := by
  have c111456 := (publicBatchWrapper89_copies3483 a h).1
  have c111457 := (publicBatchWrapper89_copies3483 a h).2.1
  have c111458 := (publicBatchWrapper89_copies3483 a h).2.2.1
  have c111459 := (publicBatchWrapper89_copies3483 a h).2.2.2.1
  have c111460 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.1
  have c111461 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_8 := arithEq_of_rows h (row := 1824) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_8
  simp only [← c111456, ← c111457, ← c111458] at e_1824_8
  have e_1824_9 := arithEq_of_rows h (row := 1824) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_9
  simp only [← c111459, ← c111460, k2, ← c111461] at e_1824_9
  have hr := e_1824_9
  simp only [e_1824_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3586)) := by
  have c111462 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.1
  have c111463 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.1
  have c111464 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.1
  have c111465 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.1
  have c111466 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111467 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_10 := arithEq_of_rows h (row := 1824) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_10
  simp only [← c111462, ← c111463, ← c111464] at e_1824_10
  have e_1824_11 := arithEq_of_rows h (row := 1824) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_11
  simp only [← c111465, ← c111466, k2, ← c111467] at e_1824_11
  have hr := e_1824_11
  simp only [e_1824_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3587)) := by
  have c111468 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111469 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111470 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111471 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111472 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111473 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_12 := arithEq_of_rows h (row := 1824) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_12
  simp only [← c111468, ← c111469, ← c111470] at e_1824_12
  have e_1824_13 := arithEq_of_rows h (row := 1824) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_13
  simp only [← c111471, ← c111472, k2, ← c111473] at e_1824_13
  have hr := e_1824_13
  simp only [e_1824_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3588)) := by
  have c111474 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111475 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111476 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111477 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111478 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111479 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_14 := arithEq_of_rows h (row := 1824) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_14
  simp only [← c111474, ← c111475, ← c111476] at e_1824_14
  have e_1824_15 := arithEq_of_rows h (row := 1824) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_15
  simp only [← c111477, ← c111478, k2, ← c111479] at e_1824_15
  have hr := e_1824_15
  simp only [e_1824_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3589)) := by
  have c111480 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111481 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111482 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111483 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111484 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111485 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_16 := arithEq_of_rows h (row := 1824) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_16
  simp only [← c111480, ← c111481, ← c111482] at e_1824_16
  have e_1824_17 := arithEq_of_rows h (row := 1824) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_17
  simp only [← c111483, ← c111484, k2, ← c111485] at e_1824_17
  have hr := e_1824_17
  simp only [e_1824_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1824 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3590)) := by
  have c111486 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111487 := (publicBatchWrapper89_copies3483 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111488 := (publicBatchWrapper89_copies3484 a h).1
  have c111489 := (publicBatchWrapper89_copies3484 a h).2.1
  have c111490 := (publicBatchWrapper89_copies3484 a h).2.2.1
  have c111491 := (publicBatchWrapper89_copies3484 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1824_18 := arithEq_of_rows h (row := 1824) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_18
  simp only [← c111486, ← c111487, ← c111488] at e_1824_18
  have e_1824_19 := arithEq_of_rows h (row := 1824) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_19
  simp only [← c111489, ← c111490, k2, ← c111491] at e_1824_19
  have hr := e_1824_19
  simp only [e_1824_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3591)) := by
  have c111492 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.1
  have c111493 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.1
  have c111494 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.1
  have c111495 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.1
  have c111496 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.1
  have c111497 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_0 := arithEq_of_rows h (row := 1825) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_0
  simp only [← c111492, ← c111493, ← c111494] at e_1825_0
  have e_1825_1 := arithEq_of_rows h (row := 1825) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_1
  simp only [← c111495, ← c111496, k2, ← c111497] at e_1825_1
  have hr := e_1825_1
  simp only [e_1825_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 15) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3592)) := by
  have c111498 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111499 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111500 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111501 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111502 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111503 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_2 := arithEq_of_rows h (row := 1825) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_2
  simp only [← c111498, ← c111499, ← c111500] at e_1825_2
  have e_1825_3 := arithEq_of_rows h (row := 1825) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_3
  simp only [← c111501, ← c111502, k2, ← c111503] at e_1825_3
  have hr := e_1825_3
  simp only [e_1825_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 23) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3593)) := by
  have c111504 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111505 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111506 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111507 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111508 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111509 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_4 := arithEq_of_rows h (row := 1825) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_4
  simp only [← c111504, ← c111505, ← c111506] at e_1825_4
  have e_1825_5 := arithEq_of_rows h (row := 1825) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_5
  simp only [← c111507, ← c111508, k2, ← c111509] at e_1825_5
  have hr := e_1825_5
  simp only [e_1825_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 31) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3594)) := by
  have c111510 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111511 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111512 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111513 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111514 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111515 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_6 := arithEq_of_rows h (row := 1825) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_6
  simp only [← c111510, ← c111511, ← c111512] at e_1825_6
  have e_1825_7 := arithEq_of_rows h (row := 1825) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_7
  simp only [← c111513, ← c111514, k2, ← c111515] at e_1825_7
  have hr := e_1825_7
  simp only [e_1825_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 39) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3595)) := by
  have c111516 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111517 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111518 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111519 := (publicBatchWrapper89_copies3484 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111520 := (publicBatchWrapper89_copies3485 a h).1
  have c111521 := (publicBatchWrapper89_copies3485 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_8 := arithEq_of_rows h (row := 1825) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_8
  simp only [← c111516, ← c111517, ← c111518] at e_1825_8
  have e_1825_9 := arithEq_of_rows h (row := 1825) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_9
  simp only [← c111519, ← c111520, k2, ← c111521] at e_1825_9
  have hr := e_1825_9
  simp only [e_1825_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 47) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3596)) := by
  have c111522 := (publicBatchWrapper89_copies3485 a h).2.2.1
  have c111523 := (publicBatchWrapper89_copies3485 a h).2.2.2.1
  have c111524 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.1
  have c111525 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.1
  have c111526 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.1
  have c111527 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_10 := arithEq_of_rows h (row := 1825) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_10
  simp only [← c111522, ← c111523, ← c111524] at e_1825_10
  have e_1825_11 := arithEq_of_rows h (row := 1825) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_11
  simp only [← c111525, ← c111526, k2, ← c111527] at e_1825_11
  have hr := e_1825_11
  simp only [e_1825_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 55) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3597)) := by
  have c111528 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.1
  have c111529 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.1
  have c111530 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111531 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111532 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111533 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_12 := arithEq_of_rows h (row := 1825) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_12
  simp only [← c111528, ← c111529, ← c111530] at e_1825_12
  have e_1825_13 := arithEq_of_rows h (row := 1825) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_13
  simp only [← c111531, ← c111532, k2, ← c111533] at e_1825_13
  have hr := e_1825_13
  simp only [e_1825_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 63) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3598)) := by
  have c111534 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111535 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111536 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111537 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111538 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111539 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_14 := arithEq_of_rows h (row := 1825) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_14
  simp only [← c111534, ← c111535, ← c111536] at e_1825_14
  have e_1825_15 := arithEq_of_rows h (row := 1825) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_15
  simp only [← c111537, ← c111538, k2, ← c111539] at e_1825_15
  have hr := e_1825_15
  simp only [e_1825_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 71) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3599)) := by
  have c111540 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111541 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111542 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111543 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111544 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111545 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_16 := arithEq_of_rows h (row := 1825) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_16
  simp only [← c111540, ← c111541, ← c111542] at e_1825_16
  have e_1825_17 := arithEq_of_rows h (row := 1825) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_17
  simp only [← c111543, ← c111544, k2, ← c111545] at e_1825_17
  have hr := e_1825_17
  simp only [e_1825_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1825 79) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3600)) := by
  have c111546 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111547 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111548 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111549 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111550 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111551 := (publicBatchWrapper89_copies3485 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1825_18 := arithEq_of_rows h (row := 1825) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_18
  simp only [← c111546, ← c111547, ← c111548] at e_1825_18
  have e_1825_19 := arithEq_of_rows h (row := 1825) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_19
  simp only [← c111549, ← c111550, k2, ← c111551] at e_1825_19
  have hr := e_1825_19
  simp only [e_1825_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 7) = bselect (a (.wire 19 7)) (a (.virt 200)) (a (.virt 3601)) := by
  have c111552 := (publicBatchWrapper89_copies3486 a h).1
  have c111553 := (publicBatchWrapper89_copies3486 a h).2.1
  have c111554 := (publicBatchWrapper89_copies3486 a h).2.2.1
  have c111555 := (publicBatchWrapper89_copies3486 a h).2.2.2.1
  have c111556 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.1
  have c111557 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_0 := arithEq_of_rows h (row := 1826) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_0
  simp only [← c111552, ← c111553, ← c111554] at e_1826_0
  have e_1826_1 := arithEq_of_rows h (row := 1826) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_1
  simp only [← c111555, ← c111556, k2, ← c111557] at e_1826_1
  have hr := e_1826_1
  simp only [e_1826_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3770)) := by
  have c111558 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.1
  have c111559 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.1
  have c111560 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.1
  have c111561 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.1
  have c111562 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111563 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_2 := arithEq_of_rows h (row := 1826) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_2
  simp only [← c111558, ← c111559, ← c111560] at e_1826_2
  have e_1826_3 := arithEq_of_rows h (row := 1826) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_3
  simp only [← c111561, ← c111562, k2, ← c111563] at e_1826_3
  have hr := e_1826_3
  simp only [e_1826_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3771)) := by
  have c111564 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111565 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111566 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111567 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111568 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111569 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_4 := arithEq_of_rows h (row := 1826) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_4
  simp only [← c111564, ← c111565, ← c111566] at e_1826_4
  have e_1826_5 := arithEq_of_rows h (row := 1826) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_5
  simp only [← c111567, ← c111568, k2, ← c111569] at e_1826_5
  have hr := e_1826_5
  simp only [e_1826_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3772)) := by
  have c111570 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111571 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111572 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111573 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111574 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111575 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_6 := arithEq_of_rows h (row := 1826) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_6
  simp only [← c111570, ← c111571, ← c111572] at e_1826_6
  have e_1826_7 := arithEq_of_rows h (row := 1826) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_7
  simp only [← c111573, ← c111574, k2, ← c111575] at e_1826_7
  have hr := e_1826_7
  simp only [e_1826_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3773)) := by
  have c111576 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111577 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111578 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111579 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111580 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111581 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_8 := arithEq_of_rows h (row := 1826) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_8
  simp only [← c111576, ← c111577, ← c111578] at e_1826_8
  have e_1826_9 := arithEq_of_rows h (row := 1826) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_9
  simp only [← c111579, ← c111580, k2, ← c111581] at e_1826_9
  have hr := e_1826_9
  simp only [e_1826_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3774)) := by
  have c111582 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111583 := (publicBatchWrapper89_copies3486 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111584 := (publicBatchWrapper89_copies3487 a h).1
  have c111585 := (publicBatchWrapper89_copies3487 a h).2.1
  have c111586 := (publicBatchWrapper89_copies3487 a h).2.2.1
  have c111587 := (publicBatchWrapper89_copies3487 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_10 := arithEq_of_rows h (row := 1826) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_10
  simp only [← c111582, ← c111583, ← c111584] at e_1826_10
  have e_1826_11 := arithEq_of_rows h (row := 1826) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_11
  simp only [← c111585, ← c111586, k2, ← c111587] at e_1826_11
  have hr := e_1826_11
  simp only [e_1826_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3775)) := by
  have c111588 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.1
  have c111589 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.1
  have c111590 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.1
  have c111591 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.1
  have c111592 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.1
  have c111593 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_12 := arithEq_of_rows h (row := 1826) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_12
  simp only [← c111588, ← c111589, ← c111590] at e_1826_12
  have e_1826_13 := arithEq_of_rows h (row := 1826) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_13
  simp only [← c111591, ← c111592, k2, ← c111593] at e_1826_13
  have hr := e_1826_13
  simp only [e_1826_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3776)) := by
  have c111594 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111595 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111596 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111597 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111598 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111599 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_14 := arithEq_of_rows h (row := 1826) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_14
  simp only [← c111594, ← c111595, ← c111596] at e_1826_14
  have e_1826_15 := arithEq_of_rows h (row := 1826) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_15
  simp only [← c111597, ← c111598, k2, ← c111599] at e_1826_15
  have hr := e_1826_15
  simp only [e_1826_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3777)) := by
  have c111600 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111601 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111602 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111603 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111604 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111605 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_16 := arithEq_of_rows h (row := 1826) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_16
  simp only [← c111600, ← c111601, ← c111602] at e_1826_16
  have e_1826_17 := arithEq_of_rows h (row := 1826) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_17
  simp only [← c111603, ← c111604, k2, ← c111605] at e_1826_17
  have hr := e_1826_17
  simp only [e_1826_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1826 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3778)) := by
  have c111606 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111607 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111608 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111609 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111610 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111611 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1826_18 := arithEq_of_rows h (row := 1826) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_18
  simp only [← c111606, ← c111607, ← c111608] at e_1826_18
  have e_1826_19 := arithEq_of_rows h (row := 1826) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_19
  simp only [← c111609, ← c111610, k2, ← c111611] at e_1826_19
  have hr := e_1826_19
  simp only [e_1826_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3779)) := by
  have c111612 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111613 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111614 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111615 := (publicBatchWrapper89_copies3487 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111616 := (publicBatchWrapper89_copies3488 a h).1
  have c111617 := (publicBatchWrapper89_copies3488 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_0 := arithEq_of_rows h (row := 1827) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_0
  simp only [← c111612, ← c111613, ← c111614] at e_1827_0
  have e_1827_1 := arithEq_of_rows h (row := 1827) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_1
  simp only [← c111615, ← c111616, k2, ← c111617] at e_1827_1
  have hr := e_1827_1
  simp only [e_1827_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3780)) := by
  have c111618 := (publicBatchWrapper89_copies3488 a h).2.2.1
  have c111619 := (publicBatchWrapper89_copies3488 a h).2.2.2.1
  have c111620 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.1
  have c111621 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.1
  have c111622 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.1
  have c111623 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_2 := arithEq_of_rows h (row := 1827) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_2
  simp only [← c111618, ← c111619, ← c111620] at e_1827_2
  have e_1827_3 := arithEq_of_rows h (row := 1827) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_3
  simp only [← c111621, ← c111622, k2, ← c111623] at e_1827_3
  have hr := e_1827_3
  simp only [e_1827_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3781)) := by
  have c111624 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.1
  have c111625 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.1
  have c111626 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111627 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111628 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111629 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_4 := arithEq_of_rows h (row := 1827) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_4
  simp only [← c111624, ← c111625, ← c111626] at e_1827_4
  have e_1827_5 := arithEq_of_rows h (row := 1827) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_5
  simp only [← c111627, ← c111628, k2, ← c111629] at e_1827_5
  have hr := e_1827_5
  simp only [e_1827_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3782)) := by
  have c111630 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111631 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111632 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111633 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111634 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111635 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_6 := arithEq_of_rows h (row := 1827) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_6
  simp only [← c111630, ← c111631, ← c111632] at e_1827_6
  have e_1827_7 := arithEq_of_rows h (row := 1827) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_7
  simp only [← c111633, ← c111634, k2, ← c111635] at e_1827_7
  have hr := e_1827_7
  simp only [e_1827_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3783)) := by
  have c111636 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111637 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111638 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111639 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111640 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111641 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_8 := arithEq_of_rows h (row := 1827) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_8
  simp only [← c111636, ← c111637, ← c111638] at e_1827_8
  have e_1827_9 := arithEq_of_rows h (row := 1827) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_9
  simp only [← c111639, ← c111640, k2, ← c111641] at e_1827_9
  have hr := e_1827_9
  simp only [e_1827_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3784)) := by
  have c111642 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111643 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111644 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111645 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111646 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111647 := (publicBatchWrapper89_copies3488 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_10 := arithEq_of_rows h (row := 1827) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_10
  simp only [← c111642, ← c111643, ← c111644] at e_1827_10
  have e_1827_11 := arithEq_of_rows h (row := 1827) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_11
  simp only [← c111645, ← c111646, k2, ← c111647] at e_1827_11
  have hr := e_1827_11
  simp only [e_1827_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3785)) := by
  have c111648 := (publicBatchWrapper89_copies3489 a h).1
  have c111649 := (publicBatchWrapper89_copies3489 a h).2.1
  have c111650 := (publicBatchWrapper89_copies3489 a h).2.2.1
  have c111651 := (publicBatchWrapper89_copies3489 a h).2.2.2.1
  have c111652 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.1
  have c111653 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_12 := arithEq_of_rows h (row := 1827) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_12
  simp only [← c111648, ← c111649, ← c111650] at e_1827_12
  have e_1827_13 := arithEq_of_rows h (row := 1827) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_13
  simp only [← c111651, ← c111652, k2, ← c111653] at e_1827_13
  have hr := e_1827_13
  simp only [e_1827_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3786)) := by
  have c111654 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.1
  have c111655 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.1
  have c111656 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.1
  have c111657 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.1
  have c111658 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111659 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_14 := arithEq_of_rows h (row := 1827) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_14
  simp only [← c111654, ← c111655, ← c111656] at e_1827_14
  have e_1827_15 := arithEq_of_rows h (row := 1827) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_15
  simp only [← c111657, ← c111658, k2, ← c111659] at e_1827_15
  have hr := e_1827_15
  simp only [e_1827_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3787)) := by
  have c111660 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111661 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111662 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111663 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111664 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111665 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_16 := arithEq_of_rows h (row := 1827) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_16
  simp only [← c111660, ← c111661, ← c111662] at e_1827_16
  have e_1827_17 := arithEq_of_rows h (row := 1827) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_17
  simp only [← c111663, ← c111664, k2, ← c111665] at e_1827_17
  have hr := e_1827_17
  simp only [e_1827_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1827 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3788)) := by
  have c111666 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111667 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111668 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111669 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111670 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111671 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1827_18 := arithEq_of_rows h (row := 1827) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_18
  simp only [← c111666, ← c111667, ← c111668] at e_1827_18
  have e_1827_19 := arithEq_of_rows h (row := 1827) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_19
  simp only [← c111669, ← c111670, k2, ← c111671] at e_1827_19
  have hr := e_1827_19
  simp only [e_1827_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3789)) := by
  have c111672 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111673 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111674 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111675 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111676 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111677 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_0 := arithEq_of_rows h (row := 1828) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_0
  simp only [← c111672, ← c111673, ← c111674] at e_1828_0
  have e_1828_1 := arithEq_of_rows h (row := 1828) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_1
  simp only [← c111675, ← c111676, k2, ← c111677] at e_1828_1
  have hr := e_1828_1
  simp only [e_1828_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3790)) := by
  have c111678 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111679 := (publicBatchWrapper89_copies3489 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111680 := (publicBatchWrapper89_copies3490 a h).1
  have c111681 := (publicBatchWrapper89_copies3490 a h).2.1
  have c111682 := (publicBatchWrapper89_copies3490 a h).2.2.1
  have c111683 := (publicBatchWrapper89_copies3490 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_2 := arithEq_of_rows h (row := 1828) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_2
  simp only [← c111678, ← c111679, ← c111680] at e_1828_2
  have e_1828_3 := arithEq_of_rows h (row := 1828) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_3
  simp only [← c111681, ← c111682, k2, ← c111683] at e_1828_3
  have hr := e_1828_3
  simp only [e_1828_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3791)) := by
  have c111684 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.1
  have c111685 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.1
  have c111686 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.1
  have c111687 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.1
  have c111688 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.1
  have c111689 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_4 := arithEq_of_rows h (row := 1828) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_4
  simp only [← c111684, ← c111685, ← c111686] at e_1828_4
  have e_1828_5 := arithEq_of_rows h (row := 1828) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_5
  simp only [← c111687, ← c111688, k2, ← c111689] at e_1828_5
  have hr := e_1828_5
  simp only [e_1828_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3792)) := by
  have c111690 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111691 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111692 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111693 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111694 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111695 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_6 := arithEq_of_rows h (row := 1828) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_6
  simp only [← c111690, ← c111691, ← c111692] at e_1828_6
  have e_1828_7 := arithEq_of_rows h (row := 1828) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_7
  simp only [← c111693, ← c111694, k2, ← c111695] at e_1828_7
  have hr := e_1828_7
  simp only [e_1828_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3793)) := by
  have c111696 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111697 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111698 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111699 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111700 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111701 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_8 := arithEq_of_rows h (row := 1828) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_8
  simp only [← c111696, ← c111697, ← c111698] at e_1828_8
  have e_1828_9 := arithEq_of_rows h (row := 1828) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_9
  simp only [← c111699, ← c111700, k2, ← c111701] at e_1828_9
  have hr := e_1828_9
  simp only [e_1828_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3794)) := by
  have c111702 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111703 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111704 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111705 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111706 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111707 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_10 := arithEq_of_rows h (row := 1828) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_10
  simp only [← c111702, ← c111703, ← c111704] at e_1828_10
  have e_1828_11 := arithEq_of_rows h (row := 1828) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_11
  simp only [← c111705, ← c111706, k2, ← c111707] at e_1828_11
  have hr := e_1828_11
  simp only [e_1828_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3795)) := by
  have c111708 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111709 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111710 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111711 := (publicBatchWrapper89_copies3490 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111712 := (publicBatchWrapper89_copies3491 a h).1
  have c111713 := (publicBatchWrapper89_copies3491 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_12 := arithEq_of_rows h (row := 1828) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_12
  simp only [← c111708, ← c111709, ← c111710] at e_1828_12
  have e_1828_13 := arithEq_of_rows h (row := 1828) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_13
  simp only [← c111711, ← c111712, k2, ← c111713] at e_1828_13
  have hr := e_1828_13
  simp only [e_1828_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3796)) := by
  have c111714 := (publicBatchWrapper89_copies3491 a h).2.2.1
  have c111715 := (publicBatchWrapper89_copies3491 a h).2.2.2.1
  have c111716 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.1
  have c111717 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.1
  have c111718 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.1
  have c111719 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_14 := arithEq_of_rows h (row := 1828) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_14
  simp only [← c111714, ← c111715, ← c111716] at e_1828_14
  have e_1828_15 := arithEq_of_rows h (row := 1828) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_15
  simp only [← c111717, ← c111718, k2, ← c111719] at e_1828_15
  have hr := e_1828_15
  simp only [e_1828_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3797)) := by
  have c111720 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.1
  have c111721 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.1
  have c111722 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111723 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111724 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111725 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_16 := arithEq_of_rows h (row := 1828) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_16
  simp only [← c111720, ← c111721, ← c111722] at e_1828_16
  have e_1828_17 := arithEq_of_rows h (row := 1828) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_17
  simp only [← c111723, ← c111724, k2, ← c111725] at e_1828_17
  have hr := e_1828_17
  simp only [e_1828_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1828 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3798)) := by
  have c111726 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111727 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111728 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111729 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111730 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111731 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1828_18 := arithEq_of_rows h (row := 1828) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_18
  simp only [← c111726, ← c111727, ← c111728] at e_1828_18
  have e_1828_19 := arithEq_of_rows h (row := 1828) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_19
  simp only [← c111729, ← c111730, k2, ← c111731] at e_1828_19
  have hr := e_1828_19
  simp only [e_1828_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3799)) := by
  have c111732 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111733 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111734 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111735 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111736 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111737 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_0 := arithEq_of_rows h (row := 1829) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_0
  simp only [← c111732, ← c111733, ← c111734] at e_1829_0
  have e_1829_1 := arithEq_of_rows h (row := 1829) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_1
  simp only [← c111735, ← c111736, k2, ← c111737] at e_1829_1
  have hr := e_1829_1
  simp only [e_1829_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3800)) := by
  have c111738 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111739 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111740 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111741 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111742 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111743 := (publicBatchWrapper89_copies3491 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_2 := arithEq_of_rows h (row := 1829) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_2
  simp only [← c111738, ← c111739, ← c111740] at e_1829_2
  have e_1829_3 := arithEq_of_rows h (row := 1829) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_3
  simp only [← c111741, ← c111742, k2, ← c111743] at e_1829_3
  have hr := e_1829_3
  simp only [e_1829_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3801)) := by
  have c111744 := (publicBatchWrapper89_copies3492 a h).1
  have c111745 := (publicBatchWrapper89_copies3492 a h).2.1
  have c111746 := (publicBatchWrapper89_copies3492 a h).2.2.1
  have c111747 := (publicBatchWrapper89_copies3492 a h).2.2.2.1
  have c111748 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.1
  have c111749 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_4 := arithEq_of_rows h (row := 1829) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_4
  simp only [← c111744, ← c111745, ← c111746] at e_1829_4
  have e_1829_5 := arithEq_of_rows h (row := 1829) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_5
  simp only [← c111747, ← c111748, k2, ← c111749] at e_1829_5
  have hr := e_1829_5
  simp only [e_1829_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3970)) := by
  have c111750 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.1
  have c111751 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.1
  have c111752 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.1
  have c111753 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.1
  have c111754 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111755 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_6 := arithEq_of_rows h (row := 1829) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_6
  simp only [← c111750, ← c111751, ← c111752] at e_1829_6
  have e_1829_7 := arithEq_of_rows h (row := 1829) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_7
  simp only [← c111753, ← c111754, k2, ← c111755] at e_1829_7
  have hr := e_1829_7
  simp only [e_1829_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3971)) := by
  have c111756 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111757 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111758 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111759 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111760 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111761 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_8 := arithEq_of_rows h (row := 1829) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_8
  simp only [← c111756, ← c111757, ← c111758] at e_1829_8
  have e_1829_9 := arithEq_of_rows h (row := 1829) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_9
  simp only [← c111759, ← c111760, k2, ← c111761] at e_1829_9
  have hr := e_1829_9
  simp only [e_1829_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3972)) := by
  have c111762 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111763 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111764 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111765 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111766 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111767 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_10 := arithEq_of_rows h (row := 1829) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_10
  simp only [← c111762, ← c111763, ← c111764] at e_1829_10
  have e_1829_11 := arithEq_of_rows h (row := 1829) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_11
  simp only [← c111765, ← c111766, k2, ← c111767] at e_1829_11
  have hr := e_1829_11
  simp only [e_1829_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3973)) := by
  have c111768 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111769 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111770 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111771 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111772 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111773 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_12 := arithEq_of_rows h (row := 1829) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_12
  simp only [← c111768, ← c111769, ← c111770] at e_1829_12
  have e_1829_13 := arithEq_of_rows h (row := 1829) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_13
  simp only [← c111771, ← c111772, k2, ← c111773] at e_1829_13
  have hr := e_1829_13
  simp only [e_1829_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3974)) := by
  have c111774 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111775 := (publicBatchWrapper89_copies3492 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111776 := (publicBatchWrapper89_copies3493 a h).1
  have c111777 := (publicBatchWrapper89_copies3493 a h).2.1
  have c111778 := (publicBatchWrapper89_copies3493 a h).2.2.1
  have c111779 := (publicBatchWrapper89_copies3493 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_14 := arithEq_of_rows h (row := 1829) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_14
  simp only [← c111774, ← c111775, ← c111776] at e_1829_14
  have e_1829_15 := arithEq_of_rows h (row := 1829) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_15
  simp only [← c111777, ← c111778, k2, ← c111779] at e_1829_15
  have hr := e_1829_15
  simp only [e_1829_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3975)) := by
  have c111780 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.1
  have c111781 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.1
  have c111782 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.1
  have c111783 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.1
  have c111784 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.1
  have c111785 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_16 := arithEq_of_rows h (row := 1829) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_16
  simp only [← c111780, ← c111781, ← c111782] at e_1829_16
  have e_1829_17 := arithEq_of_rows h (row := 1829) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_17
  simp only [← c111783, ← c111784, k2, ← c111785] at e_1829_17
  have hr := e_1829_17
  simp only [e_1829_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1829 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3976)) := by
  have c111786 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111787 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111788 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111789 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111790 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111791 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1829_18 := arithEq_of_rows h (row := 1829) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_18
  simp only [← c111786, ← c111787, ← c111788] at e_1829_18
  have e_1829_19 := arithEq_of_rows h (row := 1829) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_19
  simp only [← c111789, ← c111790, k2, ← c111791] at e_1829_19
  have hr := e_1829_19
  simp only [e_1829_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3977)) := by
  have c111792 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111793 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111794 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111795 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111796 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111797 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_0 := arithEq_of_rows h (row := 1830) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_0
  simp only [← c111792, ← c111793, ← c111794] at e_1830_0
  have e_1830_1 := arithEq_of_rows h (row := 1830) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_1
  simp only [← c111795, ← c111796, k2, ← c111797] at e_1830_1
  have hr := e_1830_1
  simp only [e_1830_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3978)) := by
  have c111798 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111799 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111800 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111801 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111802 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111803 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_2 := arithEq_of_rows h (row := 1830) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_2
  simp only [← c111798, ← c111799, ← c111800] at e_1830_2
  have e_1830_3 := arithEq_of_rows h (row := 1830) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_3
  simp only [← c111801, ← c111802, k2, ← c111803] at e_1830_3
  have hr := e_1830_3
  simp only [e_1830_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3979)) := by
  have c111804 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111805 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111806 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111807 := (publicBatchWrapper89_copies3493 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111808 := (publicBatchWrapper89_copies3494 a h).1
  have c111809 := (publicBatchWrapper89_copies3494 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_4 := arithEq_of_rows h (row := 1830) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_4
  simp only [← c111804, ← c111805, ← c111806] at e_1830_4
  have e_1830_5 := arithEq_of_rows h (row := 1830) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_5
  simp only [← c111807, ← c111808, k2, ← c111809] at e_1830_5
  have hr := e_1830_5
  simp only [e_1830_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3980)) := by
  have c111810 := (publicBatchWrapper89_copies3494 a h).2.2.1
  have c111811 := (publicBatchWrapper89_copies3494 a h).2.2.2.1
  have c111812 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.1
  have c111813 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.1
  have c111814 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.1
  have c111815 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_6 := arithEq_of_rows h (row := 1830) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_6
  simp only [← c111810, ← c111811, ← c111812] at e_1830_6
  have e_1830_7 := arithEq_of_rows h (row := 1830) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_7
  simp only [← c111813, ← c111814, k2, ← c111815] at e_1830_7
  have hr := e_1830_7
  simp only [e_1830_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3981)) := by
  have c111816 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.1
  have c111817 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.1
  have c111818 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111819 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111820 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111821 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_8 := arithEq_of_rows h (row := 1830) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_8
  simp only [← c111816, ← c111817, ← c111818] at e_1830_8
  have e_1830_9 := arithEq_of_rows h (row := 1830) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_9
  simp only [← c111819, ← c111820, k2, ← c111821] at e_1830_9
  have hr := e_1830_9
  simp only [e_1830_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3982)) := by
  have c111822 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111823 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111824 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111825 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111826 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111827 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_10 := arithEq_of_rows h (row := 1830) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_10
  simp only [← c111822, ← c111823, ← c111824] at e_1830_10
  have e_1830_11 := arithEq_of_rows h (row := 1830) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_11
  simp only [← c111825, ← c111826, k2, ← c111827] at e_1830_11
  have hr := e_1830_11
  simp only [e_1830_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3983)) := by
  have c111828 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111829 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111830 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111831 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111832 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111833 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_12 := arithEq_of_rows h (row := 1830) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_12
  simp only [← c111828, ← c111829, ← c111830] at e_1830_12
  have e_1830_13 := arithEq_of_rows h (row := 1830) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_13
  simp only [← c111831, ← c111832, k2, ← c111833] at e_1830_13
  have hr := e_1830_13
  simp only [e_1830_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3984)) := by
  have c111834 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111835 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111836 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111837 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111838 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111839 := (publicBatchWrapper89_copies3494 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_14 := arithEq_of_rows h (row := 1830) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_14
  simp only [← c111834, ← c111835, ← c111836] at e_1830_14
  have e_1830_15 := arithEq_of_rows h (row := 1830) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_15
  simp only [← c111837, ← c111838, k2, ← c111839] at e_1830_15
  have hr := e_1830_15
  simp only [e_1830_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3985)) := by
  have c111840 := (publicBatchWrapper89_copies3495 a h).1
  have c111841 := (publicBatchWrapper89_copies3495 a h).2.1
  have c111842 := (publicBatchWrapper89_copies3495 a h).2.2.1
  have c111843 := (publicBatchWrapper89_copies3495 a h).2.2.2.1
  have c111844 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.1
  have c111845 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_16 := arithEq_of_rows h (row := 1830) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_16
  simp only [← c111840, ← c111841, ← c111842] at e_1830_16
  have e_1830_17 := arithEq_of_rows h (row := 1830) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_17
  simp only [← c111843, ← c111844, k2, ← c111845] at e_1830_17
  have hr := e_1830_17
  simp only [e_1830_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1830 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3986)) := by
  have c111846 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.1
  have c111847 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.1
  have c111848 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.1
  have c111849 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.1
  have c111850 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111851 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1830_18 := arithEq_of_rows h (row := 1830) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_18
  simp only [← c111846, ← c111847, ← c111848] at e_1830_18
  have e_1830_19 := arithEq_of_rows h (row := 1830) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_19
  simp only [← c111849, ← c111850, k2, ← c111851] at e_1830_19
  have hr := e_1830_19
  simp only [e_1830_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3987)) := by
  have c111852 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111853 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111854 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111855 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111856 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111857 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_0 := arithEq_of_rows h (row := 1831) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_0
  simp only [← c111852, ← c111853, ← c111854] at e_1831_0
  have e_1831_1 := arithEq_of_rows h (row := 1831) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_1
  simp only [← c111855, ← c111856, k2, ← c111857] at e_1831_1
  have hr := e_1831_1
  simp only [e_1831_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3988)) := by
  have c111858 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111859 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111860 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111861 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111862 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111863 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_2 := arithEq_of_rows h (row := 1831) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_2
  simp only [← c111858, ← c111859, ← c111860] at e_1831_2
  have e_1831_3 := arithEq_of_rows h (row := 1831) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_3
  simp only [← c111861, ← c111862, k2, ← c111863] at e_1831_3
  have hr := e_1831_3
  simp only [e_1831_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3989)) := by
  have c111864 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111865 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111866 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111867 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111868 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111869 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_4 := arithEq_of_rows h (row := 1831) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_4
  simp only [← c111864, ← c111865, ← c111866] at e_1831_4
  have e_1831_5 := arithEq_of_rows h (row := 1831) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_5
  simp only [← c111867, ← c111868, k2, ← c111869] at e_1831_5
  have hr := e_1831_5
  simp only [e_1831_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3990)) := by
  have c111870 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111871 := (publicBatchWrapper89_copies3495 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111872 := (publicBatchWrapper89_copies3496 a h).1
  have c111873 := (publicBatchWrapper89_copies3496 a h).2.1
  have c111874 := (publicBatchWrapper89_copies3496 a h).2.2.1
  have c111875 := (publicBatchWrapper89_copies3496 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_6 := arithEq_of_rows h (row := 1831) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_6
  simp only [← c111870, ← c111871, ← c111872] at e_1831_6
  have e_1831_7 := arithEq_of_rows h (row := 1831) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_7
  simp only [← c111873, ← c111874, k2, ← c111875] at e_1831_7
  have hr := e_1831_7
  simp only [e_1831_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3991)) := by
  have c111876 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.1
  have c111877 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.1
  have c111878 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.1
  have c111879 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.1
  have c111880 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.1
  have c111881 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_8 := arithEq_of_rows h (row := 1831) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_8
  simp only [← c111876, ← c111877, ← c111878] at e_1831_8
  have e_1831_9 := arithEq_of_rows h (row := 1831) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_9
  simp only [← c111879, ← c111880, k2, ← c111881] at e_1831_9
  have hr := e_1831_9
  simp only [e_1831_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3992)) := by
  have c111882 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111883 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111884 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111885 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111886 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111887 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_10 := arithEq_of_rows h (row := 1831) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_10
  simp only [← c111882, ← c111883, ← c111884] at e_1831_10
  have e_1831_11 := arithEq_of_rows h (row := 1831) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_11
  simp only [← c111885, ← c111886, k2, ← c111887] at e_1831_11
  have hr := e_1831_11
  simp only [e_1831_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3993)) := by
  have c111888 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111889 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111890 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111891 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111892 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111893 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_12 := arithEq_of_rows h (row := 1831) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_12
  simp only [← c111888, ← c111889, ← c111890] at e_1831_12
  have e_1831_13 := arithEq_of_rows h (row := 1831) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_13
  simp only [← c111891, ← c111892, k2, ← c111893] at e_1831_13
  have hr := e_1831_13
  simp only [e_1831_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3994)) := by
  have c111894 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111895 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111896 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111897 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111898 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111899 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_14 := arithEq_of_rows h (row := 1831) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_14
  simp only [← c111894, ← c111895, ← c111896] at e_1831_14
  have e_1831_15 := arithEq_of_rows h (row := 1831) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_15
  simp only [← c111897, ← c111898, k2, ← c111899] at e_1831_15
  have hr := e_1831_15
  simp only [e_1831_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3995)) := by
  have c111900 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111901 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111902 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111903 := (publicBatchWrapper89_copies3496 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111904 := (publicBatchWrapper89_copies3497 a h).1
  have c111905 := (publicBatchWrapper89_copies3497 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_16 := arithEq_of_rows h (row := 1831) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_16
  simp only [← c111900, ← c111901, ← c111902] at e_1831_16
  have e_1831_17 := arithEq_of_rows h (row := 1831) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_17
  simp only [← c111903, ← c111904, k2, ← c111905] at e_1831_17
  have hr := e_1831_17
  simp only [e_1831_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1831 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3996)) := by
  have c111906 := (publicBatchWrapper89_copies3497 a h).2.2.1
  have c111907 := (publicBatchWrapper89_copies3497 a h).2.2.2.1
  have c111908 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.1
  have c111909 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.1
  have c111910 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.1
  have c111911 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1831_18 := arithEq_of_rows h (row := 1831) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_18
  simp only [← c111906, ← c111907, ← c111908] at e_1831_18
  have e_1831_19 := arithEq_of_rows h (row := 1831) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_19
  simp only [← c111909, ← c111910, k2, ← c111911] at e_1831_19
  have hr := e_1831_19
  simp only [e_1831_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3997)) := by
  have c111912 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.1
  have c111913 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.1
  have c111914 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111915 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111916 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111917 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_0 := arithEq_of_rows h (row := 1832) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_0
  simp only [← c111912, ← c111913, ← c111914] at e_1832_0
  have e_1832_1 := arithEq_of_rows h (row := 1832) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_1
  simp only [← c111915, ← c111916, k2, ← c111917] at e_1832_1
  have hr := e_1832_1
  simp only [e_1832_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3998)) := by
  have c111918 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111919 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111920 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111921 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111922 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111923 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_2 := arithEq_of_rows h (row := 1832) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_2
  simp only [← c111918, ← c111919, ← c111920] at e_1832_2
  have e_1832_3 := arithEq_of_rows h (row := 1832) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_3
  simp only [← c111921, ← c111922, k2, ← c111923] at e_1832_3
  have hr := e_1832_3
  simp only [e_1832_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3999)) := by
  have c111924 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111925 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111926 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111927 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111928 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111929 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_4 := arithEq_of_rows h (row := 1832) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_4
  simp only [← c111924, ← c111925, ← c111926] at e_1832_4
  have e_1832_5 := arithEq_of_rows h (row := 1832) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_5
  simp only [← c111927, ← c111928, k2, ← c111929] at e_1832_5
  have hr := e_1832_5
  simp only [e_1832_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 4000)) := by
  have c111930 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111931 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111932 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111933 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111934 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111935 := (publicBatchWrapper89_copies3497 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_6 := arithEq_of_rows h (row := 1832) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_6
  simp only [← c111930, ← c111931, ← c111932] at e_1832_6
  have e_1832_7 := arithEq_of_rows h (row := 1832) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_7
  simp only [← c111933, ← c111934, k2, ← c111935] at e_1832_7
  have hr := e_1832_7
  simp only [e_1832_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 4001)) := by
  have c111936 := (publicBatchWrapper89_copies3498 a h).1
  have c111937 := (publicBatchWrapper89_copies3498 a h).2.1
  have c111938 := (publicBatchWrapper89_copies3498 a h).2.2.1
  have c111939 := (publicBatchWrapper89_copies3498 a h).2.2.2.1
  have c111940 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.1
  have c111941 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_8 := arithEq_of_rows h (row := 1832) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_8
  simp only [← c111936, ← c111937, ← c111938] at e_1832_8
  have e_1832_9 := arithEq_of_rows h (row := 1832) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_9
  simp only [← c111939, ← c111940, k2, ← c111941] at e_1832_9
  have hr := e_1832_9
  simp only [e_1832_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4170)) := by
  have c111942 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.1
  have c111943 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.1
  have c111944 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.1
  have c111945 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.1
  have c111946 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111947 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_10 := arithEq_of_rows h (row := 1832) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_10
  simp only [← c111942, ← c111943, ← c111944] at e_1832_10
  have e_1832_11 := arithEq_of_rows h (row := 1832) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_11
  simp only [← c111945, ← c111946, k2, ← c111947] at e_1832_11
  have hr := e_1832_11
  simp only [e_1832_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4171)) := by
  have c111948 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111949 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111950 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111951 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111952 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111953 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_12 := arithEq_of_rows h (row := 1832) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_12
  simp only [← c111948, ← c111949, ← c111950] at e_1832_12
  have e_1832_13 := arithEq_of_rows h (row := 1832) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_13
  simp only [← c111951, ← c111952, k2, ← c111953] at e_1832_13
  have hr := e_1832_13
  simp only [e_1832_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4172)) := by
  have c111954 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111955 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111956 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111957 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111958 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111959 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_14 := arithEq_of_rows h (row := 1832) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_14
  simp only [← c111954, ← c111955, ← c111956] at e_1832_14
  have e_1832_15 := arithEq_of_rows h (row := 1832) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_15
  simp only [← c111957, ← c111958, k2, ← c111959] at e_1832_15
  have hr := e_1832_15
  simp only [e_1832_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4173)) := by
  have c111960 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111961 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111962 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111963 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111964 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111965 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_16 := arithEq_of_rows h (row := 1832) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_16
  simp only [← c111960, ← c111961, ← c111962] at e_1832_16
  have e_1832_17 := arithEq_of_rows h (row := 1832) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_17
  simp only [← c111963, ← c111964, k2, ← c111965] at e_1832_17
  have hr := e_1832_17
  simp only [e_1832_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1832 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4174)) := by
  have c111966 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111967 := (publicBatchWrapper89_copies3498 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111968 := (publicBatchWrapper89_copies3499 a h).1
  have c111969 := (publicBatchWrapper89_copies3499 a h).2.1
  have c111970 := (publicBatchWrapper89_copies3499 a h).2.2.1
  have c111971 := (publicBatchWrapper89_copies3499 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1832_18 := arithEq_of_rows h (row := 1832) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_18
  simp only [← c111966, ← c111967, ← c111968] at e_1832_18
  have e_1832_19 := arithEq_of_rows h (row := 1832) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_19
  simp only [← c111969, ← c111970, k2, ← c111971] at e_1832_19
  have hr := e_1832_19
  simp only [e_1832_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4175)) := by
  have c111972 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.1
  have c111973 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.1
  have c111974 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.1
  have c111975 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.1
  have c111976 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.1
  have c111977 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_0 := arithEq_of_rows h (row := 1833) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_0
  simp only [← c111972, ← c111973, ← c111974] at e_1833_0
  have e_1833_1 := arithEq_of_rows h (row := 1833) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_1
  simp only [← c111975, ← c111976, k2, ← c111977] at e_1833_1
  have hr := e_1833_1
  simp only [e_1833_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4176)) := by
  have c111978 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111979 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111980 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111981 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111982 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111983 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_2 := arithEq_of_rows h (row := 1833) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_2
  simp only [← c111978, ← c111979, ← c111980] at e_1833_2
  have e_1833_3 := arithEq_of_rows h (row := 1833) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_3
  simp only [← c111981, ← c111982, k2, ← c111983] at e_1833_3
  have hr := e_1833_3
  simp only [e_1833_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4177)) := by
  have c111984 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111985 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111986 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111987 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111988 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111989 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_4 := arithEq_of_rows h (row := 1833) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_4
  simp only [← c111984, ← c111985, ← c111986] at e_1833_4
  have e_1833_5 := arithEq_of_rows h (row := 1833) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_5
  simp only [← c111987, ← c111988, k2, ← c111989] at e_1833_5
  have hr := e_1833_5
  simp only [e_1833_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4178)) := by
  have c111990 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111991 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111992 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111993 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111994 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111995 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_6 := arithEq_of_rows h (row := 1833) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_6
  simp only [← c111990, ← c111991, ← c111992] at e_1833_6
  have e_1833_7 := arithEq_of_rows h (row := 1833) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_7
  simp only [← c111993, ← c111994, k2, ← c111995] at e_1833_7
  have hr := e_1833_7
  simp only [e_1833_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4179)) := by
  have c111996 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111997 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111998 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111999 := (publicBatchWrapper89_copies3499 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112000 := (publicBatchWrapper89_copies3500 a h).1
  have c112001 := (publicBatchWrapper89_copies3500 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_8 := arithEq_of_rows h (row := 1833) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_8
  simp only [← c111996, ← c111997, ← c111998] at e_1833_8
  have e_1833_9 := arithEq_of_rows h (row := 1833) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_9
  simp only [← c111999, ← c112000, k2, ← c112001] at e_1833_9
  have hr := e_1833_9
  simp only [e_1833_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4180)) := by
  have c112002 := (publicBatchWrapper89_copies3500 a h).2.2.1
  have c112003 := (publicBatchWrapper89_copies3500 a h).2.2.2.1
  have c112004 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.1
  have c112005 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.1
  have c112006 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.1
  have c112007 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_10 := arithEq_of_rows h (row := 1833) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_10
  simp only [← c112002, ← c112003, ← c112004] at e_1833_10
  have e_1833_11 := arithEq_of_rows h (row := 1833) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_11
  simp only [← c112005, ← c112006, k2, ← c112007] at e_1833_11
  have hr := e_1833_11
  simp only [e_1833_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4181)) := by
  have c112008 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.1
  have c112009 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.1
  have c112010 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112011 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112012 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112013 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_12 := arithEq_of_rows h (row := 1833) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_12
  simp only [← c112008, ← c112009, ← c112010] at e_1833_12
  have e_1833_13 := arithEq_of_rows h (row := 1833) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_13
  simp only [← c112011, ← c112012, k2, ← c112013] at e_1833_13
  have hr := e_1833_13
  simp only [e_1833_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4182)) := by
  have c112014 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112015 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112016 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112017 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112018 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112019 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_14 := arithEq_of_rows h (row := 1833) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_14
  simp only [← c112014, ← c112015, ← c112016] at e_1833_14
  have e_1833_15 := arithEq_of_rows h (row := 1833) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_15
  simp only [← c112017, ← c112018, k2, ← c112019] at e_1833_15
  have hr := e_1833_15
  simp only [e_1833_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4183)) := by
  have c112020 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112021 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112022 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112023 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112024 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112025 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_16 := arithEq_of_rows h (row := 1833) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_16
  simp only [← c112020, ← c112021, ← c112022] at e_1833_16
  have e_1833_17 := arithEq_of_rows h (row := 1833) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_17
  simp only [← c112023, ← c112024, k2, ← c112025] at e_1833_17
  have hr := e_1833_17
  simp only [e_1833_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1833 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4184)) := by
  have c112026 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112027 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112028 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112029 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112030 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112031 := (publicBatchWrapper89_copies3500 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1833_18 := arithEq_of_rows h (row := 1833) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_18
  simp only [← c112026, ← c112027, ← c112028] at e_1833_18
  have e_1833_19 := arithEq_of_rows h (row := 1833) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_19
  simp only [← c112029, ← c112030, k2, ← c112031] at e_1833_19
  have hr := e_1833_19
  simp only [e_1833_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4185)) := by
  have c112032 := (publicBatchWrapper89_copies3501 a h).1
  have c112033 := (publicBatchWrapper89_copies3501 a h).2.1
  have c112034 := (publicBatchWrapper89_copies3501 a h).2.2.1
  have c112035 := (publicBatchWrapper89_copies3501 a h).2.2.2.1
  have c112036 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.1
  have c112037 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_0 := arithEq_of_rows h (row := 1834) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_0
  simp only [← c112032, ← c112033, ← c112034] at e_1834_0
  have e_1834_1 := arithEq_of_rows h (row := 1834) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_1
  simp only [← c112035, ← c112036, k2, ← c112037] at e_1834_1
  have hr := e_1834_1
  simp only [e_1834_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4186)) := by
  have c112038 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.1
  have c112039 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.1
  have c112040 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.1
  have c112041 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.1
  have c112042 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112043 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_2 := arithEq_of_rows h (row := 1834) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_2
  simp only [← c112038, ← c112039, ← c112040] at e_1834_2
  have e_1834_3 := arithEq_of_rows h (row := 1834) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_3
  simp only [← c112041, ← c112042, k2, ← c112043] at e_1834_3
  have hr := e_1834_3
  simp only [e_1834_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4187)) := by
  have c112044 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112045 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112046 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112047 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112048 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112049 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_4 := arithEq_of_rows h (row := 1834) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_4
  simp only [← c112044, ← c112045, ← c112046] at e_1834_4
  have e_1834_5 := arithEq_of_rows h (row := 1834) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_5
  simp only [← c112047, ← c112048, k2, ← c112049] at e_1834_5
  have hr := e_1834_5
  simp only [e_1834_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4188)) := by
  have c112050 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112051 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112052 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112053 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112054 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112055 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_6 := arithEq_of_rows h (row := 1834) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_6
  simp only [← c112050, ← c112051, ← c112052] at e_1834_6
  have e_1834_7 := arithEq_of_rows h (row := 1834) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_7
  simp only [← c112053, ← c112054, k2, ← c112055] at e_1834_7
  have hr := e_1834_7
  simp only [e_1834_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4189)) := by
  have c112056 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112057 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112058 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112059 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112060 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112061 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_8 := arithEq_of_rows h (row := 1834) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_8
  simp only [← c112056, ← c112057, ← c112058] at e_1834_8
  have e_1834_9 := arithEq_of_rows h (row := 1834) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_9
  simp only [← c112059, ← c112060, k2, ← c112061] at e_1834_9
  have hr := e_1834_9
  simp only [e_1834_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4190)) := by
  have c112062 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112063 := (publicBatchWrapper89_copies3501 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112064 := (publicBatchWrapper89_copies3502 a h).1
  have c112065 := (publicBatchWrapper89_copies3502 a h).2.1
  have c112066 := (publicBatchWrapper89_copies3502 a h).2.2.1
  have c112067 := (publicBatchWrapper89_copies3502 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_10 := arithEq_of_rows h (row := 1834) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_10
  simp only [← c112062, ← c112063, ← c112064] at e_1834_10
  have e_1834_11 := arithEq_of_rows h (row := 1834) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_11
  simp only [← c112065, ← c112066, k2, ← c112067] at e_1834_11
  have hr := e_1834_11
  simp only [e_1834_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4191)) := by
  have c112068 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.1
  have c112069 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.1
  have c112070 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.1
  have c112071 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.1
  have c112072 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.1
  have c112073 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_12 := arithEq_of_rows h (row := 1834) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_12
  simp only [← c112068, ← c112069, ← c112070] at e_1834_12
  have e_1834_13 := arithEq_of_rows h (row := 1834) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_13
  simp only [← c112071, ← c112072, k2, ← c112073] at e_1834_13
  have hr := e_1834_13
  simp only [e_1834_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4192)) := by
  have c112074 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112075 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112076 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112077 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112078 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112079 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_14 := arithEq_of_rows h (row := 1834) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_14
  simp only [← c112074, ← c112075, ← c112076] at e_1834_14
  have e_1834_15 := arithEq_of_rows h (row := 1834) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_15
  simp only [← c112077, ← c112078, k2, ← c112079] at e_1834_15
  have hr := e_1834_15
  simp only [e_1834_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4193)) := by
  have c112080 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112081 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112082 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112083 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112084 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112085 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_16 := arithEq_of_rows h (row := 1834) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_16
  simp only [← c112080, ← c112081, ← c112082] at e_1834_16
  have e_1834_17 := arithEq_of_rows h (row := 1834) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_17
  simp only [← c112083, ← c112084, k2, ← c112085] at e_1834_17
  have hr := e_1834_17
  simp only [e_1834_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1834 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4194)) := by
  have c112086 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112087 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112088 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112089 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112090 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112091 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1834_18 := arithEq_of_rows h (row := 1834) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_18
  simp only [← c112086, ← c112087, ← c112088] at e_1834_18
  have e_1834_19 := arithEq_of_rows h (row := 1834) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_19
  simp only [← c112089, ← c112090, k2, ← c112091] at e_1834_19
  have hr := e_1834_19
  simp only [e_1834_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4195)) := by
  have c112092 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112093 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112094 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112095 := (publicBatchWrapper89_copies3502 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112096 := (publicBatchWrapper89_copies3503 a h).1
  have c112097 := (publicBatchWrapper89_copies3503 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_0 := arithEq_of_rows h (row := 1835) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_0
  simp only [← c112092, ← c112093, ← c112094] at e_1835_0
  have e_1835_1 := arithEq_of_rows h (row := 1835) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_1
  simp only [← c112095, ← c112096, k2, ← c112097] at e_1835_1
  have hr := e_1835_1
  simp only [e_1835_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4196)) := by
  have c112098 := (publicBatchWrapper89_copies3503 a h).2.2.1
  have c112099 := (publicBatchWrapper89_copies3503 a h).2.2.2.1
  have c112100 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.1
  have c112101 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.1
  have c112102 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.1
  have c112103 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_2 := arithEq_of_rows h (row := 1835) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_2
  simp only [← c112098, ← c112099, ← c112100] at e_1835_2
  have e_1835_3 := arithEq_of_rows h (row := 1835) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_3
  simp only [← c112101, ← c112102, k2, ← c112103] at e_1835_3
  have hr := e_1835_3
  simp only [e_1835_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4197)) := by
  have c112104 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.1
  have c112105 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.1
  have c112106 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112107 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112108 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112109 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_4 := arithEq_of_rows h (row := 1835) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_4
  simp only [← c112104, ← c112105, ← c112106] at e_1835_4
  have e_1835_5 := arithEq_of_rows h (row := 1835) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_5
  simp only [← c112107, ← c112108, k2, ← c112109] at e_1835_5
  have hr := e_1835_5
  simp only [e_1835_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4198)) := by
  have c112110 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112111 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112112 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112113 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112114 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112115 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_6 := arithEq_of_rows h (row := 1835) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_6
  simp only [← c112110, ← c112111, ← c112112] at e_1835_6
  have e_1835_7 := arithEq_of_rows h (row := 1835) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_7
  simp only [← c112113, ← c112114, k2, ← c112115] at e_1835_7
  have hr := e_1835_7
  simp only [e_1835_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4199)) := by
  have c112116 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112117 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112118 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112119 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112120 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112121 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_8 := arithEq_of_rows h (row := 1835) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_8
  simp only [← c112116, ← c112117, ← c112118] at e_1835_8
  have e_1835_9 := arithEq_of_rows h (row := 1835) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_9
  simp only [← c112119, ← c112120, k2, ← c112121] at e_1835_9
  have hr := e_1835_9
  simp only [e_1835_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4200)) := by
  have c112122 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112123 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112124 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112125 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112126 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112127 := (publicBatchWrapper89_copies3503 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_10 := arithEq_of_rows h (row := 1835) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_10
  simp only [← c112122, ← c112123, ← c112124] at e_1835_10
  have e_1835_11 := arithEq_of_rows h (row := 1835) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_11
  simp only [← c112125, ← c112126, k2, ← c112127] at e_1835_11
  have hr := e_1835_11
  simp only [e_1835_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4201)) := by
  have c112128 := (publicBatchWrapper89_copies3504 a h).1
  have c112129 := (publicBatchWrapper89_copies3504 a h).2.1
  have c112130 := (publicBatchWrapper89_copies3504 a h).2.2.1
  have c112131 := (publicBatchWrapper89_copies3504 a h).2.2.2.1
  have c112132 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.1
  have c112133 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_12 := arithEq_of_rows h (row := 1835) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_12
  simp only [← c112128, ← c112129, ← c112130] at e_1835_12
  have e_1835_13 := arithEq_of_rows h (row := 1835) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_13
  simp only [← c112131, ← c112132, k2, ← c112133] at e_1835_13
  have hr := e_1835_13
  simp only [e_1835_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4370)) := by
  have c112134 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.1
  have c112135 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.1
  have c112136 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.1
  have c112137 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.1
  have c112138 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112139 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_14 := arithEq_of_rows h (row := 1835) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_14
  simp only [← c112134, ← c112135, ← c112136] at e_1835_14
  have e_1835_15 := arithEq_of_rows h (row := 1835) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_15
  simp only [← c112137, ← c112138, k2, ← c112139] at e_1835_15
  have hr := e_1835_15
  simp only [e_1835_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4371)) := by
  have c112140 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112141 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112142 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112143 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112144 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112145 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_16 := arithEq_of_rows h (row := 1835) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_16
  simp only [← c112140, ← c112141, ← c112142] at e_1835_16
  have e_1835_17 := arithEq_of_rows h (row := 1835) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_17
  simp only [← c112143, ← c112144, k2, ← c112145] at e_1835_17
  have hr := e_1835_17
  simp only [e_1835_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1835 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4372)) := by
  have c112146 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112147 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112148 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112149 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112150 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112151 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1835_18 := arithEq_of_rows h (row := 1835) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_18
  simp only [← c112146, ← c112147, ← c112148] at e_1835_18
  have e_1835_19 := arithEq_of_rows h (row := 1835) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_19
  simp only [← c112149, ← c112150, k2, ← c112151] at e_1835_19
  have hr := e_1835_19
  simp only [e_1835_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4373)) := by
  have c112152 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112153 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112154 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112155 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112156 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112157 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_0 := arithEq_of_rows h (row := 1836) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_0
  simp only [← c112152, ← c112153, ← c112154] at e_1836_0
  have e_1836_1 := arithEq_of_rows h (row := 1836) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_1
  simp only [← c112155, ← c112156, k2, ← c112157] at e_1836_1
  have hr := e_1836_1
  simp only [e_1836_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4374)) := by
  have c112158 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112159 := (publicBatchWrapper89_copies3504 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112160 := (publicBatchWrapper89_copies3505 a h).1
  have c112161 := (publicBatchWrapper89_copies3505 a h).2.1
  have c112162 := (publicBatchWrapper89_copies3505 a h).2.2.1
  have c112163 := (publicBatchWrapper89_copies3505 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_2 := arithEq_of_rows h (row := 1836) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_2
  simp only [← c112158, ← c112159, ← c112160] at e_1836_2
  have e_1836_3 := arithEq_of_rows h (row := 1836) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_3
  simp only [← c112161, ← c112162, k2, ← c112163] at e_1836_3
  have hr := e_1836_3
  simp only [e_1836_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4375)) := by
  have c112164 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.1
  have c112165 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.1
  have c112166 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.1
  have c112167 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.1
  have c112168 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.1
  have c112169 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_4 := arithEq_of_rows h (row := 1836) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_4
  simp only [← c112164, ← c112165, ← c112166] at e_1836_4
  have e_1836_5 := arithEq_of_rows h (row := 1836) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_5
  simp only [← c112167, ← c112168, k2, ← c112169] at e_1836_5
  have hr := e_1836_5
  simp only [e_1836_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4376)) := by
  have c112170 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112171 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112172 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112173 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112174 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112175 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_6 := arithEq_of_rows h (row := 1836) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_6
  simp only [← c112170, ← c112171, ← c112172] at e_1836_6
  have e_1836_7 := arithEq_of_rows h (row := 1836) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_7
  simp only [← c112173, ← c112174, k2, ← c112175] at e_1836_7
  have hr := e_1836_7
  simp only [e_1836_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4377)) := by
  have c112176 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112177 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112178 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112179 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112180 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112181 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_8 := arithEq_of_rows h (row := 1836) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_8
  simp only [← c112176, ← c112177, ← c112178] at e_1836_8
  have e_1836_9 := arithEq_of_rows h (row := 1836) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_9
  simp only [← c112179, ← c112180, k2, ← c112181] at e_1836_9
  have hr := e_1836_9
  simp only [e_1836_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4378)) := by
  have c112182 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112183 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112184 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112185 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112186 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112187 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_10 := arithEq_of_rows h (row := 1836) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_10
  simp only [← c112182, ← c112183, ← c112184] at e_1836_10
  have e_1836_11 := arithEq_of_rows h (row := 1836) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_11
  simp only [← c112185, ← c112186, k2, ← c112187] at e_1836_11
  have hr := e_1836_11
  simp only [e_1836_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4379)) := by
  have c112188 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112189 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112190 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112191 := (publicBatchWrapper89_copies3505 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112192 := (publicBatchWrapper89_copies3506 a h).1
  have c112193 := (publicBatchWrapper89_copies3506 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_12 := arithEq_of_rows h (row := 1836) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_12
  simp only [← c112188, ← c112189, ← c112190] at e_1836_12
  have e_1836_13 := arithEq_of_rows h (row := 1836) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_13
  simp only [← c112191, ← c112192, k2, ← c112193] at e_1836_13
  have hr := e_1836_13
  simp only [e_1836_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4380)) := by
  have c112194 := (publicBatchWrapper89_copies3506 a h).2.2.1
  have c112195 := (publicBatchWrapper89_copies3506 a h).2.2.2.1
  have c112196 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.1
  have c112197 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.1
  have c112198 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.1
  have c112199 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_14 := arithEq_of_rows h (row := 1836) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_14
  simp only [← c112194, ← c112195, ← c112196] at e_1836_14
  have e_1836_15 := arithEq_of_rows h (row := 1836) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_15
  simp only [← c112197, ← c112198, k2, ← c112199] at e_1836_15
  have hr := e_1836_15
  simp only [e_1836_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4381)) := by
  have c112200 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.1
  have c112201 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.1
  have c112202 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112203 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112204 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112205 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_16 := arithEq_of_rows h (row := 1836) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_16
  simp only [← c112200, ← c112201, ← c112202] at e_1836_16
  have e_1836_17 := arithEq_of_rows h (row := 1836) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_17
  simp only [← c112203, ← c112204, k2, ← c112205] at e_1836_17
  have hr := e_1836_17
  simp only [e_1836_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1836 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4382)) := by
  have c112206 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112207 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112208 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112209 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112210 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112211 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1836_18 := arithEq_of_rows h (row := 1836) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_18
  simp only [← c112206, ← c112207, ← c112208] at e_1836_18
  have e_1836_19 := arithEq_of_rows h (row := 1836) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_19
  simp only [← c112209, ← c112210, k2, ← c112211] at e_1836_19
  have hr := e_1836_19
  simp only [e_1836_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4383)) := by
  have c112212 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112213 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112214 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112215 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112216 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112217 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_0 := arithEq_of_rows h (row := 1837) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_0
  simp only [← c112212, ← c112213, ← c112214] at e_1837_0
  have e_1837_1 := arithEq_of_rows h (row := 1837) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_1
  simp only [← c112215, ← c112216, k2, ← c112217] at e_1837_1
  have hr := e_1837_1
  simp only [e_1837_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4384)) := by
  have c112218 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112219 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112220 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112221 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112222 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112223 := (publicBatchWrapper89_copies3506 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_2 := arithEq_of_rows h (row := 1837) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_2
  simp only [← c112218, ← c112219, ← c112220] at e_1837_2
  have e_1837_3 := arithEq_of_rows h (row := 1837) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_3
  simp only [← c112221, ← c112222, k2, ← c112223] at e_1837_3
  have hr := e_1837_3
  simp only [e_1837_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4385)) := by
  have c112224 := (publicBatchWrapper89_copies3507 a h).1
  have c112225 := (publicBatchWrapper89_copies3507 a h).2.1
  have c112226 := (publicBatchWrapper89_copies3507 a h).2.2.1
  have c112227 := (publicBatchWrapper89_copies3507 a h).2.2.2.1
  have c112228 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.1
  have c112229 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_4 := arithEq_of_rows h (row := 1837) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_4
  simp only [← c112224, ← c112225, ← c112226] at e_1837_4
  have e_1837_5 := arithEq_of_rows h (row := 1837) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_5
  simp only [← c112227, ← c112228, k2, ← c112229] at e_1837_5
  have hr := e_1837_5
  simp only [e_1837_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4386)) := by
  have c112230 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.1
  have c112231 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.1
  have c112232 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.1
  have c112233 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.1
  have c112234 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112235 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_6 := arithEq_of_rows h (row := 1837) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_6
  simp only [← c112230, ← c112231, ← c112232] at e_1837_6
  have e_1837_7 := arithEq_of_rows h (row := 1837) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_7
  simp only [← c112233, ← c112234, k2, ← c112235] at e_1837_7
  have hr := e_1837_7
  simp only [e_1837_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4387)) := by
  have c112236 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112237 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112238 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112239 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112240 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112241 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_8 := arithEq_of_rows h (row := 1837) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_8
  simp only [← c112236, ← c112237, ← c112238] at e_1837_8
  have e_1837_9 := arithEq_of_rows h (row := 1837) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_9
  simp only [← c112239, ← c112240, k2, ← c112241] at e_1837_9
  have hr := e_1837_9
  simp only [e_1837_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4388)) := by
  have c112242 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112243 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112244 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112245 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112246 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112247 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_10 := arithEq_of_rows h (row := 1837) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_10
  simp only [← c112242, ← c112243, ← c112244] at e_1837_10
  have e_1837_11 := arithEq_of_rows h (row := 1837) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_11
  simp only [← c112245, ← c112246, k2, ← c112247] at e_1837_11
  have hr := e_1837_11
  simp only [e_1837_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4389)) := by
  have c112248 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112249 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112250 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112251 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112252 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112253 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_12 := arithEq_of_rows h (row := 1837) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_12
  simp only [← c112248, ← c112249, ← c112250] at e_1837_12
  have e_1837_13 := arithEq_of_rows h (row := 1837) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_13
  simp only [← c112251, ← c112252, k2, ← c112253] at e_1837_13
  have hr := e_1837_13
  simp only [e_1837_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4390)) := by
  have c112254 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112255 := (publicBatchWrapper89_copies3507 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112256 := (publicBatchWrapper89_copies3508 a h).1
  have c112257 := (publicBatchWrapper89_copies3508 a h).2.1
  have c112258 := (publicBatchWrapper89_copies3508 a h).2.2.1
  have c112259 := (publicBatchWrapper89_copies3508 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_14 := arithEq_of_rows h (row := 1837) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_14
  simp only [← c112254, ← c112255, ← c112256] at e_1837_14
  have e_1837_15 := arithEq_of_rows h (row := 1837) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_15
  simp only [← c112257, ← c112258, k2, ← c112259] at e_1837_15
  have hr := e_1837_15
  simp only [e_1837_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4391)) := by
  have c112260 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.1
  have c112261 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.1
  have c112262 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.1
  have c112263 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.1
  have c112264 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.1
  have c112265 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_16 := arithEq_of_rows h (row := 1837) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_16
  simp only [← c112260, ← c112261, ← c112262] at e_1837_16
  have e_1837_17 := arithEq_of_rows h (row := 1837) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_17
  simp only [← c112263, ← c112264, k2, ← c112265] at e_1837_17
  have hr := e_1837_17
  simp only [e_1837_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1837 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4392)) := by
  have c112266 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112267 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112268 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112269 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112270 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112271 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1837_18 := arithEq_of_rows h (row := 1837) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_18
  simp only [← c112266, ← c112267, ← c112268] at e_1837_18
  have e_1837_19 := arithEq_of_rows h (row := 1837) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_19
  simp only [← c112269, ← c112270, k2, ← c112271] at e_1837_19
  have hr := e_1837_19
  simp only [e_1837_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4393)) := by
  have c112272 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112273 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112274 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112275 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112276 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112277 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_0 := arithEq_of_rows h (row := 1838) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_0
  simp only [← c112272, ← c112273, ← c112274] at e_1838_0
  have e_1838_1 := arithEq_of_rows h (row := 1838) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_1
  simp only [← c112275, ← c112276, k2, ← c112277] at e_1838_1
  have hr := e_1838_1
  simp only [e_1838_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4394)) := by
  have c112278 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112279 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112280 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112281 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112282 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112283 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_2 := arithEq_of_rows h (row := 1838) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_2
  simp only [← c112278, ← c112279, ← c112280] at e_1838_2
  have e_1838_3 := arithEq_of_rows h (row := 1838) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_3
  simp only [← c112281, ← c112282, k2, ← c112283] at e_1838_3
  have hr := e_1838_3
  simp only [e_1838_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4395)) := by
  have c112284 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112285 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112286 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112287 := (publicBatchWrapper89_copies3508 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112288 := (publicBatchWrapper89_copies3509 a h).1
  have c112289 := (publicBatchWrapper89_copies3509 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_4 := arithEq_of_rows h (row := 1838) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_4
  simp only [← c112284, ← c112285, ← c112286] at e_1838_4
  have e_1838_5 := arithEq_of_rows h (row := 1838) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_5
  simp only [← c112287, ← c112288, k2, ← c112289] at e_1838_5
  have hr := e_1838_5
  simp only [e_1838_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4396)) := by
  have c112290 := (publicBatchWrapper89_copies3509 a h).2.2.1
  have c112291 := (publicBatchWrapper89_copies3509 a h).2.2.2.1
  have c112292 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.1
  have c112293 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.1
  have c112294 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.1
  have c112295 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_6 := arithEq_of_rows h (row := 1838) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_6
  simp only [← c112290, ← c112291, ← c112292] at e_1838_6
  have e_1838_7 := arithEq_of_rows h (row := 1838) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_7
  simp only [← c112293, ← c112294, k2, ← c112295] at e_1838_7
  have hr := e_1838_7
  simp only [e_1838_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4397)) := by
  have c112296 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.1
  have c112297 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.1
  have c112298 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112299 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112300 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112301 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_8 := arithEq_of_rows h (row := 1838) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_8
  simp only [← c112296, ← c112297, ← c112298] at e_1838_8
  have e_1838_9 := arithEq_of_rows h (row := 1838) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_9
  simp only [← c112299, ← c112300, k2, ← c112301] at e_1838_9
  have hr := e_1838_9
  simp only [e_1838_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4398)) := by
  have c112302 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112303 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112304 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112305 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112306 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112307 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_10 := arithEq_of_rows h (row := 1838) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_10
  simp only [← c112302, ← c112303, ← c112304] at e_1838_10
  have e_1838_11 := arithEq_of_rows h (row := 1838) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_11
  simp only [← c112305, ← c112306, k2, ← c112307] at e_1838_11
  have hr := e_1838_11
  simp only [e_1838_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4399)) := by
  have c112308 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112309 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112310 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112311 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112312 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112313 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_12 := arithEq_of_rows h (row := 1838) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_12
  simp only [← c112308, ← c112309, ← c112310] at e_1838_12
  have e_1838_13 := arithEq_of_rows h (row := 1838) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_13
  simp only [← c112311, ← c112312, k2, ← c112313] at e_1838_13
  have hr := e_1838_13
  simp only [e_1838_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4400)) := by
  have c112314 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112315 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112316 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112317 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112318 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112319 := (publicBatchWrapper89_copies3509 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_14 := arithEq_of_rows h (row := 1838) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_14
  simp only [← c112314, ← c112315, ← c112316] at e_1838_14
  have e_1838_15 := arithEq_of_rows h (row := 1838) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_15
  simp only [← c112317, ← c112318, k2, ← c112319] at e_1838_15
  have hr := e_1838_15
  simp only [e_1838_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4401)) := by
  have c112320 := (publicBatchWrapper89_copies3510 a h).1
  have c112321 := (publicBatchWrapper89_copies3510 a h).2.1
  have c112322 := (publicBatchWrapper89_copies3510 a h).2.2.1
  have c112323 := (publicBatchWrapper89_copies3510 a h).2.2.2.1
  have c112324 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.1
  have c112325 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_16 := arithEq_of_rows h (row := 1838) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_16
  simp only [← c112320, ← c112321, ← c112322] at e_1838_16
  have e_1838_17 := arithEq_of_rows h (row := 1838) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_17
  simp only [← c112323, ← c112324, k2, ← c112325] at e_1838_17
  have hr := e_1838_17
  simp only [e_1838_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1838 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4570)) := by
  have c112326 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.1
  have c112327 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.1
  have c112328 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.1
  have c112329 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.1
  have c112330 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112331 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1838_18 := arithEq_of_rows h (row := 1838) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_18
  simp only [← c112326, ← c112327, ← c112328] at e_1838_18
  have e_1838_19 := arithEq_of_rows h (row := 1838) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_19
  simp only [← c112329, ← c112330, k2, ← c112331] at e_1838_19
  have hr := e_1838_19
  simp only [e_1838_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4571)) := by
  have c112332 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112333 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112334 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112335 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112336 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112337 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_0 := arithEq_of_rows h (row := 1839) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_0
  simp only [← c112332, ← c112333, ← c112334] at e_1839_0
  have e_1839_1 := arithEq_of_rows h (row := 1839) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_1
  simp only [← c112335, ← c112336, k2, ← c112337] at e_1839_1
  have hr := e_1839_1
  simp only [e_1839_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4572)) := by
  have c112338 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112339 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112340 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112341 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112342 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112343 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_2 := arithEq_of_rows h (row := 1839) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_2
  simp only [← c112338, ← c112339, ← c112340] at e_1839_2
  have e_1839_3 := arithEq_of_rows h (row := 1839) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_3
  simp only [← c112341, ← c112342, k2, ← c112343] at e_1839_3
  have hr := e_1839_3
  simp only [e_1839_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4573)) := by
  have c112344 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112345 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112346 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112347 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112348 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112349 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_4 := arithEq_of_rows h (row := 1839) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_4
  simp only [← c112344, ← c112345, ← c112346] at e_1839_4
  have e_1839_5 := arithEq_of_rows h (row := 1839) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_5
  simp only [← c112347, ← c112348, k2, ← c112349] at e_1839_5
  have hr := e_1839_5
  simp only [e_1839_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4574)) := by
  have c112350 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112351 := (publicBatchWrapper89_copies3510 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112352 := (publicBatchWrapper89_copies3511 a h).1
  have c112353 := (publicBatchWrapper89_copies3511 a h).2.1
  have c112354 := (publicBatchWrapper89_copies3511 a h).2.2.1
  have c112355 := (publicBatchWrapper89_copies3511 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_6 := arithEq_of_rows h (row := 1839) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_6
  simp only [← c112350, ← c112351, ← c112352] at e_1839_6
  have e_1839_7 := arithEq_of_rows h (row := 1839) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_7
  simp only [← c112353, ← c112354, k2, ← c112355] at e_1839_7
  have hr := e_1839_7
  simp only [e_1839_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4575)) := by
  have c112356 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.1
  have c112357 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.1
  have c112358 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.1
  have c112359 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.1
  have c112360 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.1
  have c112361 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_8 := arithEq_of_rows h (row := 1839) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_8
  simp only [← c112356, ← c112357, ← c112358] at e_1839_8
  have e_1839_9 := arithEq_of_rows h (row := 1839) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_9
  simp only [← c112359, ← c112360, k2, ← c112361] at e_1839_9
  have hr := e_1839_9
  simp only [e_1839_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4576)) := by
  have c112362 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112363 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112364 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112365 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112366 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112367 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_10 := arithEq_of_rows h (row := 1839) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_10
  simp only [← c112362, ← c112363, ← c112364] at e_1839_10
  have e_1839_11 := arithEq_of_rows h (row := 1839) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_11
  simp only [← c112365, ← c112366, k2, ← c112367] at e_1839_11
  have hr := e_1839_11
  simp only [e_1839_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4577)) := by
  have c112368 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112369 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112370 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112371 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112372 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112373 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_12 := arithEq_of_rows h (row := 1839) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_12
  simp only [← c112368, ← c112369, ← c112370] at e_1839_12
  have e_1839_13 := arithEq_of_rows h (row := 1839) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_13
  simp only [← c112371, ← c112372, k2, ← c112373] at e_1839_13
  have hr := e_1839_13
  simp only [e_1839_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4578)) := by
  have c112374 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112375 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112376 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112377 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112378 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112379 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_14 := arithEq_of_rows h (row := 1839) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_14
  simp only [← c112374, ← c112375, ← c112376] at e_1839_14
  have e_1839_15 := arithEq_of_rows h (row := 1839) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_15
  simp only [← c112377, ← c112378, k2, ← c112379] at e_1839_15
  have hr := e_1839_15
  simp only [e_1839_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4579)) := by
  have c112380 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112381 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112382 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112383 := (publicBatchWrapper89_copies3511 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112384 := (publicBatchWrapper89_copies3512 a h).1
  have c112385 := (publicBatchWrapper89_copies3512 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_16 := arithEq_of_rows h (row := 1839) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_16
  simp only [← c112380, ← c112381, ← c112382] at e_1839_16
  have e_1839_17 := arithEq_of_rows h (row := 1839) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_17
  simp only [← c112383, ← c112384, k2, ← c112385] at e_1839_17
  have hr := e_1839_17
  simp only [e_1839_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1839 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4580)) := by
  have c112386 := (publicBatchWrapper89_copies3512 a h).2.2.1
  have c112387 := (publicBatchWrapper89_copies3512 a h).2.2.2.1
  have c112388 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.1
  have c112389 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.1
  have c112390 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.1
  have c112391 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1839_18 := arithEq_of_rows h (row := 1839) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_18
  simp only [← c112386, ← c112387, ← c112388] at e_1839_18
  have e_1839_19 := arithEq_of_rows h (row := 1839) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_19
  simp only [← c112389, ← c112390, k2, ← c112391] at e_1839_19
  have hr := e_1839_19
  simp only [e_1839_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4581)) := by
  have c112392 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.1
  have c112393 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.1
  have c112394 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112395 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112396 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112397 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_0 := arithEq_of_rows h (row := 1840) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_0
  simp only [← c112392, ← c112393, ← c112394] at e_1840_0
  have e_1840_1 := arithEq_of_rows h (row := 1840) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_1
  simp only [← c112395, ← c112396, k2, ← c112397] at e_1840_1
  have hr := e_1840_1
  simp only [e_1840_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4582)) := by
  have c112398 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112399 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112400 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112401 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112402 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112403 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_2 := arithEq_of_rows h (row := 1840) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_2
  simp only [← c112398, ← c112399, ← c112400] at e_1840_2
  have e_1840_3 := arithEq_of_rows h (row := 1840) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_3
  simp only [← c112401, ← c112402, k2, ← c112403] at e_1840_3
  have hr := e_1840_3
  simp only [e_1840_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4583)) := by
  have c112404 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112405 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112406 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112407 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112408 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112409 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_4 := arithEq_of_rows h (row := 1840) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_4
  simp only [← c112404, ← c112405, ← c112406] at e_1840_4
  have e_1840_5 := arithEq_of_rows h (row := 1840) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_5
  simp only [← c112407, ← c112408, k2, ← c112409] at e_1840_5
  have hr := e_1840_5
  simp only [e_1840_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4584)) := by
  have c112410 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112411 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112412 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112413 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112414 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112415 := (publicBatchWrapper89_copies3512 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_6 := arithEq_of_rows h (row := 1840) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_6
  simp only [← c112410, ← c112411, ← c112412] at e_1840_6
  have e_1840_7 := arithEq_of_rows h (row := 1840) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_7
  simp only [← c112413, ← c112414, k2, ← c112415] at e_1840_7
  have hr := e_1840_7
  simp only [e_1840_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4585)) := by
  have c112416 := (publicBatchWrapper89_copies3513 a h).1
  have c112417 := (publicBatchWrapper89_copies3513 a h).2.1
  have c112418 := (publicBatchWrapper89_copies3513 a h).2.2.1
  have c112419 := (publicBatchWrapper89_copies3513 a h).2.2.2.1
  have c112420 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.1
  have c112421 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_8 := arithEq_of_rows h (row := 1840) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_8
  simp only [← c112416, ← c112417, ← c112418] at e_1840_8
  have e_1840_9 := arithEq_of_rows h (row := 1840) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_9
  simp only [← c112419, ← c112420, k2, ← c112421] at e_1840_9
  have hr := e_1840_9
  simp only [e_1840_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4586)) := by
  have c112422 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.1
  have c112423 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.1
  have c112424 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.1
  have c112425 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.1
  have c112426 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112427 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_10 := arithEq_of_rows h (row := 1840) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_10
  simp only [← c112422, ← c112423, ← c112424] at e_1840_10
  have e_1840_11 := arithEq_of_rows h (row := 1840) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_11
  simp only [← c112425, ← c112426, k2, ← c112427] at e_1840_11
  have hr := e_1840_11
  simp only [e_1840_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4587)) := by
  have c112428 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112429 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112430 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112431 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112432 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112433 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_12 := arithEq_of_rows h (row := 1840) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_12
  simp only [← c112428, ← c112429, ← c112430] at e_1840_12
  have e_1840_13 := arithEq_of_rows h (row := 1840) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_13
  simp only [← c112431, ← c112432, k2, ← c112433] at e_1840_13
  have hr := e_1840_13
  simp only [e_1840_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4588)) := by
  have c112434 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112435 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112436 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112437 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112438 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112439 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_14 := arithEq_of_rows h (row := 1840) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_14
  simp only [← c112434, ← c112435, ← c112436] at e_1840_14
  have e_1840_15 := arithEq_of_rows h (row := 1840) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_15
  simp only [← c112437, ← c112438, k2, ← c112439] at e_1840_15
  have hr := e_1840_15
  simp only [e_1840_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4589)) := by
  have c112440 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112441 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112442 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112443 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112444 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112445 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_16 := arithEq_of_rows h (row := 1840) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_16
  simp only [← c112440, ← c112441, ← c112442] at e_1840_16
  have e_1840_17 := arithEq_of_rows h (row := 1840) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_17
  simp only [← c112443, ← c112444, k2, ← c112445] at e_1840_17
  have hr := e_1840_17
  simp only [e_1840_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1840 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4590)) := by
  have c112446 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112447 := (publicBatchWrapper89_copies3513 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112448 := (publicBatchWrapper89_copies3514 a h).1
  have c112449 := (publicBatchWrapper89_copies3514 a h).2.1
  have c112450 := (publicBatchWrapper89_copies3514 a h).2.2.1
  have c112451 := (publicBatchWrapper89_copies3514 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1840_18 := arithEq_of_rows h (row := 1840) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_18
  simp only [← c112446, ← c112447, ← c112448] at e_1840_18
  have e_1840_19 := arithEq_of_rows h (row := 1840) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_19
  simp only [← c112449, ← c112450, k2, ← c112451] at e_1840_19
  have hr := e_1840_19
  simp only [e_1840_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4591)) := by
  have c112452 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.1
  have c112453 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.1
  have c112454 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.1
  have c112455 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.1
  have c112456 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.1
  have c112457 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_0 := arithEq_of_rows h (row := 1841) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_0
  simp only [← c112452, ← c112453, ← c112454] at e_1841_0
  have e_1841_1 := arithEq_of_rows h (row := 1841) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_1
  simp only [← c112455, ← c112456, k2, ← c112457] at e_1841_1
  have hr := e_1841_1
  simp only [e_1841_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4592)) := by
  have c112458 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112459 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112460 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112461 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112462 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112463 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_2 := arithEq_of_rows h (row := 1841) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_2
  simp only [← c112458, ← c112459, ← c112460] at e_1841_2
  have e_1841_3 := arithEq_of_rows h (row := 1841) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_3
  simp only [← c112461, ← c112462, k2, ← c112463] at e_1841_3
  have hr := e_1841_3
  simp only [e_1841_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4593)) := by
  have c112464 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112465 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112466 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112467 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112468 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112469 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_4 := arithEq_of_rows h (row := 1841) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_4
  simp only [← c112464, ← c112465, ← c112466] at e_1841_4
  have e_1841_5 := arithEq_of_rows h (row := 1841) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_5
  simp only [← c112467, ← c112468, k2, ← c112469] at e_1841_5
  have hr := e_1841_5
  simp only [e_1841_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4594)) := by
  have c112470 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112471 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112472 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112473 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112474 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112475 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_6 := arithEq_of_rows h (row := 1841) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_6
  simp only [← c112470, ← c112471, ← c112472] at e_1841_6
  have e_1841_7 := arithEq_of_rows h (row := 1841) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_7
  simp only [← c112473, ← c112474, k2, ← c112475] at e_1841_7
  have hr := e_1841_7
  simp only [e_1841_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4595)) := by
  have c112476 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112477 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112478 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112479 := (publicBatchWrapper89_copies3514 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112480 := (publicBatchWrapper89_copies3515 a h).1
  have c112481 := (publicBatchWrapper89_copies3515 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_8 := arithEq_of_rows h (row := 1841) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_8
  simp only [← c112476, ← c112477, ← c112478] at e_1841_8
  have e_1841_9 := arithEq_of_rows h (row := 1841) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_9
  simp only [← c112479, ← c112480, k2, ← c112481] at e_1841_9
  have hr := e_1841_9
  simp only [e_1841_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4596)) := by
  have c112482 := (publicBatchWrapper89_copies3515 a h).2.2.1
  have c112483 := (publicBatchWrapper89_copies3515 a h).2.2.2.1
  have c112484 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.1
  have c112485 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.1
  have c112486 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.1
  have c112487 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_10 := arithEq_of_rows h (row := 1841) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_10
  simp only [← c112482, ← c112483, ← c112484] at e_1841_10
  have e_1841_11 := arithEq_of_rows h (row := 1841) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_11
  simp only [← c112485, ← c112486, k2, ← c112487] at e_1841_11
  have hr := e_1841_11
  simp only [e_1841_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4597)) := by
  have c112488 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.1
  have c112489 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.1
  have c112490 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112491 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112492 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112493 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_12 := arithEq_of_rows h (row := 1841) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_12
  simp only [← c112488, ← c112489, ← c112490] at e_1841_12
  have e_1841_13 := arithEq_of_rows h (row := 1841) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_13
  simp only [← c112491, ← c112492, k2, ← c112493] at e_1841_13
  have hr := e_1841_13
  simp only [e_1841_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4598)) := by
  have c112494 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112495 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112496 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112497 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112498 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112499 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_14 := arithEq_of_rows h (row := 1841) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_14
  simp only [← c112494, ← c112495, ← c112496] at e_1841_14
  have e_1841_15 := arithEq_of_rows h (row := 1841) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_15
  simp only [← c112497, ← c112498, k2, ← c112499] at e_1841_15
  have hr := e_1841_15
  simp only [e_1841_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4599)) := by
  have c112500 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112501 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112502 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112503 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112504 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112505 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_16 := arithEq_of_rows h (row := 1841) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_16
  simp only [← c112500, ← c112501, ← c112502] at e_1841_16
  have e_1841_17 := arithEq_of_rows h (row := 1841) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_17
  simp only [← c112503, ← c112504, k2, ← c112505] at e_1841_17
  have hr := e_1841_17
  simp only [e_1841_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1841 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4600)) := by
  have c112506 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112507 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112508 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112509 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112510 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112511 := (publicBatchWrapper89_copies3515 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1841_18 := arithEq_of_rows h (row := 1841) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_18
  simp only [← c112506, ← c112507, ← c112508] at e_1841_18
  have e_1841_19 := arithEq_of_rows h (row := 1841) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_19
  simp only [← c112509, ← c112510, k2, ← c112511] at e_1841_19
  have hr := e_1841_19
  simp only [e_1841_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4601)) := by
  have c112512 := (publicBatchWrapper89_copies3516 a h).1
  have c112513 := (publicBatchWrapper89_copies3516 a h).2.1
  have c112514 := (publicBatchWrapper89_copies3516 a h).2.2.1
  have c112515 := (publicBatchWrapper89_copies3516 a h).2.2.2.1
  have c112516 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.1
  have c112517 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_0 := arithEq_of_rows h (row := 1842) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_0
  simp only [← c112512, ← c112513, ← c112514] at e_1842_0
  have e_1842_1 := arithEq_of_rows h (row := 1842) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_1
  simp only [← c112515, ← c112516, k2, ← c112517] at e_1842_1
  have hr := e_1842_1
  simp only [e_1842_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4770)) := by
  have c112518 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.1
  have c112519 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.1
  have c112520 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.1
  have c112521 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.1
  have c112522 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112523 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_2 := arithEq_of_rows h (row := 1842) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_2
  simp only [← c112518, ← c112519, ← c112520] at e_1842_2
  have e_1842_3 := arithEq_of_rows h (row := 1842) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_3
  simp only [← c112521, ← c112522, k2, ← c112523] at e_1842_3
  have hr := e_1842_3
  simp only [e_1842_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4771)) := by
  have c112524 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112525 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112526 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112527 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112528 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112529 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_4 := arithEq_of_rows h (row := 1842) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_4
  simp only [← c112524, ← c112525, ← c112526] at e_1842_4
  have e_1842_5 := arithEq_of_rows h (row := 1842) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_5
  simp only [← c112527, ← c112528, k2, ← c112529] at e_1842_5
  have hr := e_1842_5
  simp only [e_1842_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4772)) := by
  have c112530 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112531 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112532 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112533 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112534 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112535 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_6 := arithEq_of_rows h (row := 1842) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_6
  simp only [← c112530, ← c112531, ← c112532] at e_1842_6
  have e_1842_7 := arithEq_of_rows h (row := 1842) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_7
  simp only [← c112533, ← c112534, k2, ← c112535] at e_1842_7
  have hr := e_1842_7
  simp only [e_1842_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4773)) := by
  have c112536 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112537 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112538 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112539 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112540 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112541 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_8 := arithEq_of_rows h (row := 1842) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_8
  simp only [← c112536, ← c112537, ← c112538] at e_1842_8
  have e_1842_9 := arithEq_of_rows h (row := 1842) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_9
  simp only [← c112539, ← c112540, k2, ← c112541] at e_1842_9
  have hr := e_1842_9
  simp only [e_1842_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4774)) := by
  have c112542 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112543 := (publicBatchWrapper89_copies3516 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112544 := (publicBatchWrapper89_copies3517 a h).1
  have c112545 := (publicBatchWrapper89_copies3517 a h).2.1
  have c112546 := (publicBatchWrapper89_copies3517 a h).2.2.1
  have c112547 := (publicBatchWrapper89_copies3517 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_10 := arithEq_of_rows h (row := 1842) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_10
  simp only [← c112542, ← c112543, ← c112544] at e_1842_10
  have e_1842_11 := arithEq_of_rows h (row := 1842) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_11
  simp only [← c112545, ← c112546, k2, ← c112547] at e_1842_11
  have hr := e_1842_11
  simp only [e_1842_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4775)) := by
  have c112548 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.1
  have c112549 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.1
  have c112550 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.1
  have c112551 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.1
  have c112552 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.1
  have c112553 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_12 := arithEq_of_rows h (row := 1842) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_12
  simp only [← c112548, ← c112549, ← c112550] at e_1842_12
  have e_1842_13 := arithEq_of_rows h (row := 1842) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_13
  simp only [← c112551, ← c112552, k2, ← c112553] at e_1842_13
  have hr := e_1842_13
  simp only [e_1842_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4776)) := by
  have c112554 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112555 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112556 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112557 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112558 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112559 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_14 := arithEq_of_rows h (row := 1842) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_14
  simp only [← c112554, ← c112555, ← c112556] at e_1842_14
  have e_1842_15 := arithEq_of_rows h (row := 1842) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_15
  simp only [← c112557, ← c112558, k2, ← c112559] at e_1842_15
  have hr := e_1842_15
  simp only [e_1842_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4777)) := by
  have c112560 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112561 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112562 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112563 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112564 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112565 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_16 := arithEq_of_rows h (row := 1842) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_16
  simp only [← c112560, ← c112561, ← c112562] at e_1842_16
  have e_1842_17 := arithEq_of_rows h (row := 1842) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_17
  simp only [← c112563, ← c112564, k2, ← c112565] at e_1842_17
  have hr := e_1842_17
  simp only [e_1842_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1842 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4778)) := by
  have c112566 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112567 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112568 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112569 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112570 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112571 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1842_18 := arithEq_of_rows h (row := 1842) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_18
  simp only [← c112566, ← c112567, ← c112568] at e_1842_18
  have e_1842_19 := arithEq_of_rows h (row := 1842) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_19
  simp only [← c112569, ← c112570, k2, ← c112571] at e_1842_19
  have hr := e_1842_19
  simp only [e_1842_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4779)) := by
  have c112572 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112573 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112574 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112575 := (publicBatchWrapper89_copies3517 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112576 := (publicBatchWrapper89_copies3518 a h).1
  have c112577 := (publicBatchWrapper89_copies3518 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_0 := arithEq_of_rows h (row := 1843) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_0
  simp only [← c112572, ← c112573, ← c112574] at e_1843_0
  have e_1843_1 := arithEq_of_rows h (row := 1843) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_1
  simp only [← c112575, ← c112576, k2, ← c112577] at e_1843_1
  have hr := e_1843_1
  simp only [e_1843_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4780)) := by
  have c112578 := (publicBatchWrapper89_copies3518 a h).2.2.1
  have c112579 := (publicBatchWrapper89_copies3518 a h).2.2.2.1
  have c112580 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.1
  have c112581 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.1
  have c112582 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.1
  have c112583 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_2 := arithEq_of_rows h (row := 1843) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_2
  simp only [← c112578, ← c112579, ← c112580] at e_1843_2
  have e_1843_3 := arithEq_of_rows h (row := 1843) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_3
  simp only [← c112581, ← c112582, k2, ← c112583] at e_1843_3
  have hr := e_1843_3
  simp only [e_1843_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4781)) := by
  have c112584 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.1
  have c112585 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.1
  have c112586 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112587 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112588 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112589 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_4 := arithEq_of_rows h (row := 1843) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_4
  simp only [← c112584, ← c112585, ← c112586] at e_1843_4
  have e_1843_5 := arithEq_of_rows h (row := 1843) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_5
  simp only [← c112587, ← c112588, k2, ← c112589] at e_1843_5
  have hr := e_1843_5
  simp only [e_1843_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4782)) := by
  have c112590 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112591 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112592 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112593 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112594 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112595 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_6 := arithEq_of_rows h (row := 1843) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_6
  simp only [← c112590, ← c112591, ← c112592] at e_1843_6
  have e_1843_7 := arithEq_of_rows h (row := 1843) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_7
  simp only [← c112593, ← c112594, k2, ← c112595] at e_1843_7
  have hr := e_1843_7
  simp only [e_1843_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4783)) := by
  have c112596 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112597 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112598 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112599 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112600 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112601 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_8 := arithEq_of_rows h (row := 1843) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_8
  simp only [← c112596, ← c112597, ← c112598] at e_1843_8
  have e_1843_9 := arithEq_of_rows h (row := 1843) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_9
  simp only [← c112599, ← c112600, k2, ← c112601] at e_1843_9
  have hr := e_1843_9
  simp only [e_1843_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4784)) := by
  have c112602 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112603 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112604 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112605 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112606 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112607 := (publicBatchWrapper89_copies3518 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_10 := arithEq_of_rows h (row := 1843) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_10
  simp only [← c112602, ← c112603, ← c112604] at e_1843_10
  have e_1843_11 := arithEq_of_rows h (row := 1843) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_11
  simp only [← c112605, ← c112606, k2, ← c112607] at e_1843_11
  have hr := e_1843_11
  simp only [e_1843_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4785)) := by
  have c112608 := (publicBatchWrapper89_copies3519 a h).1
  have c112609 := (publicBatchWrapper89_copies3519 a h).2.1
  have c112610 := (publicBatchWrapper89_copies3519 a h).2.2.1
  have c112611 := (publicBatchWrapper89_copies3519 a h).2.2.2.1
  have c112612 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.1
  have c112613 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_12 := arithEq_of_rows h (row := 1843) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_12
  simp only [← c112608, ← c112609, ← c112610] at e_1843_12
  have e_1843_13 := arithEq_of_rows h (row := 1843) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_13
  simp only [← c112611, ← c112612, k2, ← c112613] at e_1843_13
  have hr := e_1843_13
  simp only [e_1843_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4786)) := by
  have c112614 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.1
  have c112615 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.1
  have c112616 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.1
  have c112617 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.1
  have c112618 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112619 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_14 := arithEq_of_rows h (row := 1843) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_14
  simp only [← c112614, ← c112615, ← c112616] at e_1843_14
  have e_1843_15 := arithEq_of_rows h (row := 1843) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_15
  simp only [← c112617, ← c112618, k2, ← c112619] at e_1843_15
  have hr := e_1843_15
  simp only [e_1843_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4787)) := by
  have c112620 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112621 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112622 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112623 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112624 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112625 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_16 := arithEq_of_rows h (row := 1843) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_16
  simp only [← c112620, ← c112621, ← c112622] at e_1843_16
  have e_1843_17 := arithEq_of_rows h (row := 1843) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_17
  simp only [← c112623, ← c112624, k2, ← c112625] at e_1843_17
  have hr := e_1843_17
  simp only [e_1843_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1843 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4788)) := by
  have c112626 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112627 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112628 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112629 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112630 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112631 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1843_18 := arithEq_of_rows h (row := 1843) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_18
  simp only [← c112626, ← c112627, ← c112628] at e_1843_18
  have e_1843_19 := arithEq_of_rows h (row := 1843) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_19
  simp only [← c112629, ← c112630, k2, ← c112631] at e_1843_19
  have hr := e_1843_19
  simp only [e_1843_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4789)) := by
  have c112632 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112633 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112634 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112635 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112636 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112637 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_0 := arithEq_of_rows h (row := 1844) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_0
  simp only [← c112632, ← c112633, ← c112634] at e_1844_0
  have e_1844_1 := arithEq_of_rows h (row := 1844) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_1
  simp only [← c112635, ← c112636, k2, ← c112637] at e_1844_1
  have hr := e_1844_1
  simp only [e_1844_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4790)) := by
  have c112638 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112639 := (publicBatchWrapper89_copies3519 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112640 := (publicBatchWrapper89_copies3520 a h).1
  have c112641 := (publicBatchWrapper89_copies3520 a h).2.1
  have c112642 := (publicBatchWrapper89_copies3520 a h).2.2.1
  have c112643 := (publicBatchWrapper89_copies3520 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_2 := arithEq_of_rows h (row := 1844) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_2
  simp only [← c112638, ← c112639, ← c112640] at e_1844_2
  have e_1844_3 := arithEq_of_rows h (row := 1844) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_3
  simp only [← c112641, ← c112642, k2, ← c112643] at e_1844_3
  have hr := e_1844_3
  simp only [e_1844_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4791)) := by
  have c112644 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.1
  have c112645 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.1
  have c112646 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.1
  have c112647 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.1
  have c112648 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.1
  have c112649 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_4 := arithEq_of_rows h (row := 1844) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_4
  simp only [← c112644, ← c112645, ← c112646] at e_1844_4
  have e_1844_5 := arithEq_of_rows h (row := 1844) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_5
  simp only [← c112647, ← c112648, k2, ← c112649] at e_1844_5
  have hr := e_1844_5
  simp only [e_1844_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4792)) := by
  have c112650 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112651 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112652 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112653 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112654 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112655 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_6 := arithEq_of_rows h (row := 1844) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_6
  simp only [← c112650, ← c112651, ← c112652] at e_1844_6
  have e_1844_7 := arithEq_of_rows h (row := 1844) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_7
  simp only [← c112653, ← c112654, k2, ← c112655] at e_1844_7
  have hr := e_1844_7
  simp only [e_1844_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4793)) := by
  have c112656 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112657 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112658 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112659 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112660 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112661 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_8 := arithEq_of_rows h (row := 1844) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_8
  simp only [← c112656, ← c112657, ← c112658] at e_1844_8
  have e_1844_9 := arithEq_of_rows h (row := 1844) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_9
  simp only [← c112659, ← c112660, k2, ← c112661] at e_1844_9
  have hr := e_1844_9
  simp only [e_1844_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4794)) := by
  have c112662 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112663 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112664 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112665 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112666 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112667 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_10 := arithEq_of_rows h (row := 1844) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_10
  simp only [← c112662, ← c112663, ← c112664] at e_1844_10
  have e_1844_11 := arithEq_of_rows h (row := 1844) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_11
  simp only [← c112665, ← c112666, k2, ← c112667] at e_1844_11
  have hr := e_1844_11
  simp only [e_1844_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4795)) := by
  have c112668 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112669 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112670 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112671 := (publicBatchWrapper89_copies3520 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112672 := (publicBatchWrapper89_copies3521 a h).1
  have c112673 := (publicBatchWrapper89_copies3521 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_12 := arithEq_of_rows h (row := 1844) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_12
  simp only [← c112668, ← c112669, ← c112670] at e_1844_12
  have e_1844_13 := arithEq_of_rows h (row := 1844) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_13
  simp only [← c112671, ← c112672, k2, ← c112673] at e_1844_13
  have hr := e_1844_13
  simp only [e_1844_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4796)) := by
  have c112674 := (publicBatchWrapper89_copies3521 a h).2.2.1
  have c112675 := (publicBatchWrapper89_copies3521 a h).2.2.2.1
  have c112676 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.1
  have c112677 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.1
  have c112678 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.1
  have c112679 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_14 := arithEq_of_rows h (row := 1844) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_14
  simp only [← c112674, ← c112675, ← c112676] at e_1844_14
  have e_1844_15 := arithEq_of_rows h (row := 1844) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_15
  simp only [← c112677, ← c112678, k2, ← c112679] at e_1844_15
  have hr := e_1844_15
  simp only [e_1844_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4797)) := by
  have c112680 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.1
  have c112681 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.1
  have c112682 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112683 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112684 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112685 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_16 := arithEq_of_rows h (row := 1844) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_16
  simp only [← c112680, ← c112681, ← c112682] at e_1844_16
  have e_1844_17 := arithEq_of_rows h (row := 1844) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_17
  simp only [← c112683, ← c112684, k2, ← c112685] at e_1844_17
  have hr := e_1844_17
  simp only [e_1844_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1844 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4798)) := by
  have c112686 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112687 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112688 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112689 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112690 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112691 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1844_18 := arithEq_of_rows h (row := 1844) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_18
  simp only [← c112686, ← c112687, ← c112688] at e_1844_18
  have e_1844_19 := arithEq_of_rows h (row := 1844) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_19
  simp only [← c112689, ← c112690, k2, ← c112691] at e_1844_19
  have hr := e_1844_19
  simp only [e_1844_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4799)) := by
  have c112692 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112693 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112694 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112695 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112696 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112697 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_0 := arithEq_of_rows h (row := 1845) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_0
  simp only [← c112692, ← c112693, ← c112694] at e_1845_0
  have e_1845_1 := arithEq_of_rows h (row := 1845) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_1
  simp only [← c112695, ← c112696, k2, ← c112697] at e_1845_1
  have hr := e_1845_1
  simp only [e_1845_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4800)) := by
  have c112698 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112699 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112700 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112701 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112702 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112703 := (publicBatchWrapper89_copies3521 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_2 := arithEq_of_rows h (row := 1845) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_2
  simp only [← c112698, ← c112699, ← c112700] at e_1845_2
  have e_1845_3 := arithEq_of_rows h (row := 1845) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_3
  simp only [← c112701, ← c112702, k2, ← c112703] at e_1845_3
  have hr := e_1845_3
  simp only [e_1845_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4801)) := by
  have c112704 := (publicBatchWrapper89_copies3522 a h).1
  have c112705 := (publicBatchWrapper89_copies3522 a h).2.1
  have c112706 := (publicBatchWrapper89_copies3522 a h).2.2.1
  have c112707 := (publicBatchWrapper89_copies3522 a h).2.2.2.1
  have c112708 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.1
  have c112709 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_4 := arithEq_of_rows h (row := 1845) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_4
  simp only [← c112704, ← c112705, ← c112706] at e_1845_4
  have e_1845_5 := arithEq_of_rows h (row := 1845) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_5
  simp only [← c112707, ← c112708, k2, ← c112709] at e_1845_5
  have hr := e_1845_5
  simp only [e_1845_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4970)) := by
  have c112710 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.1
  have c112711 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.1
  have c112712 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.1
  have c112713 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.1
  have c112714 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112715 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_6 := arithEq_of_rows h (row := 1845) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_6
  simp only [← c112710, ← c112711, ← c112712] at e_1845_6
  have e_1845_7 := arithEq_of_rows h (row := 1845) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_7
  simp only [← c112713, ← c112714, k2, ← c112715] at e_1845_7
  have hr := e_1845_7
  simp only [e_1845_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4971)) := by
  have c112716 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112717 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112718 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112719 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112720 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112721 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_8 := arithEq_of_rows h (row := 1845) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_8
  simp only [← c112716, ← c112717, ← c112718] at e_1845_8
  have e_1845_9 := arithEq_of_rows h (row := 1845) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_9
  simp only [← c112719, ← c112720, k2, ← c112721] at e_1845_9
  have hr := e_1845_9
  simp only [e_1845_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4972)) := by
  have c112722 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112723 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112724 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112725 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112726 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112727 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_10 := arithEq_of_rows h (row := 1845) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_10
  simp only [← c112722, ← c112723, ← c112724] at e_1845_10
  have e_1845_11 := arithEq_of_rows h (row := 1845) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_11
  simp only [← c112725, ← c112726, k2, ← c112727] at e_1845_11
  have hr := e_1845_11
  simp only [e_1845_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4973)) := by
  have c112728 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112729 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112730 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112731 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112732 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112733 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_12 := arithEq_of_rows h (row := 1845) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_12
  simp only [← c112728, ← c112729, ← c112730] at e_1845_12
  have e_1845_13 := arithEq_of_rows h (row := 1845) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_13
  simp only [← c112731, ← c112732, k2, ← c112733] at e_1845_13
  have hr := e_1845_13
  simp only [e_1845_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4974)) := by
  have c112734 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112735 := (publicBatchWrapper89_copies3522 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112736 := (publicBatchWrapper89_copies3523 a h).1
  have c112737 := (publicBatchWrapper89_copies3523 a h).2.1
  have c112738 := (publicBatchWrapper89_copies3523 a h).2.2.1
  have c112739 := (publicBatchWrapper89_copies3523 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_14 := arithEq_of_rows h (row := 1845) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_14
  simp only [← c112734, ← c112735, ← c112736] at e_1845_14
  have e_1845_15 := arithEq_of_rows h (row := 1845) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_15
  simp only [← c112737, ← c112738, k2, ← c112739] at e_1845_15
  have hr := e_1845_15
  simp only [e_1845_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4975)) := by
  have c112740 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.1
  have c112741 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.1
  have c112742 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.1
  have c112743 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.1
  have c112744 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.1
  have c112745 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_16 := arithEq_of_rows h (row := 1845) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_16
  simp only [← c112740, ← c112741, ← c112742] at e_1845_16
  have e_1845_17 := arithEq_of_rows h (row := 1845) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_17
  simp only [← c112743, ← c112744, k2, ← c112745] at e_1845_17
  have hr := e_1845_17
  simp only [e_1845_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1845 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4976)) := by
  have c112746 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112747 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112748 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112749 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112750 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112751 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1845_18 := arithEq_of_rows h (row := 1845) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_18
  simp only [← c112746, ← c112747, ← c112748] at e_1845_18
  have e_1845_19 := arithEq_of_rows h (row := 1845) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_19
  simp only [← c112749, ← c112750, k2, ← c112751] at e_1845_19
  have hr := e_1845_19
  simp only [e_1845_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4977)) := by
  have c112752 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112753 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112754 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112755 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112756 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112757 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_0 := arithEq_of_rows h (row := 1846) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_0
  simp only [← c112752, ← c112753, ← c112754] at e_1846_0
  have e_1846_1 := arithEq_of_rows h (row := 1846) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_1
  simp only [← c112755, ← c112756, k2, ← c112757] at e_1846_1
  have hr := e_1846_1
  simp only [e_1846_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4978)) := by
  have c112758 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112759 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112760 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112761 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112762 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112763 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_2 := arithEq_of_rows h (row := 1846) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_2
  simp only [← c112758, ← c112759, ← c112760] at e_1846_2
  have e_1846_3 := arithEq_of_rows h (row := 1846) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_3
  simp only [← c112761, ← c112762, k2, ← c112763] at e_1846_3
  have hr := e_1846_3
  simp only [e_1846_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4979)) := by
  have c112764 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112765 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112766 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112767 := (publicBatchWrapper89_copies3523 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112768 := (publicBatchWrapper89_copies3524 a h).1
  have c112769 := (publicBatchWrapper89_copies3524 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_4 := arithEq_of_rows h (row := 1846) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_4
  simp only [← c112764, ← c112765, ← c112766] at e_1846_4
  have e_1846_5 := arithEq_of_rows h (row := 1846) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_5
  simp only [← c112767, ← c112768, k2, ← c112769] at e_1846_5
  have hr := e_1846_5
  simp only [e_1846_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4980)) := by
  have c112770 := (publicBatchWrapper89_copies3524 a h).2.2.1
  have c112771 := (publicBatchWrapper89_copies3524 a h).2.2.2.1
  have c112772 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.1
  have c112773 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.1
  have c112774 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.1
  have c112775 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_6 := arithEq_of_rows h (row := 1846) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_6
  simp only [← c112770, ← c112771, ← c112772] at e_1846_6
  have e_1846_7 := arithEq_of_rows h (row := 1846) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_7
  simp only [← c112773, ← c112774, k2, ← c112775] at e_1846_7
  have hr := e_1846_7
  simp only [e_1846_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4981)) := by
  have c112776 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.1
  have c112777 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.1
  have c112778 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112779 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112780 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112781 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_8 := arithEq_of_rows h (row := 1846) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_8
  simp only [← c112776, ← c112777, ← c112778] at e_1846_8
  have e_1846_9 := arithEq_of_rows h (row := 1846) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_9
  simp only [← c112779, ← c112780, k2, ← c112781] at e_1846_9
  have hr := e_1846_9
  simp only [e_1846_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4982)) := by
  have c112782 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112783 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112784 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112785 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112786 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112787 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_10 := arithEq_of_rows h (row := 1846) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_10
  simp only [← c112782, ← c112783, ← c112784] at e_1846_10
  have e_1846_11 := arithEq_of_rows h (row := 1846) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_11
  simp only [← c112785, ← c112786, k2, ← c112787] at e_1846_11
  have hr := e_1846_11
  simp only [e_1846_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4983)) := by
  have c112788 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112789 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112790 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112791 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112792 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112793 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_12 := arithEq_of_rows h (row := 1846) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_12
  simp only [← c112788, ← c112789, ← c112790] at e_1846_12
  have e_1846_13 := arithEq_of_rows h (row := 1846) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_13
  simp only [← c112791, ← c112792, k2, ← c112793] at e_1846_13
  have hr := e_1846_13
  simp only [e_1846_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4984)) := by
  have c112794 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112795 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112796 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112797 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112798 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112799 := (publicBatchWrapper89_copies3524 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_14 := arithEq_of_rows h (row := 1846) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_14
  simp only [← c112794, ← c112795, ← c112796] at e_1846_14
  have e_1846_15 := arithEq_of_rows h (row := 1846) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_15
  simp only [← c112797, ← c112798, k2, ← c112799] at e_1846_15
  have hr := e_1846_15
  simp only [e_1846_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4985)) := by
  have c112800 := (publicBatchWrapper89_copies3525 a h).1
  have c112801 := (publicBatchWrapper89_copies3525 a h).2.1
  have c112802 := (publicBatchWrapper89_copies3525 a h).2.2.1
  have c112803 := (publicBatchWrapper89_copies3525 a h).2.2.2.1
  have c112804 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.1
  have c112805 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_16 := arithEq_of_rows h (row := 1846) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_16
  simp only [← c112800, ← c112801, ← c112802] at e_1846_16
  have e_1846_17 := arithEq_of_rows h (row := 1846) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_17
  simp only [← c112803, ← c112804, k2, ← c112805] at e_1846_17
  have hr := e_1846_17
  simp only [e_1846_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1846 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4986)) := by
  have c112806 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.1
  have c112807 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.1
  have c112808 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.1
  have c112809 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.1
  have c112810 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112811 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1846_18 := arithEq_of_rows h (row := 1846) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_18
  simp only [← c112806, ← c112807, ← c112808] at e_1846_18
  have e_1846_19 := arithEq_of_rows h (row := 1846) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_19
  simp only [← c112809, ← c112810, k2, ← c112811] at e_1846_19
  have hr := e_1846_19
  simp only [e_1846_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4987)) := by
  have c112812 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112813 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112814 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112815 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112816 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112817 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_0 := arithEq_of_rows h (row := 1847) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_0
  simp only [← c112812, ← c112813, ← c112814] at e_1847_0
  have e_1847_1 := arithEq_of_rows h (row := 1847) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_1
  simp only [← c112815, ← c112816, k2, ← c112817] at e_1847_1
  have hr := e_1847_1
  simp only [e_1847_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4988)) := by
  have c112818 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112819 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112820 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112821 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112822 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112823 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_2 := arithEq_of_rows h (row := 1847) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_2
  simp only [← c112818, ← c112819, ← c112820] at e_1847_2
  have e_1847_3 := arithEq_of_rows h (row := 1847) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_3
  simp only [← c112821, ← c112822, k2, ← c112823] at e_1847_3
  have hr := e_1847_3
  simp only [e_1847_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4989)) := by
  have c112824 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112825 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112826 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112827 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112828 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112829 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_4 := arithEq_of_rows h (row := 1847) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_4
  simp only [← c112824, ← c112825, ← c112826] at e_1847_4
  have e_1847_5 := arithEq_of_rows h (row := 1847) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_5
  simp only [← c112827, ← c112828, k2, ← c112829] at e_1847_5
  have hr := e_1847_5
  simp only [e_1847_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4990)) := by
  have c112830 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112831 := (publicBatchWrapper89_copies3525 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112832 := (publicBatchWrapper89_copies3526 a h).1
  have c112833 := (publicBatchWrapper89_copies3526 a h).2.1
  have c112834 := (publicBatchWrapper89_copies3526 a h).2.2.1
  have c112835 := (publicBatchWrapper89_copies3526 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_6 := arithEq_of_rows h (row := 1847) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_6
  simp only [← c112830, ← c112831, ← c112832] at e_1847_6
  have e_1847_7 := arithEq_of_rows h (row := 1847) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_7
  simp only [← c112833, ← c112834, k2, ← c112835] at e_1847_7
  have hr := e_1847_7
  simp only [e_1847_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4991)) := by
  have c112836 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.1
  have c112837 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.1
  have c112838 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.1
  have c112839 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.1
  have c112840 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.1
  have c112841 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_8 := arithEq_of_rows h (row := 1847) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_8
  simp only [← c112836, ← c112837, ← c112838] at e_1847_8
  have e_1847_9 := arithEq_of_rows h (row := 1847) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_9
  simp only [← c112839, ← c112840, k2, ← c112841] at e_1847_9
  have hr := e_1847_9
  simp only [e_1847_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 47) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4992)) := by
  have c112842 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112843 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112844 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112845 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112846 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112847 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_10 := arithEq_of_rows h (row := 1847) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_10
  simp only [← c112842, ← c112843, ← c112844] at e_1847_10
  have e_1847_11 := arithEq_of_rows h (row := 1847) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_11
  simp only [← c112845, ← c112846, k2, ← c112847] at e_1847_11
  have hr := e_1847_11
  simp only [e_1847_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 55) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4993)) := by
  have c112848 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112849 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112850 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112851 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112852 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112853 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_12 := arithEq_of_rows h (row := 1847) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_12
  simp only [← c112848, ← c112849, ← c112850] at e_1847_12
  have e_1847_13 := arithEq_of_rows h (row := 1847) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_13
  simp only [← c112851, ← c112852, k2, ← c112853] at e_1847_13
  have hr := e_1847_13
  simp only [e_1847_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 63) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4994)) := by
  have c112854 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112855 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112856 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112857 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112858 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112859 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_14 := arithEq_of_rows h (row := 1847) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_14
  simp only [← c112854, ← c112855, ← c112856] at e_1847_14
  have e_1847_15 := arithEq_of_rows h (row := 1847) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_15
  simp only [← c112857, ← c112858, k2, ← c112859] at e_1847_15
  have hr := e_1847_15
  simp only [e_1847_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 71) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4995)) := by
  have c112860 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112861 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112862 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112863 := (publicBatchWrapper89_copies3526 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112864 := (publicBatchWrapper89_copies3527 a h).1
  have c112865 := (publicBatchWrapper89_copies3527 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_16 := arithEq_of_rows h (row := 1847) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_16
  simp only [← c112860, ← c112861, ← c112862] at e_1847_16
  have e_1847_17 := arithEq_of_rows h (row := 1847) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_17
  simp only [← c112863, ← c112864, k2, ← c112865] at e_1847_17
  have hr := e_1847_17
  simp only [e_1847_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1847 79) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4996)) := by
  have c112866 := (publicBatchWrapper89_copies3527 a h).2.2.1
  have c112867 := (publicBatchWrapper89_copies3527 a h).2.2.2.1
  have c112868 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.1
  have c112869 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.1
  have c112870 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.1
  have c112871 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1847_18 := arithEq_of_rows h (row := 1847) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_18
  simp only [← c112866, ← c112867, ← c112868] at e_1847_18
  have e_1847_19 := arithEq_of_rows h (row := 1847) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_19
  simp only [← c112869, ← c112870, k2, ← c112871] at e_1847_19
  have hr := e_1847_19
  simp only [e_1847_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 7) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4997)) := by
  have c112872 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.1
  have c112873 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.1
  have c112874 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112875 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112876 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112877 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_0 := arithEq_of_rows h (row := 1848) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_0
  simp only [← c112872, ← c112873, ← c112874] at e_1848_0
  have e_1848_1 := arithEq_of_rows h (row := 1848) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_1
  simp only [← c112875, ← c112876, k2, ← c112877] at e_1848_1
  have hr := e_1848_1
  simp only [e_1848_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 15) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4998)) := by
  have c112878 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112879 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112880 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112881 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112882 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112883 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_2 := arithEq_of_rows h (row := 1848) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_2
  simp only [← c112878, ← c112879, ← c112880] at e_1848_2
  have e_1848_3 := arithEq_of_rows h (row := 1848) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_3
  simp only [← c112881, ← c112882, k2, ← c112883] at e_1848_3
  have hr := e_1848_3
  simp only [e_1848_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 23) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 4999)) := by
  have c112884 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112885 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112886 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112887 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112888 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112889 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_4 := arithEq_of_rows h (row := 1848) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_4
  simp only [← c112884, ← c112885, ← c112886] at e_1848_4
  have e_1848_5 := arithEq_of_rows h (row := 1848) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_5
  simp only [← c112887, ← c112888, k2, ← c112889] at e_1848_5
  have hr := e_1848_5
  simp only [e_1848_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 31) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 5000)) := by
  have c112890 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112891 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112892 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112893 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112894 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112895 := (publicBatchWrapper89_copies3527 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_6 := arithEq_of_rows h (row := 1848) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_6
  simp only [← c112890, ← c112891, ← c112892] at e_1848_6
  have e_1848_7 := arithEq_of_rows h (row := 1848) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_7
  simp only [← c112893, ← c112894, k2, ← c112895] at e_1848_7
  have hr := e_1848_7
  simp only [e_1848_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 39) = bselect (a (.wire 24 75)) (a (.virt 200)) (a (.virt 5001)) := by
  have c112896 := (publicBatchWrapper89_copies3528 a h).1
  have c112897 := (publicBatchWrapper89_copies3528 a h).2.1
  have c112898 := (publicBatchWrapper89_copies3528 a h).2.2.1
  have c112899 := (publicBatchWrapper89_copies3528 a h).2.2.2.1
  have c112900 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.1
  have c112901 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_8 := arithEq_of_rows h (row := 1848) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_8
  simp only [← c112896, ← c112897, ← c112898] at e_1848_8
  have e_1848_9 := arithEq_of_rows h (row := 1848) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_9
  simp only [← c112899, ← c112900, k2, ← c112901] at e_1848_9
  have hr := e_1848_9
  simp only [e_1848_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5170)) := by
  have c112902 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.1
  have c112903 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.1
  have c112904 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.1
  have c112905 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.1
  have c112906 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112907 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_10 := arithEq_of_rows h (row := 1848) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_10
  simp only [← c112902, ← c112903, ← c112904] at e_1848_10
  have e_1848_11 := arithEq_of_rows h (row := 1848) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_11
  simp only [← c112905, ← c112906, k2, ← c112907] at e_1848_11
  have hr := e_1848_11
  simp only [e_1848_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5171)) := by
  have c112908 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112909 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112910 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112911 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112912 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112913 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_12 := arithEq_of_rows h (row := 1848) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_12
  simp only [← c112908, ← c112909, ← c112910] at e_1848_12
  have e_1848_13 := arithEq_of_rows h (row := 1848) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_13
  simp only [← c112911, ← c112912, k2, ← c112913] at e_1848_13
  have hr := e_1848_13
  simp only [e_1848_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5172)) := by
  have c112914 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112915 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112916 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112917 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112918 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112919 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_14 := arithEq_of_rows h (row := 1848) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_14
  simp only [← c112914, ← c112915, ← c112916] at e_1848_14
  have e_1848_15 := arithEq_of_rows h (row := 1848) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_15
  simp only [← c112917, ← c112918, k2, ← c112919] at e_1848_15
  have hr := e_1848_15
  simp only [e_1848_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5173)) := by
  have c112920 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112921 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112922 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112923 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112924 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112925 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_16 := arithEq_of_rows h (row := 1848) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_16
  simp only [← c112920, ← c112921, ← c112922] at e_1848_16
  have e_1848_17 := arithEq_of_rows h (row := 1848) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_17
  simp only [← c112923, ← c112924, k2, ← c112925] at e_1848_17
  have hr := e_1848_17
  simp only [e_1848_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1848 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5174)) := by
  have c112926 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112927 := (publicBatchWrapper89_copies3528 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112928 := (publicBatchWrapper89_copies3529 a h).1
  have c112929 := (publicBatchWrapper89_copies3529 a h).2.1
  have c112930 := (publicBatchWrapper89_copies3529 a h).2.2.1
  have c112931 := (publicBatchWrapper89_copies3529 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1848_18 := arithEq_of_rows h (row := 1848) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_18
  simp only [← c112926, ← c112927, ← c112928] at e_1848_18
  have e_1848_19 := arithEq_of_rows h (row := 1848) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_19
  simp only [← c112929, ← c112930, k2, ← c112931] at e_1848_19
  have hr := e_1848_19
  simp only [e_1848_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5175)) := by
  have c112932 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.1
  have c112933 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.1
  have c112934 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.1
  have c112935 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.1
  have c112936 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.1
  have c112937 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_0 := arithEq_of_rows h (row := 1849) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_0
  simp only [← c112932, ← c112933, ← c112934] at e_1849_0
  have e_1849_1 := arithEq_of_rows h (row := 1849) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_1
  simp only [← c112935, ← c112936, k2, ← c112937] at e_1849_1
  have hr := e_1849_1
  simp only [e_1849_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5176)) := by
  have c112938 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112939 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112940 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112941 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112942 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112943 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_2 := arithEq_of_rows h (row := 1849) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_2
  simp only [← c112938, ← c112939, ← c112940] at e_1849_2
  have e_1849_3 := arithEq_of_rows h (row := 1849) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_3
  simp only [← c112941, ← c112942, k2, ← c112943] at e_1849_3
  have hr := e_1849_3
  simp only [e_1849_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5177)) := by
  have c112944 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112945 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112946 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112947 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112948 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112949 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_4 := arithEq_of_rows h (row := 1849) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_4
  simp only [← c112944, ← c112945, ← c112946] at e_1849_4
  have e_1849_5 := arithEq_of_rows h (row := 1849) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_5
  simp only [← c112947, ← c112948, k2, ← c112949] at e_1849_5
  have hr := e_1849_5
  simp only [e_1849_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5178)) := by
  have c112950 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112951 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112952 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112953 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112954 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112955 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_6 := arithEq_of_rows h (row := 1849) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_6
  simp only [← c112950, ← c112951, ← c112952] at e_1849_6
  have e_1849_7 := arithEq_of_rows h (row := 1849) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_7
  simp only [← c112953, ← c112954, k2, ← c112955] at e_1849_7
  have hr := e_1849_7
  simp only [e_1849_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5179)) := by
  have c112956 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112957 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112958 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112959 := (publicBatchWrapper89_copies3529 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c112960 := (publicBatchWrapper89_copies3530 a h).1
  have c112961 := (publicBatchWrapper89_copies3530 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_8 := arithEq_of_rows h (row := 1849) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_8
  simp only [← c112956, ← c112957, ← c112958] at e_1849_8
  have e_1849_9 := arithEq_of_rows h (row := 1849) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_9
  simp only [← c112959, ← c112960, k2, ← c112961] at e_1849_9
  have hr := e_1849_9
  simp only [e_1849_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5180)) := by
  have c112962 := (publicBatchWrapper89_copies3530 a h).2.2.1
  have c112963 := (publicBatchWrapper89_copies3530 a h).2.2.2.1
  have c112964 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.1
  have c112965 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.1
  have c112966 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.1
  have c112967 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_10 := arithEq_of_rows h (row := 1849) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_10
  simp only [← c112962, ← c112963, ← c112964] at e_1849_10
  have e_1849_11 := arithEq_of_rows h (row := 1849) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_11
  simp only [← c112965, ← c112966, k2, ← c112967] at e_1849_11
  have hr := e_1849_11
  simp only [e_1849_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5181)) := by
  have c112968 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.1
  have c112969 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.1
  have c112970 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.1
  have c112971 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c112972 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112973 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_12 := arithEq_of_rows h (row := 1849) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_12
  simp only [← c112968, ← c112969, ← c112970] at e_1849_12
  have e_1849_13 := arithEq_of_rows h (row := 1849) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_13
  simp only [← c112971, ← c112972, k2, ← c112973] at e_1849_13
  have hr := e_1849_13
  simp only [e_1849_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5182)) := by
  have c112974 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112975 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112976 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112977 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112978 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112979 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_14 := arithEq_of_rows h (row := 1849) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_14
  simp only [← c112974, ← c112975, ← c112976] at e_1849_14
  have e_1849_15 := arithEq_of_rows h (row := 1849) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_15
  simp only [← c112977, ← c112978, k2, ← c112979] at e_1849_15
  have hr := e_1849_15
  simp only [e_1849_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5183)) := by
  have c112980 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112981 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112982 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112983 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112984 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112985 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_16 := arithEq_of_rows h (row := 1849) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_16
  simp only [← c112980, ← c112981, ← c112982] at e_1849_16
  have e_1849_17 := arithEq_of_rows h (row := 1849) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_17
  simp only [← c112983, ← c112984, k2, ← c112985] at e_1849_17
  have hr := e_1849_17
  simp only [e_1849_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1849 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5184)) := by
  have c112986 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112987 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112988 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112989 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112990 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c112991 := (publicBatchWrapper89_copies3530 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1849_18 := arithEq_of_rows h (row := 1849) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_18
  simp only [← c112986, ← c112987, ← c112988] at e_1849_18
  have e_1849_19 := arithEq_of_rows h (row := 1849) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_19
  simp only [← c112989, ← c112990, k2, ← c112991] at e_1849_19
  have hr := e_1849_19
  simp only [e_1849_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5185)) := by
  have c112992 := (publicBatchWrapper89_copies3531 a h).1
  have c112993 := (publicBatchWrapper89_copies3531 a h).2.1
  have c112994 := (publicBatchWrapper89_copies3531 a h).2.2.1
  have c112995 := (publicBatchWrapper89_copies3531 a h).2.2.2.1
  have c112996 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.1
  have c112997 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_0 := arithEq_of_rows h (row := 1850) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_0
  simp only [← c112992, ← c112993, ← c112994] at e_1850_0
  have e_1850_1 := arithEq_of_rows h (row := 1850) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_1
  simp only [← c112995, ← c112996, k2, ← c112997] at e_1850_1
  have hr := e_1850_1
  simp only [e_1850_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5186)) := by
  have c112998 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.1
  have c112999 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.1
  have c113000 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.1
  have c113001 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.1
  have c113002 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113003 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_2 := arithEq_of_rows h (row := 1850) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_2
  simp only [← c112998, ← c112999, ← c113000] at e_1850_2
  have e_1850_3 := arithEq_of_rows h (row := 1850) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_3
  simp only [← c113001, ← c113002, k2, ← c113003] at e_1850_3
  have hr := e_1850_3
  simp only [e_1850_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5187)) := by
  have c113004 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113005 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113006 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113007 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113008 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113009 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_4 := arithEq_of_rows h (row := 1850) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_4
  simp only [← c113004, ← c113005, ← c113006] at e_1850_4
  have e_1850_5 := arithEq_of_rows h (row := 1850) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_5
  simp only [← c113007, ← c113008, k2, ← c113009] at e_1850_5
  have hr := e_1850_5
  simp only [e_1850_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5188)) := by
  have c113010 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113011 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113012 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113013 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113014 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113015 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_6 := arithEq_of_rows h (row := 1850) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_6
  simp only [← c113010, ← c113011, ← c113012] at e_1850_6
  have e_1850_7 := arithEq_of_rows h (row := 1850) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_7
  simp only [← c113013, ← c113014, k2, ← c113015] at e_1850_7
  have hr := e_1850_7
  simp only [e_1850_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5189)) := by
  have c113016 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113017 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113018 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113019 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113020 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113021 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_8 := arithEq_of_rows h (row := 1850) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_8
  simp only [← c113016, ← c113017, ← c113018] at e_1850_8
  have e_1850_9 := arithEq_of_rows h (row := 1850) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_9
  simp only [← c113019, ← c113020, k2, ← c113021] at e_1850_9
  have hr := e_1850_9
  simp only [e_1850_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5190)) := by
  have c113022 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113023 := (publicBatchWrapper89_copies3531 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113024 := (publicBatchWrapper89_copies3532 a h).1
  have c113025 := (publicBatchWrapper89_copies3532 a h).2.1
  have c113026 := (publicBatchWrapper89_copies3532 a h).2.2.1
  have c113027 := (publicBatchWrapper89_copies3532 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_10 := arithEq_of_rows h (row := 1850) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_10
  simp only [← c113022, ← c113023, ← c113024] at e_1850_10
  have e_1850_11 := arithEq_of_rows h (row := 1850) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_11
  simp only [← c113025, ← c113026, k2, ← c113027] at e_1850_11
  have hr := e_1850_11
  simp only [e_1850_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5191)) := by
  have c113028 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.1
  have c113029 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.1
  have c113030 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.1
  have c113031 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.1
  have c113032 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.1
  have c113033 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_12 := arithEq_of_rows h (row := 1850) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_12
  simp only [← c113028, ← c113029, ← c113030] at e_1850_12
  have e_1850_13 := arithEq_of_rows h (row := 1850) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_13
  simp only [← c113031, ← c113032, k2, ← c113033] at e_1850_13
  have hr := e_1850_13
  simp only [e_1850_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 63) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5192)) := by
  have c113034 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113035 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113036 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113037 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113038 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113039 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_14 := arithEq_of_rows h (row := 1850) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_14
  simp only [← c113034, ← c113035, ← c113036] at e_1850_14
  have e_1850_15 := arithEq_of_rows h (row := 1850) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_15
  simp only [← c113037, ← c113038, k2, ← c113039] at e_1850_15
  have hr := e_1850_15
  simp only [e_1850_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 71) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5193)) := by
  have c113040 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113041 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113042 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113043 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113044 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113045 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_16 := arithEq_of_rows h (row := 1850) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_16
  simp only [← c113040, ← c113041, ← c113042] at e_1850_16
  have e_1850_17 := arithEq_of_rows h (row := 1850) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_17
  simp only [← c113043, ← c113044, k2, ← c113045] at e_1850_17
  have hr := e_1850_17
  simp only [e_1850_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1850 79) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5194)) := by
  have c113046 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113047 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113048 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113049 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113050 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113051 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1850_18 := arithEq_of_rows h (row := 1850) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_18
  simp only [← c113046, ← c113047, ← c113048] at e_1850_18
  have e_1850_19 := arithEq_of_rows h (row := 1850) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_19
  simp only [← c113049, ← c113050, k2, ← c113051] at e_1850_19
  have hr := e_1850_19
  simp only [e_1850_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 7) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5195)) := by
  have c113052 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113053 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113054 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113055 := (publicBatchWrapper89_copies3532 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113056 := (publicBatchWrapper89_copies3533 a h).1
  have c113057 := (publicBatchWrapper89_copies3533 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_0 := arithEq_of_rows h (row := 1851) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_0
  simp only [← c113052, ← c113053, ← c113054] at e_1851_0
  have e_1851_1 := arithEq_of_rows h (row := 1851) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_1
  simp only [← c113055, ← c113056, k2, ← c113057] at e_1851_1
  have hr := e_1851_1
  simp only [e_1851_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 15) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5196)) := by
  have c113058 := (publicBatchWrapper89_copies3533 a h).2.2.1
  have c113059 := (publicBatchWrapper89_copies3533 a h).2.2.2.1
  have c113060 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.1
  have c113061 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.1
  have c113062 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.1
  have c113063 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_2 := arithEq_of_rows h (row := 1851) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_2
  simp only [← c113058, ← c113059, ← c113060] at e_1851_2
  have e_1851_3 := arithEq_of_rows h (row := 1851) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_3
  simp only [← c113061, ← c113062, k2, ← c113063] at e_1851_3
  have hr := e_1851_3
  simp only [e_1851_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 23) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5197)) := by
  have c113064 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.1
  have c113065 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.1
  have c113066 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113067 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113068 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113069 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_4 := arithEq_of_rows h (row := 1851) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_4
  simp only [← c113064, ← c113065, ← c113066] at e_1851_4
  have e_1851_5 := arithEq_of_rows h (row := 1851) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_5
  simp only [← c113067, ← c113068, k2, ← c113069] at e_1851_5
  have hr := e_1851_5
  simp only [e_1851_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 31) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5198)) := by
  have c113070 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113071 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113072 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113073 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113074 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113075 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_6 := arithEq_of_rows h (row := 1851) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_6
  simp only [← c113070, ← c113071, ← c113072] at e_1851_6
  have e_1851_7 := arithEq_of_rows h (row := 1851) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_7
  simp only [← c113073, ← c113074, k2, ← c113075] at e_1851_7
  have hr := e_1851_7
  simp only [e_1851_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 39) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5199)) := by
  have c113076 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113077 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113078 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113079 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113080 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113081 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_8 := arithEq_of_rows h (row := 1851) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_8
  simp only [← c113076, ← c113077, ← c113078] at e_1851_8
  have e_1851_9 := arithEq_of_rows h (row := 1851) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_9
  simp only [← c113079, ← c113080, k2, ← c113081] at e_1851_9
  have hr := e_1851_9
  simp only [e_1851_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 47) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5200)) := by
  have c113082 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113083 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113084 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113085 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113086 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113087 := (publicBatchWrapper89_copies3533 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_10 := arithEq_of_rows h (row := 1851) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_10
  simp only [← c113082, ← c113083, ← c113084] at e_1851_10
  have e_1851_11 := arithEq_of_rows h (row := 1851) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_11
  simp only [← c113085, ← c113086, k2, ← c113087] at e_1851_11
  have hr := e_1851_11
  simp only [e_1851_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 55) = bselect (a (.wire 26 39)) (a (.virt 200)) (a (.virt 5201)) := by
  have c113088 := (publicBatchWrapper89_copies3534 a h).1
  have c113089 := (publicBatchWrapper89_copies3534 a h).2.1
  have c113090 := (publicBatchWrapper89_copies3534 a h).2.2.1
  have c113091 := (publicBatchWrapper89_copies3534 a h).2.2.2.1
  have c113092 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.1
  have c113093 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_12 := arithEq_of_rows h (row := 1851) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_12
  simp only [← c113088, ← c113089, ← c113090] at e_1851_12
  have e_1851_13 := arithEq_of_rows h (row := 1851) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_13
  simp only [← c113091, ← c113092, k2, ← c113093] at e_1851_13
  have hr := e_1851_13
  simp only [e_1851_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5370)) := by
  have c113094 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.1
  have c113095 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.1
  have c113096 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.1
  have c113097 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.1
  have c113098 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113099 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_14 := arithEq_of_rows h (row := 1851) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_14
  simp only [← c113094, ← c113095, ← c113096] at e_1851_14
  have e_1851_15 := arithEq_of_rows h (row := 1851) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_15
  simp only [← c113097, ← c113098, k2, ← c113099] at e_1851_15
  have hr := e_1851_15
  simp only [e_1851_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5371)) := by
  have c113100 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113101 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113102 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113103 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113104 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113105 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_16 := arithEq_of_rows h (row := 1851) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_16
  simp only [← c113100, ← c113101, ← c113102] at e_1851_16
  have e_1851_17 := arithEq_of_rows h (row := 1851) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_17
  simp only [← c113103, ← c113104, k2, ← c113105] at e_1851_17
  have hr := e_1851_17
  simp only [e_1851_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1851 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5372)) := by
  have c113106 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113107 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113108 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113109 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113110 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113111 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1851_18 := arithEq_of_rows h (row := 1851) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_18
  simp only [← c113106, ← c113107, ← c113108] at e_1851_18
  have e_1851_19 := arithEq_of_rows h (row := 1851) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_19
  simp only [← c113109, ← c113110, k2, ← c113111] at e_1851_19
  have hr := e_1851_19
  simp only [e_1851_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5373)) := by
  have c113112 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113113 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113114 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113115 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113116 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113117 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_0 := arithEq_of_rows h (row := 1852) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_0
  simp only [← c113112, ← c113113, ← c113114] at e_1852_0
  have e_1852_1 := arithEq_of_rows h (row := 1852) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_1
  simp only [← c113115, ← c113116, k2, ← c113117] at e_1852_1
  have hr := e_1852_1
  simp only [e_1852_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5374)) := by
  have c113118 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113119 := (publicBatchWrapper89_copies3534 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113120 := (publicBatchWrapper89_copies3535 a h).1
  have c113121 := (publicBatchWrapper89_copies3535 a h).2.1
  have c113122 := (publicBatchWrapper89_copies3535 a h).2.2.1
  have c113123 := (publicBatchWrapper89_copies3535 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_2 := arithEq_of_rows h (row := 1852) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_2
  simp only [← c113118, ← c113119, ← c113120] at e_1852_2
  have e_1852_3 := arithEq_of_rows h (row := 1852) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_3
  simp only [← c113121, ← c113122, k2, ← c113123] at e_1852_3
  have hr := e_1852_3
  simp only [e_1852_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5375)) := by
  have c113124 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.1
  have c113125 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.1
  have c113126 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.1
  have c113127 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.1
  have c113128 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.1
  have c113129 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_4 := arithEq_of_rows h (row := 1852) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_4
  simp only [← c113124, ← c113125, ← c113126] at e_1852_4
  have e_1852_5 := arithEq_of_rows h (row := 1852) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_5
  simp only [← c113127, ← c113128, k2, ← c113129] at e_1852_5
  have hr := e_1852_5
  simp only [e_1852_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5376)) := by
  have c113130 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113131 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113132 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113133 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113134 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113135 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_6 := arithEq_of_rows h (row := 1852) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_6
  simp only [← c113130, ← c113131, ← c113132] at e_1852_6
  have e_1852_7 := arithEq_of_rows h (row := 1852) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_7
  simp only [← c113133, ← c113134, k2, ← c113135] at e_1852_7
  have hr := e_1852_7
  simp only [e_1852_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5377)) := by
  have c113136 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113137 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113138 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113139 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113140 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113141 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_8 := arithEq_of_rows h (row := 1852) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_8
  simp only [← c113136, ← c113137, ← c113138] at e_1852_8
  have e_1852_9 := arithEq_of_rows h (row := 1852) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_9
  simp only [← c113139, ← c113140, k2, ← c113141] at e_1852_9
  have hr := e_1852_9
  simp only [e_1852_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5378)) := by
  have c113142 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113143 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113144 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113145 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113146 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113147 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_10 := arithEq_of_rows h (row := 1852) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_10
  simp only [← c113142, ← c113143, ← c113144] at e_1852_10
  have e_1852_11 := arithEq_of_rows h (row := 1852) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_11
  simp only [← c113145, ← c113146, k2, ← c113147] at e_1852_11
  have hr := e_1852_11
  simp only [e_1852_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5379)) := by
  have c113148 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113149 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113150 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113151 := (publicBatchWrapper89_copies3535 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113152 := (publicBatchWrapper89_copies3536 a h).1
  have c113153 := (publicBatchWrapper89_copies3536 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_12 := arithEq_of_rows h (row := 1852) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_12
  simp only [← c113148, ← c113149, ← c113150] at e_1852_12
  have e_1852_13 := arithEq_of_rows h (row := 1852) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_13
  simp only [← c113151, ← c113152, k2, ← c113153] at e_1852_13
  have hr := e_1852_13
  simp only [e_1852_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5380)) := by
  have c113154 := (publicBatchWrapper89_copies3536 a h).2.2.1
  have c113155 := (publicBatchWrapper89_copies3536 a h).2.2.2.1
  have c113156 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.1
  have c113157 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.1
  have c113158 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.1
  have c113159 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_14 := arithEq_of_rows h (row := 1852) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_14
  simp only [← c113154, ← c113155, ← c113156] at e_1852_14
  have e_1852_15 := arithEq_of_rows h (row := 1852) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_15
  simp only [← c113157, ← c113158, k2, ← c113159] at e_1852_15
  have hr := e_1852_15
  simp only [e_1852_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5381)) := by
  have c113160 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.1
  have c113161 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.1
  have c113162 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113163 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113164 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113165 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_16 := arithEq_of_rows h (row := 1852) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_16
  simp only [← c113160, ← c113161, ← c113162] at e_1852_16
  have e_1852_17 := arithEq_of_rows h (row := 1852) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_17
  simp only [← c113163, ← c113164, k2, ← c113165] at e_1852_17
  have hr := e_1852_17
  simp only [e_1852_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1852 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5382)) := by
  have c113166 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113167 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113168 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113169 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113170 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113171 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1852_18 := arithEq_of_rows h (row := 1852) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_18
  simp only [← c113166, ← c113167, ← c113168] at e_1852_18
  have e_1852_19 := arithEq_of_rows h (row := 1852) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_19
  simp only [← c113169, ← c113170, k2, ← c113171] at e_1852_19
  have hr := e_1852_19
  simp only [e_1852_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5383)) := by
  have c113172 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113173 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113174 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113175 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113176 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113177 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_0 := arithEq_of_rows h (row := 1853) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_0
  simp only [← c113172, ← c113173, ← c113174] at e_1853_0
  have e_1853_1 := arithEq_of_rows h (row := 1853) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_1
  simp only [← c113175, ← c113176, k2, ← c113177] at e_1853_1
  have hr := e_1853_1
  simp only [e_1853_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5384)) := by
  have c113178 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113179 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113180 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113181 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113182 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113183 := (publicBatchWrapper89_copies3536 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_2 := arithEq_of_rows h (row := 1853) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_2
  simp only [← c113178, ← c113179, ← c113180] at e_1853_2
  have e_1853_3 := arithEq_of_rows h (row := 1853) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_3
  simp only [← c113181, ← c113182, k2, ← c113183] at e_1853_3
  have hr := e_1853_3
  simp only [e_1853_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5385)) := by
  have c113184 := (publicBatchWrapper89_copies3537 a h).1
  have c113185 := (publicBatchWrapper89_copies3537 a h).2.1
  have c113186 := (publicBatchWrapper89_copies3537 a h).2.2.1
  have c113187 := (publicBatchWrapper89_copies3537 a h).2.2.2.1
  have c113188 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.1
  have c113189 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_4 := arithEq_of_rows h (row := 1853) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_4
  simp only [← c113184, ← c113185, ← c113186] at e_1853_4
  have e_1853_5 := arithEq_of_rows h (row := 1853) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_5
  simp only [← c113187, ← c113188, k2, ← c113189] at e_1853_5
  have hr := e_1853_5
  simp only [e_1853_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5386)) := by
  have c113190 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.1
  have c113191 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.1
  have c113192 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.1
  have c113193 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.1
  have c113194 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113195 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_6 := arithEq_of_rows h (row := 1853) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_6
  simp only [← c113190, ← c113191, ← c113192] at e_1853_6
  have e_1853_7 := arithEq_of_rows h (row := 1853) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_7
  simp only [← c113193, ← c113194, k2, ← c113195] at e_1853_7
  have hr := e_1853_7
  simp only [e_1853_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5387)) := by
  have c113196 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113197 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113198 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113199 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113200 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113201 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_8 := arithEq_of_rows h (row := 1853) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_8
  simp only [← c113196, ← c113197, ← c113198] at e_1853_8
  have e_1853_9 := arithEq_of_rows h (row := 1853) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_9
  simp only [← c113199, ← c113200, k2, ← c113201] at e_1853_9
  have hr := e_1853_9
  simp only [e_1853_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5388)) := by
  have c113202 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113203 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113204 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113205 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113206 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113207 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_10 := arithEq_of_rows h (row := 1853) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_10
  simp only [← c113202, ← c113203, ← c113204] at e_1853_10
  have e_1853_11 := arithEq_of_rows h (row := 1853) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_11
  simp only [← c113205, ← c113206, k2, ← c113207] at e_1853_11
  have hr := e_1853_11
  simp only [e_1853_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5389)) := by
  have c113208 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113209 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113210 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113211 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113212 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113213 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_12 := arithEq_of_rows h (row := 1853) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_12
  simp only [← c113208, ← c113209, ← c113210] at e_1853_12
  have e_1853_13 := arithEq_of_rows h (row := 1853) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_13
  simp only [← c113211, ← c113212, k2, ← c113213] at e_1853_13
  have hr := e_1853_13
  simp only [e_1853_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5390)) := by
  have c113214 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113215 := (publicBatchWrapper89_copies3537 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113216 := (publicBatchWrapper89_copies3538 a h).1
  have c113217 := (publicBatchWrapper89_copies3538 a h).2.1
  have c113218 := (publicBatchWrapper89_copies3538 a h).2.2.1
  have c113219 := (publicBatchWrapper89_copies3538 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_14 := arithEq_of_rows h (row := 1853) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_14
  simp only [← c113214, ← c113215, ← c113216] at e_1853_14
  have e_1853_15 := arithEq_of_rows h (row := 1853) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_15
  simp only [← c113217, ← c113218, k2, ← c113219] at e_1853_15
  have hr := e_1853_15
  simp only [e_1853_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5391)) := by
  have c113220 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.1
  have c113221 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.1
  have c113222 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.1
  have c113223 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.1
  have c113224 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.1
  have c113225 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_16 := arithEq_of_rows h (row := 1853) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_16
  simp only [← c113220, ← c113221, ← c113222] at e_1853_16
  have e_1853_17 := arithEq_of_rows h (row := 1853) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_17
  simp only [← c113223, ← c113224, k2, ← c113225] at e_1853_17
  have hr := e_1853_17
  simp only [e_1853_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1853 79) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5392)) := by
  have c113226 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113227 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113228 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113229 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113230 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113231 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1853_18 := arithEq_of_rows h (row := 1853) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_18
  simp only [← c113226, ← c113227, ← c113228] at e_1853_18
  have e_1853_19 := arithEq_of_rows h (row := 1853) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_19
  simp only [← c113229, ← c113230, k2, ← c113231] at e_1853_19
  have hr := e_1853_19
  simp only [e_1853_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 7) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5393)) := by
  have c113232 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113233 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113234 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113235 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113236 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113237 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_0 := arithEq_of_rows h (row := 1854) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_0
  simp only [← c113232, ← c113233, ← c113234] at e_1854_0
  have e_1854_1 := arithEq_of_rows h (row := 1854) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_1
  simp only [← c113235, ← c113236, k2, ← c113237] at e_1854_1
  have hr := e_1854_1
  simp only [e_1854_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 15) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5394)) := by
  have c113238 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113239 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113240 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113241 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113242 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113243 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_2 := arithEq_of_rows h (row := 1854) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_2
  simp only [← c113238, ← c113239, ← c113240] at e_1854_2
  have e_1854_3 := arithEq_of_rows h (row := 1854) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_3
  simp only [← c113241, ← c113242, k2, ← c113243] at e_1854_3
  have hr := e_1854_3
  simp only [e_1854_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 23) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5395)) := by
  have c113244 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113245 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113246 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113247 := (publicBatchWrapper89_copies3538 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113248 := (publicBatchWrapper89_copies3539 a h).1
  have c113249 := (publicBatchWrapper89_copies3539 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_4 := arithEq_of_rows h (row := 1854) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_4
  simp only [← c113244, ← c113245, ← c113246] at e_1854_4
  have e_1854_5 := arithEq_of_rows h (row := 1854) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_5
  simp only [← c113247, ← c113248, k2, ← c113249] at e_1854_5
  have hr := e_1854_5
  simp only [e_1854_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 31) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5396)) := by
  have c113250 := (publicBatchWrapper89_copies3539 a h).2.2.1
  have c113251 := (publicBatchWrapper89_copies3539 a h).2.2.2.1
  have c113252 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.1
  have c113253 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.1
  have c113254 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.1
  have c113255 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_6 := arithEq_of_rows h (row := 1854) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_6
  simp only [← c113250, ← c113251, ← c113252] at e_1854_6
  have e_1854_7 := arithEq_of_rows h (row := 1854) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_7
  simp only [← c113253, ← c113254, k2, ← c113255] at e_1854_7
  have hr := e_1854_7
  simp only [e_1854_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 39) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5397)) := by
  have c113256 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.1
  have c113257 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.1
  have c113258 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113259 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113260 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113261 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_8 := arithEq_of_rows h (row := 1854) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_8
  simp only [← c113256, ← c113257, ← c113258] at e_1854_8
  have e_1854_9 := arithEq_of_rows h (row := 1854) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_9
  simp only [← c113259, ← c113260, k2, ← c113261] at e_1854_9
  have hr := e_1854_9
  simp only [e_1854_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 47) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5398)) := by
  have c113262 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113263 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113264 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113265 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113266 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113267 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_10 := arithEq_of_rows h (row := 1854) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_10
  simp only [← c113262, ← c113263, ← c113264] at e_1854_10
  have e_1854_11 := arithEq_of_rows h (row := 1854) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_11
  simp only [← c113265, ← c113266, k2, ← c113267] at e_1854_11
  have hr := e_1854_11
  simp only [e_1854_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 55) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5399)) := by
  have c113268 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113269 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113270 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113271 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113272 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113273 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_12 := arithEq_of_rows h (row := 1854) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_12
  simp only [← c113268, ← c113269, ← c113270] at e_1854_12
  have e_1854_13 := arithEq_of_rows h (row := 1854) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_13
  simp only [← c113271, ← c113272, k2, ← c113273] at e_1854_13
  have hr := e_1854_13
  simp only [e_1854_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 63) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5400)) := by
  have c113274 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113275 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113276 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113277 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113278 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113279 := (publicBatchWrapper89_copies3539 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_14 := arithEq_of_rows h (row := 1854) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_14
  simp only [← c113274, ← c113275, ← c113276] at e_1854_14
  have e_1854_15 := arithEq_of_rows h (row := 1854) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_15
  simp only [← c113277, ← c113278, k2, ← c113279] at e_1854_15
  have hr := e_1854_15
  simp only [e_1854_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 71) = bselect (a (.wire 27 3)) (a (.virt 200)) (a (.virt 5401)) := by
  have c113280 := (publicBatchWrapper89_copies3540 a h).1
  have c113281 := (publicBatchWrapper89_copies3540 a h).2.1
  have c113282 := (publicBatchWrapper89_copies3540 a h).2.2.1
  have c113283 := (publicBatchWrapper89_copies3540 a h).2.2.2.1
  have c113284 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.1
  have c113285 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_16 := arithEq_of_rows h (row := 1854) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_16
  simp only [← c113280, ← c113281, ← c113282] at e_1854_16
  have e_1854_17 := arithEq_of_rows h (row := 1854) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_17
  simp only [← c113283, ← c113284, k2, ← c113285] at e_1854_17
  have hr := e_1854_17
  simp only [e_1854_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1854 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5570)) := by
  have c113286 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.1
  have c113287 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.1
  have c113288 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.1
  have c113289 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.1
  have c113290 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113291 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1854_18 := arithEq_of_rows h (row := 1854) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_18
  simp only [← c113286, ← c113287, ← c113288] at e_1854_18
  have e_1854_19 := arithEq_of_rows h (row := 1854) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_19
  simp only [← c113289, ← c113290, k2, ← c113291] at e_1854_19
  have hr := e_1854_19
  simp only [e_1854_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5571)) := by
  have c113292 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113293 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113294 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113295 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113296 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113297 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_0 := arithEq_of_rows h (row := 1855) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_0
  simp only [← c113292, ← c113293, ← c113294] at e_1855_0
  have e_1855_1 := arithEq_of_rows h (row := 1855) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_1
  simp only [← c113295, ← c113296, k2, ← c113297] at e_1855_1
  have hr := e_1855_1
  simp only [e_1855_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5572)) := by
  have c113298 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113299 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113300 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113301 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113302 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113303 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_2 := arithEq_of_rows h (row := 1855) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_2
  simp only [← c113298, ← c113299, ← c113300] at e_1855_2
  have e_1855_3 := arithEq_of_rows h (row := 1855) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_3
  simp only [← c113301, ← c113302, k2, ← c113303] at e_1855_3
  have hr := e_1855_3
  simp only [e_1855_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5573)) := by
  have c113304 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113305 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113306 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113307 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113308 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113309 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_4 := arithEq_of_rows h (row := 1855) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_4
  simp only [← c113304, ← c113305, ← c113306] at e_1855_4
  have e_1855_5 := arithEq_of_rows h (row := 1855) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_5
  simp only [← c113307, ← c113308, k2, ← c113309] at e_1855_5
  have hr := e_1855_5
  simp only [e_1855_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5574)) := by
  have c113310 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113311 := (publicBatchWrapper89_copies3540 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113312 := (publicBatchWrapper89_copies3541 a h).1
  have c113313 := (publicBatchWrapper89_copies3541 a h).2.1
  have c113314 := (publicBatchWrapper89_copies3541 a h).2.2.1
  have c113315 := (publicBatchWrapper89_copies3541 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_6 := arithEq_of_rows h (row := 1855) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_6
  simp only [← c113310, ← c113311, ← c113312] at e_1855_6
  have e_1855_7 := arithEq_of_rows h (row := 1855) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_7
  simp only [← c113313, ← c113314, k2, ← c113315] at e_1855_7
  have hr := e_1855_7
  simp only [e_1855_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5575)) := by
  have c113316 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.1
  have c113317 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.1
  have c113318 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.1
  have c113319 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.1
  have c113320 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.1
  have c113321 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_8 := arithEq_of_rows h (row := 1855) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_8
  simp only [← c113316, ← c113317, ← c113318] at e_1855_8
  have e_1855_9 := arithEq_of_rows h (row := 1855) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_9
  simp only [← c113319, ← c113320, k2, ← c113321] at e_1855_9
  have hr := e_1855_9
  simp only [e_1855_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5576)) := by
  have c113322 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113323 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113324 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113325 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113326 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113327 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_10 := arithEq_of_rows h (row := 1855) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_10
  simp only [← c113322, ← c113323, ← c113324] at e_1855_10
  have e_1855_11 := arithEq_of_rows h (row := 1855) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_11
  simp only [← c113325, ← c113326, k2, ← c113327] at e_1855_11
  have hr := e_1855_11
  simp only [e_1855_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5577)) := by
  have c113328 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113329 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113330 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113331 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113332 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113333 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_12 := arithEq_of_rows h (row := 1855) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_12
  simp only [← c113328, ← c113329, ← c113330] at e_1855_12
  have e_1855_13 := arithEq_of_rows h (row := 1855) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_13
  simp only [← c113331, ← c113332, k2, ← c113333] at e_1855_13
  have hr := e_1855_13
  simp only [e_1855_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5578)) := by
  have c113334 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113335 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113336 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113337 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113338 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113339 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_14 := arithEq_of_rows h (row := 1855) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_14
  simp only [← c113334, ← c113335, ← c113336] at e_1855_14
  have e_1855_15 := arithEq_of_rows h (row := 1855) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_15
  simp only [← c113337, ← c113338, k2, ← c113339] at e_1855_15
  have hr := e_1855_15
  simp only [e_1855_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5579)) := by
  have c113340 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113341 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113342 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113343 := (publicBatchWrapper89_copies3541 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113344 := (publicBatchWrapper89_copies3542 a h).1
  have c113345 := (publicBatchWrapper89_copies3542 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_16 := arithEq_of_rows h (row := 1855) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_16
  simp only [← c113340, ← c113341, ← c113342] at e_1855_16
  have e_1855_17 := arithEq_of_rows h (row := 1855) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_17
  simp only [← c113343, ← c113344, k2, ← c113345] at e_1855_17
  have hr := e_1855_17
  simp only [e_1855_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1855 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5580)) := by
  have c113346 := (publicBatchWrapper89_copies3542 a h).2.2.1
  have c113347 := (publicBatchWrapper89_copies3542 a h).2.2.2.1
  have c113348 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.1
  have c113349 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.1
  have c113350 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.1
  have c113351 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1855_18 := arithEq_of_rows h (row := 1855) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_18
  simp only [← c113346, ← c113347, ← c113348] at e_1855_18
  have e_1855_19 := arithEq_of_rows h (row := 1855) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_19
  simp only [← c113349, ← c113350, k2, ← c113351] at e_1855_19
  have hr := e_1855_19
  simp only [e_1855_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5581)) := by
  have c113352 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.1
  have c113353 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.1
  have c113354 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113355 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113356 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113357 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_0 := arithEq_of_rows h (row := 1856) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_0
  simp only [← c113352, ← c113353, ← c113354] at e_1856_0
  have e_1856_1 := arithEq_of_rows h (row := 1856) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_1
  simp only [← c113355, ← c113356, k2, ← c113357] at e_1856_1
  have hr := e_1856_1
  simp only [e_1856_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5582)) := by
  have c113358 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113359 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113360 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113361 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113362 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113363 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_2 := arithEq_of_rows h (row := 1856) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_2
  simp only [← c113358, ← c113359, ← c113360] at e_1856_2
  have e_1856_3 := arithEq_of_rows h (row := 1856) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_3
  simp only [← c113361, ← c113362, k2, ← c113363] at e_1856_3
  have hr := e_1856_3
  simp only [e_1856_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5583)) := by
  have c113364 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113365 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113366 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113367 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113368 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113369 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_4 := arithEq_of_rows h (row := 1856) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_4
  simp only [← c113364, ← c113365, ← c113366] at e_1856_4
  have e_1856_5 := arithEq_of_rows h (row := 1856) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_5
  simp only [← c113367, ← c113368, k2, ← c113369] at e_1856_5
  have hr := e_1856_5
  simp only [e_1856_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5584)) := by
  have c113370 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113371 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113372 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113373 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113374 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113375 := (publicBatchWrapper89_copies3542 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_6 := arithEq_of_rows h (row := 1856) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_6
  simp only [← c113370, ← c113371, ← c113372] at e_1856_6
  have e_1856_7 := arithEq_of_rows h (row := 1856) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_7
  simp only [← c113373, ← c113374, k2, ← c113375] at e_1856_7
  have hr := e_1856_7
  simp only [e_1856_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5585)) := by
  have c113376 := (publicBatchWrapper89_copies3543 a h).1
  have c113377 := (publicBatchWrapper89_copies3543 a h).2.1
  have c113378 := (publicBatchWrapper89_copies3543 a h).2.2.1
  have c113379 := (publicBatchWrapper89_copies3543 a h).2.2.2.1
  have c113380 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.1
  have c113381 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_8 := arithEq_of_rows h (row := 1856) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_8
  simp only [← c113376, ← c113377, ← c113378] at e_1856_8
  have e_1856_9 := arithEq_of_rows h (row := 1856) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_9
  simp only [← c113379, ← c113380, k2, ← c113381] at e_1856_9
  have hr := e_1856_9
  simp only [e_1856_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5586)) := by
  have c113382 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.1
  have c113383 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.1
  have c113384 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.1
  have c113385 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.1
  have c113386 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113387 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_10 := arithEq_of_rows h (row := 1856) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_10
  simp only [← c113382, ← c113383, ← c113384] at e_1856_10
  have e_1856_11 := arithEq_of_rows h (row := 1856) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_11
  simp only [← c113385, ← c113386, k2, ← c113387] at e_1856_11
  have hr := e_1856_11
  simp only [e_1856_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5587)) := by
  have c113388 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113389 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113390 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113391 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113392 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113393 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_12 := arithEq_of_rows h (row := 1856) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_12
  simp only [← c113388, ← c113389, ← c113390] at e_1856_12
  have e_1856_13 := arithEq_of_rows h (row := 1856) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_13
  simp only [← c113391, ← c113392, k2, ← c113393] at e_1856_13
  have hr := e_1856_13
  simp only [e_1856_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5588)) := by
  have c113394 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113395 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113396 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113397 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113398 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113399 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_14 := arithEq_of_rows h (row := 1856) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_14
  simp only [← c113394, ← c113395, ← c113396] at e_1856_14
  have e_1856_15 := arithEq_of_rows h (row := 1856) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_15
  simp only [← c113397, ← c113398, k2, ← c113399] at e_1856_15
  have hr := e_1856_15
  simp only [e_1856_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5589)) := by
  have c113400 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113401 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113402 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113403 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113404 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113405 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_16 := arithEq_of_rows h (row := 1856) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_16
  simp only [← c113400, ← c113401, ← c113402] at e_1856_16
  have e_1856_17 := arithEq_of_rows h (row := 1856) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_17
  simp only [← c113403, ← c113404, k2, ← c113405] at e_1856_17
  have hr := e_1856_17
  simp only [e_1856_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1856 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5590)) := by
  have c113406 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113407 := (publicBatchWrapper89_copies3543 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113408 := (publicBatchWrapper89_copies3544 a h).1
  have c113409 := (publicBatchWrapper89_copies3544 a h).2.1
  have c113410 := (publicBatchWrapper89_copies3544 a h).2.2.1
  have c113411 := (publicBatchWrapper89_copies3544 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1856_18 := arithEq_of_rows h (row := 1856) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_18
  simp only [← c113406, ← c113407, ← c113408] at e_1856_18
  have e_1856_19 := arithEq_of_rows h (row := 1856) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_19
  simp only [← c113409, ← c113410, k2, ← c113411] at e_1856_19
  have hr := e_1856_19
  simp only [e_1856_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5591)) := by
  have c113412 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.1
  have c113413 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.1
  have c113414 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.1
  have c113415 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.1
  have c113416 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.1
  have c113417 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_0 := arithEq_of_rows h (row := 1857) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_0
  simp only [← c113412, ← c113413, ← c113414] at e_1857_0
  have e_1857_1 := arithEq_of_rows h (row := 1857) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_1
  simp only [← c113415, ← c113416, k2, ← c113417] at e_1857_1
  have hr := e_1857_1
  simp only [e_1857_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 15) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5592)) := by
  have c113418 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113419 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113420 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113421 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113422 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113423 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_2 := arithEq_of_rows h (row := 1857) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_2
  simp only [← c113418, ← c113419, ← c113420] at e_1857_2
  have e_1857_3 := arithEq_of_rows h (row := 1857) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_3
  simp only [← c113421, ← c113422, k2, ← c113423] at e_1857_3
  have hr := e_1857_3
  simp only [e_1857_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 23) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5593)) := by
  have c113424 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113425 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113426 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113427 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113428 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113429 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_4 := arithEq_of_rows h (row := 1857) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_4
  simp only [← c113424, ← c113425, ← c113426] at e_1857_4
  have e_1857_5 := arithEq_of_rows h (row := 1857) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_5
  simp only [← c113427, ← c113428, k2, ← c113429] at e_1857_5
  have hr := e_1857_5
  simp only [e_1857_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 31) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5594)) := by
  have c113430 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113431 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113432 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113433 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113434 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113435 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_6 := arithEq_of_rows h (row := 1857) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_6
  simp only [← c113430, ← c113431, ← c113432] at e_1857_6
  have e_1857_7 := arithEq_of_rows h (row := 1857) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_7
  simp only [← c113433, ← c113434, k2, ← c113435] at e_1857_7
  have hr := e_1857_7
  simp only [e_1857_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 39) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5595)) := by
  have c113436 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113437 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113438 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113439 := (publicBatchWrapper89_copies3544 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113440 := (publicBatchWrapper89_copies3545 a h).1
  have c113441 := (publicBatchWrapper89_copies3545 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_8 := arithEq_of_rows h (row := 1857) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_8
  simp only [← c113436, ← c113437, ← c113438] at e_1857_8
  have e_1857_9 := arithEq_of_rows h (row := 1857) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_9
  simp only [← c113439, ← c113440, k2, ← c113441] at e_1857_9
  have hr := e_1857_9
  simp only [e_1857_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 47) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5596)) := by
  have c113442 := (publicBatchWrapper89_copies3545 a h).2.2.1
  have c113443 := (publicBatchWrapper89_copies3545 a h).2.2.2.1
  have c113444 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.1
  have c113445 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.1
  have c113446 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.1
  have c113447 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_10 := arithEq_of_rows h (row := 1857) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_10
  simp only [← c113442, ← c113443, ← c113444] at e_1857_10
  have e_1857_11 := arithEq_of_rows h (row := 1857) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_11
  simp only [← c113445, ← c113446, k2, ← c113447] at e_1857_11
  have hr := e_1857_11
  simp only [e_1857_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 55) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5597)) := by
  have c113448 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.1
  have c113449 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.1
  have c113450 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113451 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113452 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113453 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_12 := arithEq_of_rows h (row := 1857) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_12
  simp only [← c113448, ← c113449, ← c113450] at e_1857_12
  have e_1857_13 := arithEq_of_rows h (row := 1857) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_13
  simp only [← c113451, ← c113452, k2, ← c113453] at e_1857_13
  have hr := e_1857_13
  simp only [e_1857_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 63) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5598)) := by
  have c113454 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113455 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113456 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113457 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113458 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113459 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_14 := arithEq_of_rows h (row := 1857) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_14
  simp only [← c113454, ← c113455, ← c113456] at e_1857_14
  have e_1857_15 := arithEq_of_rows h (row := 1857) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_15
  simp only [← c113457, ← c113458, k2, ← c113459] at e_1857_15
  have hr := e_1857_15
  simp only [e_1857_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 71) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5599)) := by
  have c113460 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113461 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113462 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113463 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113464 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113465 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_16 := arithEq_of_rows h (row := 1857) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_16
  simp only [← c113460, ← c113461, ← c113462] at e_1857_16
  have e_1857_17 := arithEq_of_rows h (row := 1857) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_17
  simp only [← c113463, ← c113464, k2, ← c113465] at e_1857_17
  have hr := e_1857_17
  simp only [e_1857_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1857 79) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5600)) := by
  have c113466 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113467 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113468 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113469 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113470 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113471 := (publicBatchWrapper89_copies3545 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1857_18 := arithEq_of_rows h (row := 1857) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_18
  simp only [← c113466, ← c113467, ← c113468] at e_1857_18
  have e_1857_19 := arithEq_of_rows h (row := 1857) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_19
  simp only [← c113469, ← c113470, k2, ← c113471] at e_1857_19
  have hr := e_1857_19
  simp only [e_1857_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 7) = bselect (a (.wire 27 47)) (a (.virt 200)) (a (.virt 5601)) := by
  have c113472 := (publicBatchWrapper89_copies3546 a h).1
  have c113473 := (publicBatchWrapper89_copies3546 a h).2.1
  have c113474 := (publicBatchWrapper89_copies3546 a h).2.2.1
  have c113475 := (publicBatchWrapper89_copies3546 a h).2.2.2.1
  have c113476 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.1
  have c113477 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_0 := arithEq_of_rows h (row := 1858) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_0
  simp only [← c113472, ← c113473, ← c113474] at e_1858_0
  have e_1858_1 := arithEq_of_rows h (row := 1858) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_1
  simp only [← c113475, ← c113476, k2, ← c113477] at e_1858_1
  have hr := e_1858_1
  simp only [e_1858_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5770)) := by
  have c113478 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.1
  have c113479 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.1
  have c113480 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.1
  have c113481 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.1
  have c113482 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113483 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_2 := arithEq_of_rows h (row := 1858) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_2
  simp only [← c113478, ← c113479, ← c113480] at e_1858_2
  have e_1858_3 := arithEq_of_rows h (row := 1858) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_3
  simp only [← c113481, ← c113482, k2, ← c113483] at e_1858_3
  have hr := e_1858_3
  simp only [e_1858_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5771)) := by
  have c113484 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113485 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113486 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113487 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113488 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113489 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_4 := arithEq_of_rows h (row := 1858) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_4
  simp only [← c113484, ← c113485, ← c113486] at e_1858_4
  have e_1858_5 := arithEq_of_rows h (row := 1858) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_5
  simp only [← c113487, ← c113488, k2, ← c113489] at e_1858_5
  have hr := e_1858_5
  simp only [e_1858_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5772)) := by
  have c113490 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113491 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113492 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113493 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113494 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113495 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_6 := arithEq_of_rows h (row := 1858) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_6
  simp only [← c113490, ← c113491, ← c113492] at e_1858_6
  have e_1858_7 := arithEq_of_rows h (row := 1858) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_7
  simp only [← c113493, ← c113494, k2, ← c113495] at e_1858_7
  have hr := e_1858_7
  simp only [e_1858_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5773)) := by
  have c113496 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113497 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113498 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113499 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113500 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113501 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_8 := arithEq_of_rows h (row := 1858) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_8
  simp only [← c113496, ← c113497, ← c113498] at e_1858_8
  have e_1858_9 := arithEq_of_rows h (row := 1858) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_9
  simp only [← c113499, ← c113500, k2, ← c113501] at e_1858_9
  have hr := e_1858_9
  simp only [e_1858_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5774)) := by
  have c113502 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113503 := (publicBatchWrapper89_copies3546 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113504 := (publicBatchWrapper89_copies3547 a h).1
  have c113505 := (publicBatchWrapper89_copies3547 a h).2.1
  have c113506 := (publicBatchWrapper89_copies3547 a h).2.2.1
  have c113507 := (publicBatchWrapper89_copies3547 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_10 := arithEq_of_rows h (row := 1858) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_10
  simp only [← c113502, ← c113503, ← c113504] at e_1858_10
  have e_1858_11 := arithEq_of_rows h (row := 1858) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_11
  simp only [← c113505, ← c113506, k2, ← c113507] at e_1858_11
  have hr := e_1858_11
  simp only [e_1858_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5775)) := by
  have c113508 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.1
  have c113509 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.1
  have c113510 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.1
  have c113511 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.1
  have c113512 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.1
  have c113513 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_12 := arithEq_of_rows h (row := 1858) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_12
  simp only [← c113508, ← c113509, ← c113510] at e_1858_12
  have e_1858_13 := arithEq_of_rows h (row := 1858) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_13
  simp only [← c113511, ← c113512, k2, ← c113513] at e_1858_13
  have hr := e_1858_13
  simp only [e_1858_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5776)) := by
  have c113514 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113515 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113516 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113517 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113518 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113519 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_14 := arithEq_of_rows h (row := 1858) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_14
  simp only [← c113514, ← c113515, ← c113516] at e_1858_14
  have e_1858_15 := arithEq_of_rows h (row := 1858) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_15
  simp only [← c113517, ← c113518, k2, ← c113519] at e_1858_15
  have hr := e_1858_15
  simp only [e_1858_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5777)) := by
  have c113520 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113521 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113522 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113523 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113524 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113525 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_16 := arithEq_of_rows h (row := 1858) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_16
  simp only [← c113520, ← c113521, ← c113522] at e_1858_16
  have e_1858_17 := arithEq_of_rows h (row := 1858) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_17
  simp only [← c113523, ← c113524, k2, ← c113525] at e_1858_17
  have hr := e_1858_17
  simp only [e_1858_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1858 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5778)) := by
  have c113526 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113527 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113528 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113529 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113530 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113531 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1858_18 := arithEq_of_rows h (row := 1858) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_18
  simp only [← c113526, ← c113527, ← c113528] at e_1858_18
  have e_1858_19 := arithEq_of_rows h (row := 1858) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_19
  simp only [← c113529, ← c113530, k2, ← c113531] at e_1858_19
  have hr := e_1858_19
  simp only [e_1858_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5779)) := by
  have c113532 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113533 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113534 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113535 := (publicBatchWrapper89_copies3547 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113536 := (publicBatchWrapper89_copies3548 a h).1
  have c113537 := (publicBatchWrapper89_copies3548 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_0 := arithEq_of_rows h (row := 1859) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_0
  simp only [← c113532, ← c113533, ← c113534] at e_1859_0
  have e_1859_1 := arithEq_of_rows h (row := 1859) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_1
  simp only [← c113535, ← c113536, k2, ← c113537] at e_1859_1
  have hr := e_1859_1
  simp only [e_1859_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5780)) := by
  have c113538 := (publicBatchWrapper89_copies3548 a h).2.2.1
  have c113539 := (publicBatchWrapper89_copies3548 a h).2.2.2.1
  have c113540 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.1
  have c113541 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.1
  have c113542 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.1
  have c113543 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_2 := arithEq_of_rows h (row := 1859) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_2
  simp only [← c113538, ← c113539, ← c113540] at e_1859_2
  have e_1859_3 := arithEq_of_rows h (row := 1859) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_3
  simp only [← c113541, ← c113542, k2, ← c113543] at e_1859_3
  have hr := e_1859_3
  simp only [e_1859_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5781)) := by
  have c113544 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.1
  have c113545 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.1
  have c113546 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113547 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113548 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113549 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_4 := arithEq_of_rows h (row := 1859) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_4
  simp only [← c113544, ← c113545, ← c113546] at e_1859_4
  have e_1859_5 := arithEq_of_rows h (row := 1859) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_5
  simp only [← c113547, ← c113548, k2, ← c113549] at e_1859_5
  have hr := e_1859_5
  simp only [e_1859_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5782)) := by
  have c113550 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113551 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113552 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113553 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113554 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113555 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_6 := arithEq_of_rows h (row := 1859) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_6
  simp only [← c113550, ← c113551, ← c113552] at e_1859_6
  have e_1859_7 := arithEq_of_rows h (row := 1859) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_7
  simp only [← c113553, ← c113554, k2, ← c113555] at e_1859_7
  have hr := e_1859_7
  simp only [e_1859_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5783)) := by
  have c113556 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113557 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113558 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113559 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113560 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113561 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_8 := arithEq_of_rows h (row := 1859) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_8
  simp only [← c113556, ← c113557, ← c113558] at e_1859_8
  have e_1859_9 := arithEq_of_rows h (row := 1859) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_9
  simp only [← c113559, ← c113560, k2, ← c113561] at e_1859_9
  have hr := e_1859_9
  simp only [e_1859_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5784)) := by
  have c113562 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113563 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113564 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113565 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113566 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113567 := (publicBatchWrapper89_copies3548 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_10 := arithEq_of_rows h (row := 1859) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_10
  simp only [← c113562, ← c113563, ← c113564] at e_1859_10
  have e_1859_11 := arithEq_of_rows h (row := 1859) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_11
  simp only [← c113565, ← c113566, k2, ← c113567] at e_1859_11
  have hr := e_1859_11
  simp only [e_1859_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5785)) := by
  have c113568 := (publicBatchWrapper89_copies3549 a h).1
  have c113569 := (publicBatchWrapper89_copies3549 a h).2.1
  have c113570 := (publicBatchWrapper89_copies3549 a h).2.2.1
  have c113571 := (publicBatchWrapper89_copies3549 a h).2.2.2.1
  have c113572 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.1
  have c113573 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_12 := arithEq_of_rows h (row := 1859) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_12
  simp only [← c113568, ← c113569, ← c113570] at e_1859_12
  have e_1859_13 := arithEq_of_rows h (row := 1859) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_13
  simp only [← c113571, ← c113572, k2, ← c113573] at e_1859_13
  have hr := e_1859_13
  simp only [e_1859_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5786)) := by
  have c113574 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.1
  have c113575 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.1
  have c113576 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.1
  have c113577 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.1
  have c113578 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113579 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_14 := arithEq_of_rows h (row := 1859) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_14
  simp only [← c113574, ← c113575, ← c113576] at e_1859_14
  have e_1859_15 := arithEq_of_rows h (row := 1859) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_15
  simp only [← c113577, ← c113578, k2, ← c113579] at e_1859_15
  have hr := e_1859_15
  simp only [e_1859_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5787)) := by
  have c113580 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113581 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113582 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113583 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113584 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113585 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_16 := arithEq_of_rows h (row := 1859) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_16
  simp only [← c113580, ← c113581, ← c113582] at e_1859_16
  have e_1859_17 := arithEq_of_rows h (row := 1859) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_17
  simp only [← c113583, ← c113584, k2, ← c113585] at e_1859_17
  have hr := e_1859_17
  simp only [e_1859_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1859 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5788)) := by
  have c113586 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113587 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113588 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113589 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113590 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113591 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1859_18 := arithEq_of_rows h (row := 1859) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_18
  simp only [← c113586, ← c113587, ← c113588] at e_1859_18
  have e_1859_19 := arithEq_of_rows h (row := 1859) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_19
  simp only [← c113589, ← c113590, k2, ← c113591] at e_1859_19
  have hr := e_1859_19
  simp only [e_1859_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5789)) := by
  have c113592 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113593 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113594 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113595 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113596 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113597 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_0 := arithEq_of_rows h (row := 1860) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_0
  simp only [← c113592, ← c113593, ← c113594] at e_1860_0
  have e_1860_1 := arithEq_of_rows h (row := 1860) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_1
  simp only [← c113595, ← c113596, k2, ← c113597] at e_1860_1
  have hr := e_1860_1
  simp only [e_1860_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5790)) := by
  have c113598 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113599 := (publicBatchWrapper89_copies3549 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113600 := (publicBatchWrapper89_copies3550 a h).1
  have c113601 := (publicBatchWrapper89_copies3550 a h).2.1
  have c113602 := (publicBatchWrapper89_copies3550 a h).2.2.1
  have c113603 := (publicBatchWrapper89_copies3550 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_2 := arithEq_of_rows h (row := 1860) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_2
  simp only [← c113598, ← c113599, ← c113600] at e_1860_2
  have e_1860_3 := arithEq_of_rows h (row := 1860) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_3
  simp only [← c113601, ← c113602, k2, ← c113603] at e_1860_3
  have hr := e_1860_3
  simp only [e_1860_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5791)) := by
  have c113604 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.1
  have c113605 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.1
  have c113606 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.1
  have c113607 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.1
  have c113608 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.1
  have c113609 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_4 := arithEq_of_rows h (row := 1860) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_4
  simp only [← c113604, ← c113605, ← c113606] at e_1860_4
  have e_1860_5 := arithEq_of_rows h (row := 1860) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_5
  simp only [← c113607, ← c113608, k2, ← c113609] at e_1860_5
  have hr := e_1860_5
  simp only [e_1860_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 31) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5792)) := by
  have c113610 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113611 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113612 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113613 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113614 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113615 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_6 := arithEq_of_rows h (row := 1860) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_6
  simp only [← c113610, ← c113611, ← c113612] at e_1860_6
  have e_1860_7 := arithEq_of_rows h (row := 1860) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_7
  simp only [← c113613, ← c113614, k2, ← c113615] at e_1860_7
  have hr := e_1860_7
  simp only [e_1860_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 39) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5793)) := by
  have c113616 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113617 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113618 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113619 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113620 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113621 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_8 := arithEq_of_rows h (row := 1860) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_8
  simp only [← c113616, ← c113617, ← c113618] at e_1860_8
  have e_1860_9 := arithEq_of_rows h (row := 1860) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_9
  simp only [← c113619, ← c113620, k2, ← c113621] at e_1860_9
  have hr := e_1860_9
  simp only [e_1860_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 47) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5794)) := by
  have c113622 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113623 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113624 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113625 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113626 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113627 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_10 := arithEq_of_rows h (row := 1860) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_10
  simp only [← c113622, ← c113623, ← c113624] at e_1860_10
  have e_1860_11 := arithEq_of_rows h (row := 1860) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_11
  simp only [← c113625, ← c113626, k2, ← c113627] at e_1860_11
  have hr := e_1860_11
  simp only [e_1860_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 55) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5795)) := by
  have c113628 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113629 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113630 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113631 := (publicBatchWrapper89_copies3550 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113632 := (publicBatchWrapper89_copies3551 a h).1
  have c113633 := (publicBatchWrapper89_copies3551 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_12 := arithEq_of_rows h (row := 1860) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_12
  simp only [← c113628, ← c113629, ← c113630] at e_1860_12
  have e_1860_13 := arithEq_of_rows h (row := 1860) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_13
  simp only [← c113631, ← c113632, k2, ← c113633] at e_1860_13
  have hr := e_1860_13
  simp only [e_1860_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 63) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5796)) := by
  have c113634 := (publicBatchWrapper89_copies3551 a h).2.2.1
  have c113635 := (publicBatchWrapper89_copies3551 a h).2.2.2.1
  have c113636 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.1
  have c113637 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.1
  have c113638 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.1
  have c113639 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_14 := arithEq_of_rows h (row := 1860) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_14
  simp only [← c113634, ← c113635, ← c113636] at e_1860_14
  have e_1860_15 := arithEq_of_rows h (row := 1860) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_15
  simp only [← c113637, ← c113638, k2, ← c113639] at e_1860_15
  have hr := e_1860_15
  simp only [e_1860_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 71) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5797)) := by
  have c113640 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.1
  have c113641 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.1
  have c113642 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113643 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113644 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113645 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_16 := arithEq_of_rows h (row := 1860) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_16
  simp only [← c113640, ← c113641, ← c113642] at e_1860_16
  have e_1860_17 := arithEq_of_rows h (row := 1860) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_17
  simp only [← c113643, ← c113644, k2, ← c113645] at e_1860_17
  have hr := e_1860_17
  simp only [e_1860_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1860 79) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5798)) := by
  have c113646 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113647 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113648 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113649 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113650 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113651 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1860_18 := arithEq_of_rows h (row := 1860) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_18
  simp only [← c113646, ← c113647, ← c113648] at e_1860_18
  have e_1860_19 := arithEq_of_rows h (row := 1860) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_19
  simp only [← c113649, ← c113650, k2, ← c113651] at e_1860_19
  have hr := e_1860_19
  simp only [e_1860_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 7) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5799)) := by
  have c113652 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113653 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113654 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113655 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113656 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113657 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_0 := arithEq_of_rows h (row := 1861) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_0
  simp only [← c113652, ← c113653, ← c113654] at e_1861_0
  have e_1861_1 := arithEq_of_rows h (row := 1861) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_1
  simp only [← c113655, ← c113656, k2, ← c113657] at e_1861_1
  have hr := e_1861_1
  simp only [e_1861_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 15) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5800)) := by
  have c113658 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113659 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113660 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113661 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113662 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113663 := (publicBatchWrapper89_copies3551 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_2 := arithEq_of_rows h (row := 1861) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_2
  simp only [← c113658, ← c113659, ← c113660] at e_1861_2
  have e_1861_3 := arithEq_of_rows h (row := 1861) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_3
  simp only [← c113661, ← c113662, k2, ← c113663] at e_1861_3
  have hr := e_1861_3
  simp only [e_1861_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 23) = bselect (a (.wire 29 11)) (a (.virt 200)) (a (.virt 5801)) := by
  have c113664 := (publicBatchWrapper89_copies3552 a h).1
  have c113665 := (publicBatchWrapper89_copies3552 a h).2.1
  have c113666 := (publicBatchWrapper89_copies3552 a h).2.2.1
  have c113667 := (publicBatchWrapper89_copies3552 a h).2.2.2.1
  have c113668 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.1
  have c113669 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_4 := arithEq_of_rows h (row := 1861) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_4
  simp only [← c113664, ← c113665, ← c113666] at e_1861_4
  have e_1861_5 := arithEq_of_rows h (row := 1861) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_5
  simp only [← c113667, ← c113668, k2, ← c113669] at e_1861_5
  have hr := e_1861_5
  simp only [e_1861_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5970)) := by
  have c113670 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.1
  have c113671 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.1
  have c113672 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.1
  have c113673 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.1
  have c113674 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113675 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_6 := arithEq_of_rows h (row := 1861) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_6
  simp only [← c113670, ← c113671, ← c113672] at e_1861_6
  have e_1861_7 := arithEq_of_rows h (row := 1861) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_7
  simp only [← c113673, ← c113674, k2, ← c113675] at e_1861_7
  have hr := e_1861_7
  simp only [e_1861_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5971)) := by
  have c113676 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113677 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113678 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113679 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113680 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113681 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_8 := arithEq_of_rows h (row := 1861) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_8
  simp only [← c113676, ← c113677, ← c113678] at e_1861_8
  have e_1861_9 := arithEq_of_rows h (row := 1861) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_9
  simp only [← c113679, ← c113680, k2, ← c113681] at e_1861_9
  have hr := e_1861_9
  simp only [e_1861_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5972)) := by
  have c113682 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113683 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113684 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113685 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113686 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113687 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_10 := arithEq_of_rows h (row := 1861) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_10
  simp only [← c113682, ← c113683, ← c113684] at e_1861_10
  have e_1861_11 := arithEq_of_rows h (row := 1861) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_11
  simp only [← c113685, ← c113686, k2, ← c113687] at e_1861_11
  have hr := e_1861_11
  simp only [e_1861_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5973)) := by
  have c113688 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113689 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113690 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113691 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113692 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113693 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_12 := arithEq_of_rows h (row := 1861) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_12
  simp only [← c113688, ← c113689, ← c113690] at e_1861_12
  have e_1861_13 := arithEq_of_rows h (row := 1861) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_13
  simp only [← c113691, ← c113692, k2, ← c113693] at e_1861_13
  have hr := e_1861_13
  simp only [e_1861_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5974)) := by
  have c113694 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113695 := (publicBatchWrapper89_copies3552 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113696 := (publicBatchWrapper89_copies3553 a h).1
  have c113697 := (publicBatchWrapper89_copies3553 a h).2.1
  have c113698 := (publicBatchWrapper89_copies3553 a h).2.2.1
  have c113699 := (publicBatchWrapper89_copies3553 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_14 := arithEq_of_rows h (row := 1861) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_14
  simp only [← c113694, ← c113695, ← c113696] at e_1861_14
  have e_1861_15 := arithEq_of_rows h (row := 1861) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_15
  simp only [← c113697, ← c113698, k2, ← c113699] at e_1861_15
  have hr := e_1861_15
  simp only [e_1861_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5975)) := by
  have c113700 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.1
  have c113701 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.1
  have c113702 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.1
  have c113703 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.1
  have c113704 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.1
  have c113705 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_16 := arithEq_of_rows h (row := 1861) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_16
  simp only [← c113700, ← c113701, ← c113702] at e_1861_16
  have e_1861_17 := arithEq_of_rows h (row := 1861) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_17
  simp only [← c113703, ← c113704, k2, ← c113705] at e_1861_17
  have hr := e_1861_17
  simp only [e_1861_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1861 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5976)) := by
  have c113706 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113707 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113708 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113709 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113710 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113711 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1861_18 := arithEq_of_rows h (row := 1861) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_18
  simp only [← c113706, ← c113707, ← c113708] at e_1861_18
  have e_1861_19 := arithEq_of_rows h (row := 1861) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_19
  simp only [← c113709, ← c113710, k2, ← c113711] at e_1861_19
  have hr := e_1861_19
  simp only [e_1861_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5977)) := by
  have c113712 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113713 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113714 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113715 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113716 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113717 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_0 := arithEq_of_rows h (row := 1862) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_0
  simp only [← c113712, ← c113713, ← c113714] at e_1862_0
  have e_1862_1 := arithEq_of_rows h (row := 1862) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_1
  simp only [← c113715, ← c113716, k2, ← c113717] at e_1862_1
  have hr := e_1862_1
  simp only [e_1862_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5978)) := by
  have c113718 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113719 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113720 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113721 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113722 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113723 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_2 := arithEq_of_rows h (row := 1862) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_2
  simp only [← c113718, ← c113719, ← c113720] at e_1862_2
  have e_1862_3 := arithEq_of_rows h (row := 1862) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_3
  simp only [← c113721, ← c113722, k2, ← c113723] at e_1862_3
  have hr := e_1862_3
  simp only [e_1862_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5979)) := by
  have c113724 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113725 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113726 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113727 := (publicBatchWrapper89_copies3553 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113728 := (publicBatchWrapper89_copies3554 a h).1
  have c113729 := (publicBatchWrapper89_copies3554 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_4 := arithEq_of_rows h (row := 1862) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_4
  simp only [← c113724, ← c113725, ← c113726] at e_1862_4
  have e_1862_5 := arithEq_of_rows h (row := 1862) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_5
  simp only [← c113727, ← c113728, k2, ← c113729] at e_1862_5
  have hr := e_1862_5
  simp only [e_1862_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5980)) := by
  have c113730 := (publicBatchWrapper89_copies3554 a h).2.2.1
  have c113731 := (publicBatchWrapper89_copies3554 a h).2.2.2.1
  have c113732 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.1
  have c113733 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.1
  have c113734 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.1
  have c113735 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_6 := arithEq_of_rows h (row := 1862) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_6
  simp only [← c113730, ← c113731, ← c113732] at e_1862_6
  have e_1862_7 := arithEq_of_rows h (row := 1862) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_7
  simp only [← c113733, ← c113734, k2, ← c113735] at e_1862_7
  have hr := e_1862_7
  simp only [e_1862_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5981)) := by
  have c113736 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.1
  have c113737 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.1
  have c113738 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113739 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113740 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113741 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_8 := arithEq_of_rows h (row := 1862) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_8
  simp only [← c113736, ← c113737, ← c113738] at e_1862_8
  have e_1862_9 := arithEq_of_rows h (row := 1862) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_9
  simp only [← c113739, ← c113740, k2, ← c113741] at e_1862_9
  have hr := e_1862_9
  simp only [e_1862_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5982)) := by
  have c113742 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113743 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113744 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113745 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113746 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113747 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_10 := arithEq_of_rows h (row := 1862) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_10
  simp only [← c113742, ← c113743, ← c113744] at e_1862_10
  have e_1862_11 := arithEq_of_rows h (row := 1862) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_11
  simp only [← c113745, ← c113746, k2, ← c113747] at e_1862_11
  have hr := e_1862_11
  simp only [e_1862_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5983)) := by
  have c113748 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113749 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113750 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113751 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113752 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113753 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_12 := arithEq_of_rows h (row := 1862) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_12
  simp only [← c113748, ← c113749, ← c113750] at e_1862_12
  have e_1862_13 := arithEq_of_rows h (row := 1862) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_13
  simp only [← c113751, ← c113752, k2, ← c113753] at e_1862_13
  have hr := e_1862_13
  simp only [e_1862_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5984)) := by
  have c113754 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113755 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113756 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113757 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113758 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113759 := (publicBatchWrapper89_copies3554 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_14 := arithEq_of_rows h (row := 1862) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_14
  simp only [← c113754, ← c113755, ← c113756] at e_1862_14
  have e_1862_15 := arithEq_of_rows h (row := 1862) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_15
  simp only [← c113757, ← c113758, k2, ← c113759] at e_1862_15
  have hr := e_1862_15
  simp only [e_1862_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5985)) := by
  have c113760 := (publicBatchWrapper89_copies3555 a h).1
  have c113761 := (publicBatchWrapper89_copies3555 a h).2.1
  have c113762 := (publicBatchWrapper89_copies3555 a h).2.2.1
  have c113763 := (publicBatchWrapper89_copies3555 a h).2.2.2.1
  have c113764 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.1
  have c113765 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_16 := arithEq_of_rows h (row := 1862) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_16
  simp only [← c113760, ← c113761, ← c113762] at e_1862_16
  have e_1862_17 := arithEq_of_rows h (row := 1862) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_17
  simp only [← c113763, ← c113764, k2, ← c113765] at e_1862_17
  have hr := e_1862_17
  simp only [e_1862_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1862 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5986)) := by
  have c113766 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.1
  have c113767 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.1
  have c113768 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.1
  have c113769 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.1
  have c113770 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113771 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1862_18 := arithEq_of_rows h (row := 1862) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_18
  simp only [← c113766, ← c113767, ← c113768] at e_1862_18
  have e_1862_19 := arithEq_of_rows h (row := 1862) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_19
  simp only [← c113769, ← c113770, k2, ← c113771] at e_1862_19
  have hr := e_1862_19
  simp only [e_1862_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5987)) := by
  have c113772 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113773 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113774 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113775 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113776 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113777 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_0 := arithEq_of_rows h (row := 1863) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_0
  simp only [← c113772, ← c113773, ← c113774] at e_1863_0
  have e_1863_1 := arithEq_of_rows h (row := 1863) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_1
  simp only [← c113775, ← c113776, k2, ← c113777] at e_1863_1
  have hr := e_1863_1
  simp only [e_1863_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5988)) := by
  have c113778 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113779 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113780 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113781 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113782 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113783 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_2 := arithEq_of_rows h (row := 1863) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_2
  simp only [← c113778, ← c113779, ← c113780] at e_1863_2
  have e_1863_3 := arithEq_of_rows h (row := 1863) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_3
  simp only [← c113781, ← c113782, k2, ← c113783] at e_1863_3
  have hr := e_1863_3
  simp only [e_1863_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5989)) := by
  have c113784 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113785 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113786 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113787 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113788 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113789 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_4 := arithEq_of_rows h (row := 1863) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_4
  simp only [← c113784, ← c113785, ← c113786] at e_1863_4
  have e_1863_5 := arithEq_of_rows h (row := 1863) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_5
  simp only [← c113787, ← c113788, k2, ← c113789] at e_1863_5
  have hr := e_1863_5
  simp only [e_1863_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5990)) := by
  have c113790 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113791 := (publicBatchWrapper89_copies3555 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113792 := (publicBatchWrapper89_copies3556 a h).1
  have c113793 := (publicBatchWrapper89_copies3556 a h).2.1
  have c113794 := (publicBatchWrapper89_copies3556 a h).2.2.1
  have c113795 := (publicBatchWrapper89_copies3556 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_6 := arithEq_of_rows h (row := 1863) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_6
  simp only [← c113790, ← c113791, ← c113792] at e_1863_6
  have e_1863_7 := arithEq_of_rows h (row := 1863) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_7
  simp only [← c113793, ← c113794, k2, ← c113795] at e_1863_7
  have hr := e_1863_7
  simp only [e_1863_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5991)) := by
  have c113796 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.1
  have c113797 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.1
  have c113798 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.1
  have c113799 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.1
  have c113800 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.1
  have c113801 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_8 := arithEq_of_rows h (row := 1863) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_8
  simp only [← c113796, ← c113797, ← c113798] at e_1863_8
  have e_1863_9 := arithEq_of_rows h (row := 1863) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_9
  simp only [← c113799, ← c113800, k2, ← c113801] at e_1863_9
  have hr := e_1863_9
  simp only [e_1863_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 47) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5992)) := by
  have c113802 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113803 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113804 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113805 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113806 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113807 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_10 := arithEq_of_rows h (row := 1863) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_10
  simp only [← c113802, ← c113803, ← c113804] at e_1863_10
  have e_1863_11 := arithEq_of_rows h (row := 1863) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_11
  simp only [← c113805, ← c113806, k2, ← c113807] at e_1863_11
  have hr := e_1863_11
  simp only [e_1863_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 55) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5993)) := by
  have c113808 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113809 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113810 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113811 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113812 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113813 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_12 := arithEq_of_rows h (row := 1863) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_12
  simp only [← c113808, ← c113809, ← c113810] at e_1863_12
  have e_1863_13 := arithEq_of_rows h (row := 1863) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_13
  simp only [← c113811, ← c113812, k2, ← c113813] at e_1863_13
  have hr := e_1863_13
  simp only [e_1863_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 63) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5994)) := by
  have c113814 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113815 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113816 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113817 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113818 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113819 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_14 := arithEq_of_rows h (row := 1863) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_14
  simp only [← c113814, ← c113815, ← c113816] at e_1863_14
  have e_1863_15 := arithEq_of_rows h (row := 1863) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_15
  simp only [← c113817, ← c113818, k2, ← c113819] at e_1863_15
  have hr := e_1863_15
  simp only [e_1863_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 71) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5995)) := by
  have c113820 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113821 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113822 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113823 := (publicBatchWrapper89_copies3556 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c113824 := (publicBatchWrapper89_copies3557 a h).1
  have c113825 := (publicBatchWrapper89_copies3557 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_16 := arithEq_of_rows h (row := 1863) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_16
  simp only [← c113820, ← c113821, ← c113822] at e_1863_16
  have e_1863_17 := arithEq_of_rows h (row := 1863) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_17
  simp only [← c113823, ← c113824, k2, ← c113825] at e_1863_17
  have hr := e_1863_17
  simp only [e_1863_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1863 79) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5996)) := by
  have c113826 := (publicBatchWrapper89_copies3557 a h).2.2.1
  have c113827 := (publicBatchWrapper89_copies3557 a h).2.2.2.1
  have c113828 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.1
  have c113829 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.1
  have c113830 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.1
  have c113831 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1863_18 := arithEq_of_rows h (row := 1863) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_18
  simp only [← c113826, ← c113827, ← c113828] at e_1863_18
  have e_1863_19 := arithEq_of_rows h (row := 1863) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_19
  simp only [← c113829, ← c113830, k2, ← c113831] at e_1863_19
  have hr := e_1863_19
  simp only [e_1863_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 7) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5997)) := by
  have c113832 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.1
  have c113833 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.1
  have c113834 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113835 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c113836 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113837 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_0 := arithEq_of_rows h (row := 1864) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_0
  simp only [← c113832, ← c113833, ← c113834] at e_1864_0
  have e_1864_1 := arithEq_of_rows h (row := 1864) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_1
  simp only [← c113835, ← c113836, k2, ← c113837] at e_1864_1
  have hr := e_1864_1
  simp only [e_1864_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 15) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5998)) := by
  have c113838 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113839 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113840 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113841 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113842 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113843 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_2 := arithEq_of_rows h (row := 1864) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_2
  simp only [← c113838, ← c113839, ← c113840] at e_1864_2
  have e_1864_3 := arithEq_of_rows h (row := 1864) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_3
  simp only [← c113841, ← c113842, k2, ← c113843] at e_1864_3
  have hr := e_1864_3
  simp only [e_1864_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 23) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 5999)) := by
  have c113844 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113845 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113846 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113847 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113848 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113849 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_4 := arithEq_of_rows h (row := 1864) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_4
  simp only [← c113844, ← c113845, ← c113846] at e_1864_4
  have e_1864_5 := arithEq_of_rows h (row := 1864) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_5
  simp only [← c113847, ← c113848, k2, ← c113849] at e_1864_5
  have hr := e_1864_5
  simp only [e_1864_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 31) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 6000)) := by
  have c113850 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113851 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113852 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113853 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113854 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113855 := (publicBatchWrapper89_copies3557 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_6 := arithEq_of_rows h (row := 1864) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_6
  simp only [← c113850, ← c113851, ← c113852] at e_1864_6
  have e_1864_7 := arithEq_of_rows h (row := 1864) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_7
  simp only [← c113853, ← c113854, k2, ← c113855] at e_1864_7
  have hr := e_1864_7
  simp only [e_1864_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 39) = bselect (a (.wire 29 55)) (a (.virt 200)) (a (.virt 6001)) := by
  have c113856 := (publicBatchWrapper89_copies3558 a h).1
  have c113857 := (publicBatchWrapper89_copies3558 a h).2.1
  have c113858 := (publicBatchWrapper89_copies3558 a h).2.2.1
  have c113859 := (publicBatchWrapper89_copies3558 a h).2.2.2.1
  have c113860 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.1
  have c113861 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_8 := arithEq_of_rows h (row := 1864) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_8
  simp only [← c113856, ← c113857, ← c113858] at e_1864_8
  have e_1864_9 := arithEq_of_rows h (row := 1864) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_9
  simp only [← c113859, ← c113860, k2, ← c113861] at e_1864_9
  have hr := e_1864_9
  simp only [e_1864_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 47) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6170)) := by
  have c113862 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.1
  have c113863 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.1
  have c113864 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.1
  have c113865 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.1
  have c113866 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.1
  have c113867 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_10 := arithEq_of_rows h (row := 1864) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_10
  simp only [← c113862, ← c113863, ← c113864] at e_1864_10
  have e_1864_11 := arithEq_of_rows h (row := 1864) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_11
  simp only [← c113865, ← c113866, k2, ← c113867] at e_1864_11
  have hr := e_1864_11
  simp only [e_1864_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 55) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6171)) := by
  have c113868 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113869 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113870 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113871 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113872 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113873 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_12 := arithEq_of_rows h (row := 1864) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_12
  simp only [← c113868, ← c113869, ← c113870] at e_1864_12
  have e_1864_13 := arithEq_of_rows h (row := 1864) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_13
  simp only [← c113871, ← c113872, k2, ← c113873] at e_1864_13
  have hr := e_1864_13
  simp only [e_1864_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 63) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6172)) := by
  have c113874 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113875 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113876 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113877 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113878 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113879 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_14 := arithEq_of_rows h (row := 1864) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_14
  simp only [← c113874, ← c113875, ← c113876] at e_1864_14
  have e_1864_15 := arithEq_of_rows h (row := 1864) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_15
  simp only [← c113877, ← c113878, k2, ← c113879] at e_1864_15
  have hr := e_1864_15
  simp only [e_1864_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1864 71) = bselect (a (.wire 31 19)) (a (.virt 200)) (a (.virt 6173)) := by
  have c113880 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113881 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113882 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113883 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113884 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c113885 := (publicBatchWrapper89_copies3558 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1864_16 := arithEq_of_rows h (row := 1864) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_16
  simp only [← c113880, ← c113881, ← c113882] at e_1864_16
  have e_1864_17 := arithEq_of_rows h (row := 1864) (i := 17) rfl (by decide)
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
  have e_1864_18 := arithEq_of_rows h (row := 1864) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_18
  simp only [← c113886, ← c113887, ← c113888] at e_1864_18
  have e_1864_19 := arithEq_of_rows h (row := 1864) (i := 19) rfl (by decide)
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
  have e_1865_0 := arithEq_of_rows h (row := 1865) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_0
  simp only [← c113892, ← c113893, ← c113894] at e_1865_0
  have e_1865_1 := arithEq_of_rows h (row := 1865) (i := 1) rfl (by decide)
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
  have e_1865_2 := arithEq_of_rows h (row := 1865) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_2
  simp only [← c113898, ← c113899, ← c113900] at e_1865_2
  have e_1865_3 := arithEq_of_rows h (row := 1865) (i := 3) rfl (by decide)
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
  have e_1865_4 := arithEq_of_rows h (row := 1865) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_4
  simp only [← c113904, ← c113905, ← c113906] at e_1865_4
  have e_1865_5 := arithEq_of_rows h (row := 1865) (i := 5) rfl (by decide)
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
  have e_1865_6 := arithEq_of_rows h (row := 1865) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_6
  simp only [← c113910, ← c113911, ← c113912] at e_1865_6
  have e_1865_7 := arithEq_of_rows h (row := 1865) (i := 7) rfl (by decide)
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
  have e_1865_8 := arithEq_of_rows h (row := 1865) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_8
  simp only [← c113916, ← c113917, ← c113918] at e_1865_8
  have e_1865_9 := arithEq_of_rows h (row := 1865) (i := 9) rfl (by decide)
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
  have e_1865_10 := arithEq_of_rows h (row := 1865) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_10
  simp only [← c113922, ← c113923, ← c113924] at e_1865_10
  have e_1865_11 := arithEq_of_rows h (row := 1865) (i := 11) rfl (by decide)
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
  have e_1865_12 := arithEq_of_rows h (row := 1865) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_12
  simp only [← c113928, ← c113929, ← c113930] at e_1865_12
  have e_1865_13 := arithEq_of_rows h (row := 1865) (i := 13) rfl (by decide)
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
  have e_1865_14 := arithEq_of_rows h (row := 1865) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_14
  simp only [← c113934, ← c113935, ← c113936] at e_1865_14
  have e_1865_15 := arithEq_of_rows h (row := 1865) (i := 15) rfl (by decide)
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
  have e_1865_16 := arithEq_of_rows h (row := 1865) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_16
  simp only [← c113940, ← c113941, ← c113942] at e_1865_16
  have e_1865_17 := arithEq_of_rows h (row := 1865) (i := 17) rfl (by decide)
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
  have e_1865_18 := arithEq_of_rows h (row := 1865) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1865_18
  simp only [← c113946, ← c113947, ← c113948] at e_1865_18
  have e_1865_19 := arithEq_of_rows h (row := 1865) (i := 19) rfl (by decide)
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
  have e_1866_0 := arithEq_of_rows h (row := 1866) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_0
  simp only [← c113952, ← c113953, ← c113954] at e_1866_0
  have e_1866_1 := arithEq_of_rows h (row := 1866) (i := 1) rfl (by decide)
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
  have e_1866_2 := arithEq_of_rows h (row := 1866) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_2
  simp only [← c113958, ← c113959, ← c113960] at e_1866_2
  have e_1866_3 := arithEq_of_rows h (row := 1866) (i := 3) rfl (by decide)
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
  have e_1866_4 := arithEq_of_rows h (row := 1866) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_4
  simp only [← c113964, ← c113965, ← c113966] at e_1866_4
  have e_1866_5 := arithEq_of_rows h (row := 1866) (i := 5) rfl (by decide)
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
  have e_1866_6 := arithEq_of_rows h (row := 1866) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_6
  simp only [← c113970, ← c113971, ← c113972] at e_1866_6
  have e_1866_7 := arithEq_of_rows h (row := 1866) (i := 7) rfl (by decide)
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
  have e_1866_8 := arithEq_of_rows h (row := 1866) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_8
  simp only [← c113976, ← c113977, ← c113978] at e_1866_8
  have e_1866_9 := arithEq_of_rows h (row := 1866) (i := 9) rfl (by decide)
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
  have e_1866_10 := arithEq_of_rows h (row := 1866) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_10
  simp only [← c113982, ← c113983, ← c113984] at e_1866_10
  have e_1866_11 := arithEq_of_rows h (row := 1866) (i := 11) rfl (by decide)
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
  have e_1866_12 := arithEq_of_rows h (row := 1866) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_12
  simp only [← c113988, ← c113989, ← c113990] at e_1866_12
  have e_1866_13 := arithEq_of_rows h (row := 1866) (i := 13) rfl (by decide)
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
  have e_1866_14 := arithEq_of_rows h (row := 1866) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_14
  simp only [← c113994, ← c113995, ← c113996] at e_1866_14
  have e_1866_15 := arithEq_of_rows h (row := 1866) (i := 15) rfl (by decide)
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
  have e_1866_16 := arithEq_of_rows h (row := 1866) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_16
  simp only [← c114000, ← c114001, ← c114002] at e_1866_16
  have e_1866_17 := arithEq_of_rows h (row := 1866) (i := 17) rfl (by decide)
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
  have e_1866_18 := arithEq_of_rows h (row := 1866) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1866_18
  simp only [← c114006, ← c114007, ← c114008] at e_1866_18
  have e_1866_19 := arithEq_of_rows h (row := 1866) (i := 19) rfl (by decide)
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
  have e_1867_0 := arithEq_of_rows h (row := 1867) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_0
  simp only [← c114012, ← c114013, ← c114014] at e_1867_0
  have e_1867_1 := arithEq_of_rows h (row := 1867) (i := 1) rfl (by decide)
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
  have e_1867_2 := arithEq_of_rows h (row := 1867) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_2
  simp only [← c114018, ← c114019, ← c114020] at e_1867_2
  have e_1867_3 := arithEq_of_rows h (row := 1867) (i := 3) rfl (by decide)
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
  have e_1867_4 := arithEq_of_rows h (row := 1867) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_4
  simp only [← c114024, ← c114025, ← c114026] at e_1867_4
  have e_1867_5 := arithEq_of_rows h (row := 1867) (i := 5) rfl (by decide)
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
  have e_1867_6 := arithEq_of_rows h (row := 1867) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_6
  simp only [← c114030, ← c114031, ← c114032] at e_1867_6
  have e_1867_7 := arithEq_of_rows h (row := 1867) (i := 7) rfl (by decide)
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
  have e_1867_8 := arithEq_of_rows h (row := 1867) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_8
  simp only [← c114036, ← c114037, ← c114038] at e_1867_8
  have e_1867_9 := arithEq_of_rows h (row := 1867) (i := 9) rfl (by decide)
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
  have e_1867_10 := arithEq_of_rows h (row := 1867) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_10
  simp only [← c114042, ← c114043, ← c114044] at e_1867_10
  have e_1867_11 := arithEq_of_rows h (row := 1867) (i := 11) rfl (by decide)
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
  have e_1867_12 := arithEq_of_rows h (row := 1867) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_12
  simp only [← c114048, ← c114049, ← c114050] at e_1867_12
  have e_1867_13 := arithEq_of_rows h (row := 1867) (i := 13) rfl (by decide)
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
  have e_1867_14 := arithEq_of_rows h (row := 1867) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_14
  simp only [← c114054, ← c114055, ← c114056] at e_1867_14
  have e_1867_15 := arithEq_of_rows h (row := 1867) (i := 15) rfl (by decide)
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
  have e_1867_16 := arithEq_of_rows h (row := 1867) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_16
  simp only [← c114060, ← c114061, ← c114062] at e_1867_16
  have e_1867_17 := arithEq_of_rows h (row := 1867) (i := 17) rfl (by decide)
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
  have e_1867_18 := arithEq_of_rows h (row := 1867) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1867_18
  simp only [← c114066, ← c114067, ← c114068] at e_1867_18
  have e_1867_19 := arithEq_of_rows h (row := 1867) (i := 19) rfl (by decide)
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
  have e_1868_0 := arithEq_of_rows h (row := 1868) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_0
  simp only [← c114072, ← c114073, ← c114074] at e_1868_0
  have e_1868_1 := arithEq_of_rows h (row := 1868) (i := 1) rfl (by decide)
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
  have e_1868_2 := arithEq_of_rows h (row := 1868) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_2
  simp only [← c114078, ← c114079, ← c114080] at e_1868_2
  have e_1868_3 := arithEq_of_rows h (row := 1868) (i := 3) rfl (by decide)
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
  have e_1868_4 := arithEq_of_rows h (row := 1868) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_4
  simp only [← c114084, ← c114085, ← c114086] at e_1868_4
  have e_1868_5 := arithEq_of_rows h (row := 1868) (i := 5) rfl (by decide)
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
  have e_1868_6 := arithEq_of_rows h (row := 1868) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_6
  simp only [← c114090, ← c114091, ← c114092] at e_1868_6
  have e_1868_7 := arithEq_of_rows h (row := 1868) (i := 7) rfl (by decide)
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
  have e_1868_8 := arithEq_of_rows h (row := 1868) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_8
  simp only [← c114096, ← c114097, ← c114098] at e_1868_8
  have e_1868_9 := arithEq_of_rows h (row := 1868) (i := 9) rfl (by decide)
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
  have e_1868_10 := arithEq_of_rows h (row := 1868) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_10
  simp only [← c114102, ← c114103, ← c114104] at e_1868_10
  have e_1868_11 := arithEq_of_rows h (row := 1868) (i := 11) rfl (by decide)
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
  have e_1868_12 := arithEq_of_rows h (row := 1868) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_12
  simp only [← c114108, ← c114109, ← c114110] at e_1868_12
  have e_1868_13 := arithEq_of_rows h (row := 1868) (i := 13) rfl (by decide)
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
  have e_1868_14 := arithEq_of_rows h (row := 1868) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_14
  simp only [← c114114, ← c114115, ← c114116] at e_1868_14
  have e_1868_15 := arithEq_of_rows h (row := 1868) (i := 15) rfl (by decide)
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
  have e_1868_16 := arithEq_of_rows h (row := 1868) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_16
  simp only [← c114120, ← c114121, ← c114122] at e_1868_16
  have e_1868_17 := arithEq_of_rows h (row := 1868) (i := 17) rfl (by decide)
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
  have e_1868_18 := arithEq_of_rows h (row := 1868) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1868_18
  simp only [← c114126, ← c114127, ← c114128] at e_1868_18
  have e_1868_19 := arithEq_of_rows h (row := 1868) (i := 19) rfl (by decide)
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
  have e_1869_0 := arithEq_of_rows h (row := 1869) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_0
  simp only [← c114132, ← c114133, ← c114134] at e_1869_0
  have e_1869_1 := arithEq_of_rows h (row := 1869) (i := 1) rfl (by decide)
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
  have e_1869_2 := arithEq_of_rows h (row := 1869) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_2
  simp only [← c114138, ← c114139, ← c114140] at e_1869_2
  have e_1869_3 := arithEq_of_rows h (row := 1869) (i := 3) rfl (by decide)
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
  have e_1869_4 := arithEq_of_rows h (row := 1869) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_4
  simp only [← c114144, ← c114145, ← c114146] at e_1869_4
  have e_1869_5 := arithEq_of_rows h (row := 1869) (i := 5) rfl (by decide)
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
  have e_1869_6 := arithEq_of_rows h (row := 1869) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_6
  simp only [← c114150, ← c114151, ← c114152] at e_1869_6
  have e_1869_7 := arithEq_of_rows h (row := 1869) (i := 7) rfl (by decide)
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
  have e_1869_8 := arithEq_of_rows h (row := 1869) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_8
  simp only [← c114156, ← c114157, ← c114158] at e_1869_8
  have e_1869_9 := arithEq_of_rows h (row := 1869) (i := 9) rfl (by decide)
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
  have e_1869_10 := arithEq_of_rows h (row := 1869) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_10
  simp only [← c114162, ← c114163, ← c114164] at e_1869_10
  have e_1869_11 := arithEq_of_rows h (row := 1869) (i := 11) rfl (by decide)
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
  have e_1869_12 := arithEq_of_rows h (row := 1869) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_12
  simp only [← c114168, ← c114169, ← c114170] at e_1869_12
  have e_1869_13 := arithEq_of_rows h (row := 1869) (i := 13) rfl (by decide)
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
  have e_1869_14 := arithEq_of_rows h (row := 1869) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_14
  simp only [← c114174, ← c114175, ← c114176] at e_1869_14
  have e_1869_15 := arithEq_of_rows h (row := 1869) (i := 15) rfl (by decide)
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
  have e_1869_16 := arithEq_of_rows h (row := 1869) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_16
  simp only [← c114180, ← c114181, ← c114182] at e_1869_16
  have e_1869_17 := arithEq_of_rows h (row := 1869) (i := 17) rfl (by decide)
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
  have e_1869_18 := arithEq_of_rows h (row := 1869) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1869_18
  simp only [← c114186, ← c114187, ← c114188] at e_1869_18
  have e_1869_19 := arithEq_of_rows h (row := 1869) (i := 19) rfl (by decide)
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
  have e_1870_0 := arithEq_of_rows h (row := 1870) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_0
  simp only [← c114192, ← c114193, ← c114194] at e_1870_0
  have e_1870_1 := arithEq_of_rows h (row := 1870) (i := 1) rfl (by decide)
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
  have e_1870_2 := arithEq_of_rows h (row := 1870) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_2
  simp only [← c114198, ← c114199, ← c114200] at e_1870_2
  have e_1870_3 := arithEq_of_rows h (row := 1870) (i := 3) rfl (by decide)
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
  have e_1870_4 := arithEq_of_rows h (row := 1870) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_4
  simp only [← c114204, ← c114205, ← c114206] at e_1870_4
  have e_1870_5 := arithEq_of_rows h (row := 1870) (i := 5) rfl (by decide)
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
  have e_1870_6 := arithEq_of_rows h (row := 1870) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_6
  simp only [← c114210, ← c114211, ← c114212] at e_1870_6
  have e_1870_7 := arithEq_of_rows h (row := 1870) (i := 7) rfl (by decide)
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
  have e_1870_8 := arithEq_of_rows h (row := 1870) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_8
  simp only [← c114216, ← c114217, ← c114218] at e_1870_8
  have e_1870_9 := arithEq_of_rows h (row := 1870) (i := 9) rfl (by decide)
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
  have e_1870_10 := arithEq_of_rows h (row := 1870) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_10
  simp only [← c114222, ← c114223, ← c114224] at e_1870_10
  have e_1870_11 := arithEq_of_rows h (row := 1870) (i := 11) rfl (by decide)
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
  have e_1870_12 := arithEq_of_rows h (row := 1870) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_12
  simp only [← c114228, ← c114229, ← c114230] at e_1870_12
  have e_1870_13 := arithEq_of_rows h (row := 1870) (i := 13) rfl (by decide)
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
  have e_1870_14 := arithEq_of_rows h (row := 1870) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_14
  simp only [← c114234, ← c114235, ← c114236] at e_1870_14
  have e_1870_15 := arithEq_of_rows h (row := 1870) (i := 15) rfl (by decide)
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
  have e_1870_16 := arithEq_of_rows h (row := 1870) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_16
  simp only [← c114240, ← c114241, ← c114242] at e_1870_16
  have e_1870_17 := arithEq_of_rows h (row := 1870) (i := 17) rfl (by decide)
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
  have e_1870_18 := arithEq_of_rows h (row := 1870) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1870_18
  simp only [← c114246, ← c114247, ← c114248] at e_1870_18
  have e_1870_19 := arithEq_of_rows h (row := 1870) (i := 19) rfl (by decide)
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
  have e_1871_0 := arithEq_of_rows h (row := 1871) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_0
  simp only [← c114252, ← c114253, ← c114254] at e_1871_0
  have e_1871_1 := arithEq_of_rows h (row := 1871) (i := 1) rfl (by decide)
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
  have e_1871_2 := arithEq_of_rows h (row := 1871) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_2
  simp only [← c114258, ← c114259, ← c114260] at e_1871_2
  have e_1871_3 := arithEq_of_rows h (row := 1871) (i := 3) rfl (by decide)
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
  have e_1871_4 := arithEq_of_rows h (row := 1871) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_4
  simp only [← c114264, ← c114265, ← c114266] at e_1871_4
  have e_1871_5 := arithEq_of_rows h (row := 1871) (i := 5) rfl (by decide)
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
  have e_1871_6 := arithEq_of_rows h (row := 1871) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_6
  simp only [← c114270, ← c114271, ← c114272] at e_1871_6
  have e_1871_7 := arithEq_of_rows h (row := 1871) (i := 7) rfl (by decide)
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
  have e_1871_8 := arithEq_of_rows h (row := 1871) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_8
  simp only [← c114276, ← c114277, ← c114278] at e_1871_8
  have e_1871_9 := arithEq_of_rows h (row := 1871) (i := 9) rfl (by decide)
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
  have e_1871_10 := arithEq_of_rows h (row := 1871) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_10
  simp only [← c114282, ← c114283, ← c114284] at e_1871_10
  have e_1871_11 := arithEq_of_rows h (row := 1871) (i := 11) rfl (by decide)
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
  have e_1871_12 := arithEq_of_rows h (row := 1871) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_12
  simp only [← c114288, ← c114289, ← c114290] at e_1871_12
  have e_1871_13 := arithEq_of_rows h (row := 1871) (i := 13) rfl (by decide)
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
  have e_1871_14 := arithEq_of_rows h (row := 1871) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_14
  simp only [← c114294, ← c114295, ← c114296] at e_1871_14
  have e_1871_15 := arithEq_of_rows h (row := 1871) (i := 15) rfl (by decide)
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
  have e_1871_16 := arithEq_of_rows h (row := 1871) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_16
  simp only [← c114300, ← c114301, ← c114302] at e_1871_16
  have e_1871_17 := arithEq_of_rows h (row := 1871) (i := 17) rfl (by decide)
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
  have e_1871_18 := arithEq_of_rows h (row := 1871) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1871_18
  simp only [← c114306, ← c114307, ← c114308] at e_1871_18
  have e_1871_19 := arithEq_of_rows h (row := 1871) (i := 19) rfl (by decide)
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
  have e_1872_0 := arithEq_of_rows h (row := 1872) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_0
  simp only [← c114312, ← c114313, ← c114314] at e_1872_0
  have e_1872_1 := arithEq_of_rows h (row := 1872) (i := 1) rfl (by decide)
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
  have e_1872_2 := arithEq_of_rows h (row := 1872) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_2
  simp only [← c114318, ← c114319, ← c114320] at e_1872_2
  have e_1872_3 := arithEq_of_rows h (row := 1872) (i := 3) rfl (by decide)
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
  have e_1872_4 := arithEq_of_rows h (row := 1872) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_4
  simp only [← c114324, ← c114325, ← c114326] at e_1872_4
  have e_1872_5 := arithEq_of_rows h (row := 1872) (i := 5) rfl (by decide)
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
  have e_1872_6 := arithEq_of_rows h (row := 1872) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_6
  simp only [← c114330, ← c114331, ← c114332] at e_1872_6
  have e_1872_7 := arithEq_of_rows h (row := 1872) (i := 7) rfl (by decide)
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
  have e_1872_8 := arithEq_of_rows h (row := 1872) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_8
  simp only [← c114336, ← c114337, ← c114338] at e_1872_8
  have e_1872_9 := arithEq_of_rows h (row := 1872) (i := 9) rfl (by decide)
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
  have e_1872_10 := arithEq_of_rows h (row := 1872) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_10
  simp only [← c114342, ← c114343, ← c114344] at e_1872_10
  have e_1872_11 := arithEq_of_rows h (row := 1872) (i := 11) rfl (by decide)
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
  have e_1872_12 := arithEq_of_rows h (row := 1872) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_12
  simp only [← c114348, ← c114349, ← c114350] at e_1872_12
  have e_1872_13 := arithEq_of_rows h (row := 1872) (i := 13) rfl (by decide)
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
  have e_1872_14 := arithEq_of_rows h (row := 1872) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_14
  simp only [← c114354, ← c114355, ← c114356] at e_1872_14
  have e_1872_15 := arithEq_of_rows h (row := 1872) (i := 15) rfl (by decide)
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
  have e_1872_16 := arithEq_of_rows h (row := 1872) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_16
  simp only [← c114360, ← c114361, ← c114362] at e_1872_16
  have e_1872_17 := arithEq_of_rows h (row := 1872) (i := 17) rfl (by decide)
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
  have e_1872_18 := arithEq_of_rows h (row := 1872) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1872_18
  simp only [← c114366, ← c114367, ← c114368] at e_1872_18
  have e_1872_19 := arithEq_of_rows h (row := 1872) (i := 19) rfl (by decide)
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
  have e_1873_0 := arithEq_of_rows h (row := 1873) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_0
  simp only [← c114372, ← c114373, ← c114374] at e_1873_0
  have e_1873_1 := arithEq_of_rows h (row := 1873) (i := 1) rfl (by decide)
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
  have e_1873_2 := arithEq_of_rows h (row := 1873) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_2
  simp only [← c114378, ← c114379, ← c114380] at e_1873_2
  have e_1873_3 := arithEq_of_rows h (row := 1873) (i := 3) rfl (by decide)
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
  have e_1873_4 := arithEq_of_rows h (row := 1873) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_4
  simp only [← c114384, ← c114385, ← c114386] at e_1873_4
  have e_1873_5 := arithEq_of_rows h (row := 1873) (i := 5) rfl (by decide)
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
  have e_1873_6 := arithEq_of_rows h (row := 1873) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_6
  simp only [← c114390, ← c114391, ← c114392] at e_1873_6
  have e_1873_7 := arithEq_of_rows h (row := 1873) (i := 7) rfl (by decide)
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
  have e_1873_8 := arithEq_of_rows h (row := 1873) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_8
  simp only [← c114396, ← c114397, ← c114398] at e_1873_8
  have e_1873_9 := arithEq_of_rows h (row := 1873) (i := 9) rfl (by decide)
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
  have e_1873_10 := arithEq_of_rows h (row := 1873) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_10
  simp only [← c114402, ← c114403, ← c114404] at e_1873_10
  have e_1873_11 := arithEq_of_rows h (row := 1873) (i := 11) rfl (by decide)
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
  have e_1873_12 := arithEq_of_rows h (row := 1873) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_12
  simp only [← c114408, ← c114409, ← c114410] at e_1873_12
  have e_1873_13 := arithEq_of_rows h (row := 1873) (i := 13) rfl (by decide)
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
  have e_1873_14 := arithEq_of_rows h (row := 1873) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_14
  simp only [← c114414, ← c114415, ← c114416] at e_1873_14
  have e_1873_15 := arithEq_of_rows h (row := 1873) (i := 15) rfl (by decide)
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
  have e_1873_16 := arithEq_of_rows h (row := 1873) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_16
  simp only [← c114420, ← c114421, ← c114422] at e_1873_16
  have e_1873_17 := arithEq_of_rows h (row := 1873) (i := 17) rfl (by decide)
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
  have e_1873_18 := arithEq_of_rows h (row := 1873) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1873_18
  simp only [← c114426, ← c114427, ← c114428] at e_1873_18
  have e_1873_19 := arithEq_of_rows h (row := 1873) (i := 19) rfl (by decide)
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
  have e_1874_0 := arithEq_of_rows h (row := 1874) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_0
  simp only [← c114432, ← c114433, ← c114434] at e_1874_0
  have e_1874_1 := arithEq_of_rows h (row := 1874) (i := 1) rfl (by decide)
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
  have e_1874_2 := arithEq_of_rows h (row := 1874) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_2
  simp only [← c114438, ← c114439, ← c114440] at e_1874_2
  have e_1874_3 := arithEq_of_rows h (row := 1874) (i := 3) rfl (by decide)
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
  have e_1874_4 := arithEq_of_rows h (row := 1874) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_4
  simp only [← c114444, ← c114445, ← c114446] at e_1874_4
  have e_1874_5 := arithEq_of_rows h (row := 1874) (i := 5) rfl (by decide)
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
  have e_1874_6 := arithEq_of_rows h (row := 1874) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_6
  simp only [← c114450, ← c114451, ← c114452] at e_1874_6
  have e_1874_7 := arithEq_of_rows h (row := 1874) (i := 7) rfl (by decide)
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
  have e_1874_8 := arithEq_of_rows h (row := 1874) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_8
  simp only [← c114456, ← c114457, ← c114458] at e_1874_8
  have e_1874_9 := arithEq_of_rows h (row := 1874) (i := 9) rfl (by decide)
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
  have e_1874_10 := arithEq_of_rows h (row := 1874) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_10
  simp only [← c114462, ← c114463, ← c114464] at e_1874_10
  have e_1874_11 := arithEq_of_rows h (row := 1874) (i := 11) rfl (by decide)
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
  have e_1874_12 := arithEq_of_rows h (row := 1874) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_12
  simp only [← c114468, ← c114469, ← c114470] at e_1874_12
  have e_1874_13 := arithEq_of_rows h (row := 1874) (i := 13) rfl (by decide)
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
  have e_1874_14 := arithEq_of_rows h (row := 1874) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_14
  simp only [← c114474, ← c114475, ← c114476] at e_1874_14
  have e_1874_15 := arithEq_of_rows h (row := 1874) (i := 15) rfl (by decide)
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
  have e_1874_16 := arithEq_of_rows h (row := 1874) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_16
  simp only [← c114480, ← c114481, ← c114482] at e_1874_16
  have e_1874_17 := arithEq_of_rows h (row := 1874) (i := 17) rfl (by decide)
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
  have e_1874_18 := arithEq_of_rows h (row := 1874) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1874_18
  simp only [← c114486, ← c114487, ← c114488] at e_1874_18
  have e_1874_19 := arithEq_of_rows h (row := 1874) (i := 19) rfl (by decide)
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
  have e_1875_0 := arithEq_of_rows h (row := 1875) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_0
  simp only [← c114492, ← c114493, ← c114494] at e_1875_0
  have e_1875_1 := arithEq_of_rows h (row := 1875) (i := 1) rfl (by decide)
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
  have e_1875_2 := arithEq_of_rows h (row := 1875) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_2
  simp only [← c114498, ← c114499, ← c114500] at e_1875_2
  have e_1875_3 := arithEq_of_rows h (row := 1875) (i := 3) rfl (by decide)
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
  have e_1875_4 := arithEq_of_rows h (row := 1875) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_4
  simp only [← c114504, ← c114505, ← c114506] at e_1875_4
  have e_1875_5 := arithEq_of_rows h (row := 1875) (i := 5) rfl (by decide)
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
  have e_1875_6 := arithEq_of_rows h (row := 1875) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_6
  simp only [← c114510, ← c114511, ← c114512] at e_1875_6
  have e_1875_7 := arithEq_of_rows h (row := 1875) (i := 7) rfl (by decide)
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
  have e_1875_8 := arithEq_of_rows h (row := 1875) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_8
  simp only [← c114516, ← c114517, ← c114518] at e_1875_8
  have e_1875_9 := arithEq_of_rows h (row := 1875) (i := 9) rfl (by decide)
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
  have e_1875_10 := arithEq_of_rows h (row := 1875) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_10
  simp only [← c114522, ← c114523, ← c114524] at e_1875_10
  have e_1875_11 := arithEq_of_rows h (row := 1875) (i := 11) rfl (by decide)
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
  have e_1875_12 := arithEq_of_rows h (row := 1875) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_12
  simp only [← c114528, ← c114529, ← c114530] at e_1875_12
  have e_1875_13 := arithEq_of_rows h (row := 1875) (i := 13) rfl (by decide)
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
  have e_1875_14 := arithEq_of_rows h (row := 1875) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_14
  simp only [← c114534, ← c114535, ← c114536] at e_1875_14
  have e_1875_15 := arithEq_of_rows h (row := 1875) (i := 15) rfl (by decide)
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
  have e_1875_16 := arithEq_of_rows h (row := 1875) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_16
  simp only [← c114540, ← c114541, ← c114542] at e_1875_16
  have e_1875_17 := arithEq_of_rows h (row := 1875) (i := 17) rfl (by decide)
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
  have e_1875_18 := arithEq_of_rows h (row := 1875) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1875_18
  simp only [← c114546, ← c114547, ← c114548] at e_1875_18
  have e_1875_19 := arithEq_of_rows h (row := 1875) (i := 19) rfl (by decide)
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
  have e_1876_0 := arithEq_of_rows h (row := 1876) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_0
  simp only [← c114552, ← c114553, ← c114554] at e_1876_0
  have e_1876_1 := arithEq_of_rows h (row := 1876) (i := 1) rfl (by decide)
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
  have e_1876_2 := arithEq_of_rows h (row := 1876) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_2
  simp only [← c114558, ← c114559, ← c114560] at e_1876_2
  have e_1876_3 := arithEq_of_rows h (row := 1876) (i := 3) rfl (by decide)
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
  have e_1876_4 := arithEq_of_rows h (row := 1876) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_4
  simp only [← c114564, ← c114565, ← c114566] at e_1876_4
  have e_1876_5 := arithEq_of_rows h (row := 1876) (i := 5) rfl (by decide)
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
  have e_1876_6 := arithEq_of_rows h (row := 1876) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_6
  simp only [← c114570, ← c114571, ← c114572] at e_1876_6
  have e_1876_7 := arithEq_of_rows h (row := 1876) (i := 7) rfl (by decide)
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
  have e_1876_8 := arithEq_of_rows h (row := 1876) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_8
  simp only [← c114576, ← c114577, ← c114578] at e_1876_8
  have e_1876_9 := arithEq_of_rows h (row := 1876) (i := 9) rfl (by decide)
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
  have e_1876_10 := arithEq_of_rows h (row := 1876) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_10
  simp only [← c114582, ← c114583, ← c114584] at e_1876_10
  have e_1876_11 := arithEq_of_rows h (row := 1876) (i := 11) rfl (by decide)
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
  have e_1876_12 := arithEq_of_rows h (row := 1876) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_12
  simp only [← c114588, ← c114589, ← c114590] at e_1876_12
  have e_1876_13 := arithEq_of_rows h (row := 1876) (i := 13) rfl (by decide)
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
  have e_1876_14 := arithEq_of_rows h (row := 1876) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_14
  simp only [← c114594, ← c114595, ← c114596] at e_1876_14
  have e_1876_15 := arithEq_of_rows h (row := 1876) (i := 15) rfl (by decide)
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
  have e_1876_16 := arithEq_of_rows h (row := 1876) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_16
  simp only [← c114600, ← c114601, ← c114602] at e_1876_16
  have e_1876_17 := arithEq_of_rows h (row := 1876) (i := 17) rfl (by decide)
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
  have e_1876_18 := arithEq_of_rows h (row := 1876) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1876_18
  simp only [← c114606, ← c114607, ← c114608] at e_1876_18
  have e_1876_19 := arithEq_of_rows h (row := 1876) (i := 19) rfl (by decide)
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
  have e_1877_0 := arithEq_of_rows h (row := 1877) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_0
  simp only [← c114612, ← c114613, ← c114614] at e_1877_0
  have e_1877_1 := arithEq_of_rows h (row := 1877) (i := 1) rfl (by decide)
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
  have e_1877_2 := arithEq_of_rows h (row := 1877) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_2
  simp only [← c114618, ← c114619, ← c114620] at e_1877_2
  have e_1877_3 := arithEq_of_rows h (row := 1877) (i := 3) rfl (by decide)
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
  have e_1877_4 := arithEq_of_rows h (row := 1877) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_4
  simp only [← c114624, ← c114625, ← c114626] at e_1877_4
  have e_1877_5 := arithEq_of_rows h (row := 1877) (i := 5) rfl (by decide)
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
  have e_1877_6 := arithEq_of_rows h (row := 1877) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_6
  simp only [← c114630, ← c114631, ← c114632] at e_1877_6
  have e_1877_7 := arithEq_of_rows h (row := 1877) (i := 7) rfl (by decide)
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
  have e_1877_8 := arithEq_of_rows h (row := 1877) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_8
  simp only [← c114636, ← c114637, ← c114638] at e_1877_8
  have e_1877_9 := arithEq_of_rows h (row := 1877) (i := 9) rfl (by decide)
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
  have e_1877_10 := arithEq_of_rows h (row := 1877) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_10
  simp only [← c114642, ← c114643, ← c114644] at e_1877_10
  have e_1877_11 := arithEq_of_rows h (row := 1877) (i := 11) rfl (by decide)
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
  have e_1877_12 := arithEq_of_rows h (row := 1877) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_12
  simp only [← c114648, ← c114649, ← c114650] at e_1877_12
  have e_1877_13 := arithEq_of_rows h (row := 1877) (i := 13) rfl (by decide)
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
  have e_1877_14 := arithEq_of_rows h (row := 1877) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_14
  simp only [← c114654, ← c114655, ← c114656] at e_1877_14
  have e_1877_15 := arithEq_of_rows h (row := 1877) (i := 15) rfl (by decide)
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
  have e_1877_16 := arithEq_of_rows h (row := 1877) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_16
  simp only [← c114660, ← c114661, ← c114662] at e_1877_16
  have e_1877_17 := arithEq_of_rows h (row := 1877) (i := 17) rfl (by decide)
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
  have e_1877_18 := arithEq_of_rows h (row := 1877) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1877_18
  simp only [← c114666, ← c114667, ← c114668] at e_1877_18
  have e_1877_19 := arithEq_of_rows h (row := 1877) (i := 19) rfl (by decide)
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
  have e_1878_0 := arithEq_of_rows h (row := 1878) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_0
  simp only [← c114672, ← c114673, ← c114674] at e_1878_0
  have e_1878_1 := arithEq_of_rows h (row := 1878) (i := 1) rfl (by decide)
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
  have e_1878_2 := arithEq_of_rows h (row := 1878) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_2
  simp only [← c114678, ← c114679, ← c114680] at e_1878_2
  have e_1878_3 := arithEq_of_rows h (row := 1878) (i := 3) rfl (by decide)
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
  have e_1878_4 := arithEq_of_rows h (row := 1878) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_4
  simp only [← c114684, ← c114685, ← c114686] at e_1878_4
  have e_1878_5 := arithEq_of_rows h (row := 1878) (i := 5) rfl (by decide)
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
  have e_1878_6 := arithEq_of_rows h (row := 1878) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_6
  simp only [← c114690, ← c114691, ← c114692] at e_1878_6
  have e_1878_7 := arithEq_of_rows h (row := 1878) (i := 7) rfl (by decide)
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
  have e_1878_8 := arithEq_of_rows h (row := 1878) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_8
  simp only [← c114696, ← c114697, ← c114698] at e_1878_8
  have e_1878_9 := arithEq_of_rows h (row := 1878) (i := 9) rfl (by decide)
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
  have e_1878_10 := arithEq_of_rows h (row := 1878) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_10
  simp only [← c114702, ← c114703, ← c114704] at e_1878_10
  have e_1878_11 := arithEq_of_rows h (row := 1878) (i := 11) rfl (by decide)
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
  have e_1878_12 := arithEq_of_rows h (row := 1878) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_12
  simp only [← c114708, ← c114709, ← c114710] at e_1878_12
  have e_1878_13 := arithEq_of_rows h (row := 1878) (i := 13) rfl (by decide)
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
  have e_1878_14 := arithEq_of_rows h (row := 1878) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_14
  simp only [← c114714, ← c114715, ← c114716] at e_1878_14
  have e_1878_15 := arithEq_of_rows h (row := 1878) (i := 15) rfl (by decide)
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
  have e_1878_16 := arithEq_of_rows h (row := 1878) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_16
  simp only [← c114720, ← c114721, ← c114722] at e_1878_16
  have e_1878_17 := arithEq_of_rows h (row := 1878) (i := 17) rfl (by decide)
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
  have e_1878_18 := arithEq_of_rows h (row := 1878) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1878_18
  simp only [← c114726, ← c114727, ← c114728] at e_1878_18
  have e_1878_19 := arithEq_of_rows h (row := 1878) (i := 19) rfl (by decide)
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
  have e_1879_0 := arithEq_of_rows h (row := 1879) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_0
  simp only [← c114732, ← c114733, ← c114734] at e_1879_0
  have e_1879_1 := arithEq_of_rows h (row := 1879) (i := 1) rfl (by decide)
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
  have e_1879_2 := arithEq_of_rows h (row := 1879) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_2
  simp only [← c114738, ← c114739, ← c114740] at e_1879_2
  have e_1879_3 := arithEq_of_rows h (row := 1879) (i := 3) rfl (by decide)
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
  have e_1879_4 := arithEq_of_rows h (row := 1879) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_4
  simp only [← c114744, ← c114745, ← c114746] at e_1879_4
  have e_1879_5 := arithEq_of_rows h (row := 1879) (i := 5) rfl (by decide)
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
  have e_1879_6 := arithEq_of_rows h (row := 1879) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_6
  simp only [← c114750, ← c114751, ← c114752] at e_1879_6
  have e_1879_7 := arithEq_of_rows h (row := 1879) (i := 7) rfl (by decide)
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
  have e_1879_8 := arithEq_of_rows h (row := 1879) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_8
  simp only [← c114756, ← c114757, ← c114758] at e_1879_8
  have e_1879_9 := arithEq_of_rows h (row := 1879) (i := 9) rfl (by decide)
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
  have e_1879_10 := arithEq_of_rows h (row := 1879) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_10
  simp only [← c114762, ← c114763, ← c114764] at e_1879_10
  have e_1879_11 := arithEq_of_rows h (row := 1879) (i := 11) rfl (by decide)
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
  have e_1879_12 := arithEq_of_rows h (row := 1879) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_12
  simp only [← c114768, ← c114769, ← c114770] at e_1879_12
  have e_1879_13 := arithEq_of_rows h (row := 1879) (i := 13) rfl (by decide)
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
  have e_1879_14 := arithEq_of_rows h (row := 1879) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_14
  simp only [← c114774, ← c114775, ← c114776] at e_1879_14
  have e_1879_15 := arithEq_of_rows h (row := 1879) (i := 15) rfl (by decide)
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
  have e_1879_16 := arithEq_of_rows h (row := 1879) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_16
  simp only [← c114780, ← c114781, ← c114782] at e_1879_16
  have e_1879_17 := arithEq_of_rows h (row := 1879) (i := 17) rfl (by decide)
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
  have e_1879_18 := arithEq_of_rows h (row := 1879) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1879_18
  simp only [← c114786, ← c114787, ← c114788] at e_1879_18
  have e_1879_19 := arithEq_of_rows h (row := 1879) (i := 19) rfl (by decide)
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
  have e_1880_0 := arithEq_of_rows h (row := 1880) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_0
  simp only [← c114792, ← c114793, ← c114794] at e_1880_0
  have e_1880_1 := arithEq_of_rows h (row := 1880) (i := 1) rfl (by decide)
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
  have e_1880_2 := arithEq_of_rows h (row := 1880) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_2
  simp only [← c114798, ← c114799, ← c114800] at e_1880_2
  have e_1880_3 := arithEq_of_rows h (row := 1880) (i := 3) rfl (by decide)
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
  have e_1880_4 := arithEq_of_rows h (row := 1880) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_4
  simp only [← c114804, ← c114805, ← c114806] at e_1880_4
  have e_1880_5 := arithEq_of_rows h (row := 1880) (i := 5) rfl (by decide)
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
  have e_1880_6 := arithEq_of_rows h (row := 1880) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_6
  simp only [← c114810, ← c114811, ← c114812] at e_1880_6
  have e_1880_7 := arithEq_of_rows h (row := 1880) (i := 7) rfl (by decide)
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
  have e_1880_8 := arithEq_of_rows h (row := 1880) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_8
  simp only [← c114816, ← c114817, ← c114818] at e_1880_8
  have e_1880_9 := arithEq_of_rows h (row := 1880) (i := 9) rfl (by decide)
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
  have e_1880_10 := arithEq_of_rows h (row := 1880) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_10
  simp only [← c114822, ← c114823, ← c114824] at e_1880_10
  have e_1880_11 := arithEq_of_rows h (row := 1880) (i := 11) rfl (by decide)
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
  have e_1880_12 := arithEq_of_rows h (row := 1880) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_12
  simp only [← c114828, ← c114829, ← c114830] at e_1880_12
  have e_1880_13 := arithEq_of_rows h (row := 1880) (i := 13) rfl (by decide)
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
  have e_1880_14 := arithEq_of_rows h (row := 1880) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_14
  simp only [← c114834, ← c114835, ← c114836] at e_1880_14
  have e_1880_15 := arithEq_of_rows h (row := 1880) (i := 15) rfl (by decide)
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
  have e_1880_16 := arithEq_of_rows h (row := 1880) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_16
  simp only [← c114840, ← c114841, ← c114842] at e_1880_16
  have e_1880_17 := arithEq_of_rows h (row := 1880) (i := 17) rfl (by decide)
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
  have e_1880_18 := arithEq_of_rows h (row := 1880) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1880_18
  simp only [← c114846, ← c114847, ← c114848] at e_1880_18
  have e_1880_19 := arithEq_of_rows h (row := 1880) (i := 19) rfl (by decide)
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
  have e_1881_0 := arithEq_of_rows h (row := 1881) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_0
  simp only [← c114852, ← c114853, ← c114854] at e_1881_0
  have e_1881_1 := arithEq_of_rows h (row := 1881) (i := 1) rfl (by decide)
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
  have e_1881_2 := arithEq_of_rows h (row := 1881) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_2
  simp only [← c114858, ← c114859, ← c114860] at e_1881_2
  have e_1881_3 := arithEq_of_rows h (row := 1881) (i := 3) rfl (by decide)
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
  have e_1881_4 := arithEq_of_rows h (row := 1881) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_4
  simp only [← c114864, ← c114865, ← c114866] at e_1881_4
  have e_1881_5 := arithEq_of_rows h (row := 1881) (i := 5) rfl (by decide)
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
  have e_1881_6 := arithEq_of_rows h (row := 1881) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_6
  simp only [← c114870, ← c114871, ← c114872] at e_1881_6
  have e_1881_7 := arithEq_of_rows h (row := 1881) (i := 7) rfl (by decide)
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
  have e_1881_8 := arithEq_of_rows h (row := 1881) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_8
  simp only [← c114876, ← c114877, ← c114878] at e_1881_8
  have e_1881_9 := arithEq_of_rows h (row := 1881) (i := 9) rfl (by decide)
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
  have e_1881_10 := arithEq_of_rows h (row := 1881) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_10
  simp only [← c114882, ← c114883, ← c114884] at e_1881_10
  have e_1881_11 := arithEq_of_rows h (row := 1881) (i := 11) rfl (by decide)
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
  have e_1881_12 := arithEq_of_rows h (row := 1881) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_12
  simp only [← c114888, ← c114889, ← c114890] at e_1881_12
  have e_1881_13 := arithEq_of_rows h (row := 1881) (i := 13) rfl (by decide)
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
  have e_1881_14 := arithEq_of_rows h (row := 1881) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_14
  simp only [← c114894, ← c114895, ← c114896] at e_1881_14
  have e_1881_15 := arithEq_of_rows h (row := 1881) (i := 15) rfl (by decide)
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
  have e_1881_16 := arithEq_of_rows h (row := 1881) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_16
  simp only [← c114900, ← c114901, ← c114902] at e_1881_16
  have e_1881_17 := arithEq_of_rows h (row := 1881) (i := 17) rfl (by decide)
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
  have e_1881_18 := arithEq_of_rows h (row := 1881) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1881_18
  simp only [← c114906, ← c114907, ← c114908] at e_1881_18
  have e_1881_19 := arithEq_of_rows h (row := 1881) (i := 19) rfl (by decide)
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
  have e_1882_0 := arithEq_of_rows h (row := 1882) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_0
  simp only [← c114912, ← c114913, ← c114914] at e_1882_0
  have e_1882_1 := arithEq_of_rows h (row := 1882) (i := 1) rfl (by decide)
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
  have e_1882_2 := arithEq_of_rows h (row := 1882) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_2
  simp only [← c114918, ← c114919, ← c114920] at e_1882_2
  have e_1882_3 := arithEq_of_rows h (row := 1882) (i := 3) rfl (by decide)
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
  have e_1882_4 := arithEq_of_rows h (row := 1882) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_4
  simp only [← c114924, ← c114925, ← c114926] at e_1882_4
  have e_1882_5 := arithEq_of_rows h (row := 1882) (i := 5) rfl (by decide)
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
  have e_1882_6 := arithEq_of_rows h (row := 1882) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_6
  simp only [← c114930, ← c114931, ← c114932] at e_1882_6
  have e_1882_7 := arithEq_of_rows h (row := 1882) (i := 7) rfl (by decide)
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
  have e_1882_8 := arithEq_of_rows h (row := 1882) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_8
  simp only [← c114936, ← c114937, ← c114938] at e_1882_8
  have e_1882_9 := arithEq_of_rows h (row := 1882) (i := 9) rfl (by decide)
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
  have e_1882_10 := arithEq_of_rows h (row := 1882) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_10
  simp only [← c114942, ← c114943, ← c114944] at e_1882_10
  have e_1882_11 := arithEq_of_rows h (row := 1882) (i := 11) rfl (by decide)
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
  have e_1882_12 := arithEq_of_rows h (row := 1882) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_12
  simp only [← c114948, ← c114949, ← c114950] at e_1882_12
  have e_1882_13 := arithEq_of_rows h (row := 1882) (i := 13) rfl (by decide)
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
  have e_1882_14 := arithEq_of_rows h (row := 1882) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_14
  simp only [← c114954, ← c114955, ← c114956] at e_1882_14
  have e_1882_15 := arithEq_of_rows h (row := 1882) (i := 15) rfl (by decide)
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
  have e_1882_16 := arithEq_of_rows h (row := 1882) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_16
  simp only [← c114960, ← c114961, ← c114962] at e_1882_16
  have e_1882_17 := arithEq_of_rows h (row := 1882) (i := 17) rfl (by decide)
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
  have e_1882_18 := arithEq_of_rows h (row := 1882) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1882_18
  simp only [← c114966, ← c114967, ← c114968] at e_1882_18
  have e_1882_19 := arithEq_of_rows h (row := 1882) (i := 19) rfl (by decide)
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
  have e_1883_0 := arithEq_of_rows h (row := 1883) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_0
  simp only [← c114972, ← c114973, ← c114974] at e_1883_0
  have e_1883_1 := arithEq_of_rows h (row := 1883) (i := 1) rfl (by decide)
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
  have e_1883_2 := arithEq_of_rows h (row := 1883) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_2
  simp only [← c114978, ← c114979, ← c114980] at e_1883_2
  have e_1883_3 := arithEq_of_rows h (row := 1883) (i := 3) rfl (by decide)
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
  have e_1883_4 := arithEq_of_rows h (row := 1883) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_4
  simp only [← c114984, ← c114985, ← c114986] at e_1883_4
  have e_1883_5 := arithEq_of_rows h (row := 1883) (i := 5) rfl (by decide)
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
  have e_1883_6 := arithEq_of_rows h (row := 1883) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_6
  simp only [← c114990, ← c114991, ← c114992] at e_1883_6
  have e_1883_7 := arithEq_of_rows h (row := 1883) (i := 7) rfl (by decide)
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
  have e_1883_8 := arithEq_of_rows h (row := 1883) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_8
  simp only [← c114996, ← c114997, ← c114998] at e_1883_8
  have e_1883_9 := arithEq_of_rows h (row := 1883) (i := 9) rfl (by decide)
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
  have e_1883_10 := arithEq_of_rows h (row := 1883) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_10
  simp only [← c115002, ← c115003, ← c115004] at e_1883_10
  have e_1883_11 := arithEq_of_rows h (row := 1883) (i := 11) rfl (by decide)
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
  have e_1883_12 := arithEq_of_rows h (row := 1883) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_12
  simp only [← c115008, ← c115009, ← c115010] at e_1883_12
  have e_1883_13 := arithEq_of_rows h (row := 1883) (i := 13) rfl (by decide)
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
  have e_1883_14 := arithEq_of_rows h (row := 1883) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_14
  simp only [← c115014, ← c115015, ← c115016] at e_1883_14
  have e_1883_15 := arithEq_of_rows h (row := 1883) (i := 15) rfl (by decide)
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
  have e_1883_16 := arithEq_of_rows h (row := 1883) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_16
  simp only [← c115020, ← c115021, ← c115022] at e_1883_16
  have e_1883_17 := arithEq_of_rows h (row := 1883) (i := 17) rfl (by decide)
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
  have e_1883_18 := arithEq_of_rows h (row := 1883) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1883_18
  simp only [← c115026, ← c115027, ← c115028] at e_1883_18
  have e_1883_19 := arithEq_of_rows h (row := 1883) (i := 19) rfl (by decide)
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
  have e_1884_0 := arithEq_of_rows h (row := 1884) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_0
  simp only [← c115032, ← c115033, ← c115034] at e_1884_0
  have e_1884_1 := arithEq_of_rows h (row := 1884) (i := 1) rfl (by decide)
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
  have e_1884_2 := arithEq_of_rows h (row := 1884) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_2
  simp only [← c115038, ← c115039, ← c115040] at e_1884_2
  have e_1884_3 := arithEq_of_rows h (row := 1884) (i := 3) rfl (by decide)
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
  have e_1884_4 := arithEq_of_rows h (row := 1884) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_4
  simp only [← c115044, ← c115045, ← c115046] at e_1884_4
  have e_1884_5 := arithEq_of_rows h (row := 1884) (i := 5) rfl (by decide)
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
  have e_1884_6 := arithEq_of_rows h (row := 1884) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_6
  simp only [← c115050, ← c115051, ← c115052] at e_1884_6
  have e_1884_7 := arithEq_of_rows h (row := 1884) (i := 7) rfl (by decide)
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
  have e_1884_8 := arithEq_of_rows h (row := 1884) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_8
  simp only [← c115056, ← c115057, ← c115058] at e_1884_8
  have e_1884_9 := arithEq_of_rows h (row := 1884) (i := 9) rfl (by decide)
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
  have e_1884_10 := arithEq_of_rows h (row := 1884) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_10
  simp only [← c115062, ← c115063, ← c115064] at e_1884_10
  have e_1884_11 := arithEq_of_rows h (row := 1884) (i := 11) rfl (by decide)
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
  have e_1884_12 := arithEq_of_rows h (row := 1884) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_12
  simp only [← c115068, ← c115069, ← c115070] at e_1884_12
  have e_1884_13 := arithEq_of_rows h (row := 1884) (i := 13) rfl (by decide)
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
  have e_1884_14 := arithEq_of_rows h (row := 1884) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_14
  simp only [← c115074, ← c115075, ← c115076] at e_1884_14
  have e_1884_15 := arithEq_of_rows h (row := 1884) (i := 15) rfl (by decide)
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
  have e_1884_16 := arithEq_of_rows h (row := 1884) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_16
  simp only [← c115080, ← c115081, ← c115082] at e_1884_16
  have e_1884_17 := arithEq_of_rows h (row := 1884) (i := 17) rfl (by decide)
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
  have e_1884_18 := arithEq_of_rows h (row := 1884) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1884_18
  simp only [← c115086, ← c115087, ← c115088] at e_1884_18
  have e_1884_19 := arithEq_of_rows h (row := 1884) (i := 19) rfl (by decide)
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
  have e_1885_0 := arithEq_of_rows h (row := 1885) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_0
  simp only [← c115092, ← c115093, ← c115094] at e_1885_0
  have e_1885_1 := arithEq_of_rows h (row := 1885) (i := 1) rfl (by decide)
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
  have e_1885_2 := arithEq_of_rows h (row := 1885) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_2
  simp only [← c115098, ← c115099, ← c115100] at e_1885_2
  have e_1885_3 := arithEq_of_rows h (row := 1885) (i := 3) rfl (by decide)
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
  have e_1885_4 := arithEq_of_rows h (row := 1885) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_4
  simp only [← c115104, ← c115105, ← c115106] at e_1885_4
  have e_1885_5 := arithEq_of_rows h (row := 1885) (i := 5) rfl (by decide)
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
  have e_1885_6 := arithEq_of_rows h (row := 1885) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_6
  simp only [← c115110, ← c115111, ← c115112] at e_1885_6
  have e_1885_7 := arithEq_of_rows h (row := 1885) (i := 7) rfl (by decide)
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
  have e_1885_8 := arithEq_of_rows h (row := 1885) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_8
  simp only [← c115116, ← c115117, ← c115118] at e_1885_8
  have e_1885_9 := arithEq_of_rows h (row := 1885) (i := 9) rfl (by decide)
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
  have e_1885_10 := arithEq_of_rows h (row := 1885) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_10
  simp only [← c115122, ← c115123, ← c115124] at e_1885_10
  have e_1885_11 := arithEq_of_rows h (row := 1885) (i := 11) rfl (by decide)
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
  have e_1885_12 := arithEq_of_rows h (row := 1885) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_12
  simp only [← c115128, ← c115129, ← c115130] at e_1885_12
  have e_1885_13 := arithEq_of_rows h (row := 1885) (i := 13) rfl (by decide)
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
  have e_1885_14 := arithEq_of_rows h (row := 1885) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_14
  simp only [← c115134, ← c115135, ← c115136] at e_1885_14
  have e_1885_15 := arithEq_of_rows h (row := 1885) (i := 15) rfl (by decide)
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
  have e_1885_16 := arithEq_of_rows h (row := 1885) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_16
  simp only [← c115140, ← c115141, ← c115142] at e_1885_16
  have e_1885_17 := arithEq_of_rows h (row := 1885) (i := 17) rfl (by decide)
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
  have e_1885_18 := arithEq_of_rows h (row := 1885) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1885_18
  simp only [← c115146, ← c115147, ← c115148] at e_1885_18
  have e_1885_19 := arithEq_of_rows h (row := 1885) (i := 19) rfl (by decide)
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
  have e_1886_0 := arithEq_of_rows h (row := 1886) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_0
  simp only [← c115152, ← c115153, ← c115154] at e_1886_0
  have e_1886_1 := arithEq_of_rows h (row := 1886) (i := 1) rfl (by decide)
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
  have e_1886_2 := arithEq_of_rows h (row := 1886) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_2
  simp only [← c115158, ← c115159, ← c115160] at e_1886_2
  have e_1886_3 := arithEq_of_rows h (row := 1886) (i := 3) rfl (by decide)
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
  have e_1886_4 := arithEq_of_rows h (row := 1886) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_4
  simp only [← c115164, ← c115165, ← c115166] at e_1886_4
  have e_1886_5 := arithEq_of_rows h (row := 1886) (i := 5) rfl (by decide)
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
  have e_1886_6 := arithEq_of_rows h (row := 1886) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_6
  simp only [← c115170, ← c115171, ← c115172] at e_1886_6
  have e_1886_7 := arithEq_of_rows h (row := 1886) (i := 7) rfl (by decide)
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
  have e_1886_8 := arithEq_of_rows h (row := 1886) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_8
  simp only [← c115176, ← c115177, ← c115178] at e_1886_8
  have e_1886_9 := arithEq_of_rows h (row := 1886) (i := 9) rfl (by decide)
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
  have e_1886_10 := arithEq_of_rows h (row := 1886) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_10
  simp only [← c115182, ← c115183, ← c115184] at e_1886_10
  have e_1886_11 := arithEq_of_rows h (row := 1886) (i := 11) rfl (by decide)
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
  have e_1886_12 := arithEq_of_rows h (row := 1886) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_12
  simp only [← c115188, ← c115189, ← c115190] at e_1886_12
  have e_1886_13 := arithEq_of_rows h (row := 1886) (i := 13) rfl (by decide)
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
  have e_1886_14 := arithEq_of_rows h (row := 1886) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_14
  simp only [← c115194, ← c115195, ← c115196] at e_1886_14
  have e_1886_15 := arithEq_of_rows h (row := 1886) (i := 15) rfl (by decide)
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
  have e_1886_16 := arithEq_of_rows h (row := 1886) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_16
  simp only [← c115200, ← c115201, ← c115202] at e_1886_16
  have e_1886_17 := arithEq_of_rows h (row := 1886) (i := 17) rfl (by decide)
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
  have e_1886_18 := arithEq_of_rows h (row := 1886) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1886_18
  simp only [← c115206, ← c115207, ← c115208] at e_1886_18
  have e_1886_19 := arithEq_of_rows h (row := 1886) (i := 19) rfl (by decide)
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
  have e_1887_0 := arithEq_of_rows h (row := 1887) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_0
  simp only [← c115212, ← c115213, ← c115214] at e_1887_0
  have e_1887_1 := arithEq_of_rows h (row := 1887) (i := 1) rfl (by decide)
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
  have e_1887_2 := arithEq_of_rows h (row := 1887) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_2
  simp only [← c115218, ← c115219, ← c115220] at e_1887_2
  have e_1887_3 := arithEq_of_rows h (row := 1887) (i := 3) rfl (by decide)
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
  have e_1887_4 := arithEq_of_rows h (row := 1887) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_4
  simp only [← c115224, ← c115225, ← c115226] at e_1887_4
  have e_1887_5 := arithEq_of_rows h (row := 1887) (i := 5) rfl (by decide)
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
  have e_1887_6 := arithEq_of_rows h (row := 1887) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_6
  simp only [← c115230, ← c115231, ← c115232] at e_1887_6
  have e_1887_7 := arithEq_of_rows h (row := 1887) (i := 7) rfl (by decide)
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
  have e_1887_8 := arithEq_of_rows h (row := 1887) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_8
  simp only [← c115236, ← c115237, ← c115238] at e_1887_8
  have e_1887_9 := arithEq_of_rows h (row := 1887) (i := 9) rfl (by decide)
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
  have e_1887_10 := arithEq_of_rows h (row := 1887) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_10
  simp only [← c115242, ← c115243, ← c115244] at e_1887_10
  have e_1887_11 := arithEq_of_rows h (row := 1887) (i := 11) rfl (by decide)
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
  have e_1887_12 := arithEq_of_rows h (row := 1887) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_12
  simp only [← c115248, ← c115249, ← c115250] at e_1887_12
  have e_1887_13 := arithEq_of_rows h (row := 1887) (i := 13) rfl (by decide)
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
  have e_1887_14 := arithEq_of_rows h (row := 1887) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_14
  simp only [← c115254, ← c115255, ← c115256] at e_1887_14
  have e_1887_15 := arithEq_of_rows h (row := 1887) (i := 15) rfl (by decide)
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
  have e_1887_16 := arithEq_of_rows h (row := 1887) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_16
  simp only [← c115260, ← c115261, ← c115262] at e_1887_16
  have e_1887_17 := arithEq_of_rows h (row := 1887) (i := 17) rfl (by decide)
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
  have e_1887_18 := arithEq_of_rows h (row := 1887) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1887_18
  simp only [← c115266, ← c115267, ← c115268] at e_1887_18
  have e_1887_19 := arithEq_of_rows h (row := 1887) (i := 19) rfl (by decide)
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
  have e_1888_0 := arithEq_of_rows h (row := 1888) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_0
  simp only [← c115272, ← c115273, ← c115274] at e_1888_0
  have e_1888_1 := arithEq_of_rows h (row := 1888) (i := 1) rfl (by decide)
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
  have e_1888_2 := arithEq_of_rows h (row := 1888) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_2
  simp only [← c115278, ← c115279, ← c115280] at e_1888_2
  have e_1888_3 := arithEq_of_rows h (row := 1888) (i := 3) rfl (by decide)
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
  have e_1888_4 := arithEq_of_rows h (row := 1888) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_4
  simp only [← c115284, ← c115285, ← c115286] at e_1888_4
  have e_1888_5 := arithEq_of_rows h (row := 1888) (i := 5) rfl (by decide)
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
  have e_1888_6 := arithEq_of_rows h (row := 1888) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_6
  simp only [← c115290, ← c115291, ← c115292] at e_1888_6
  have e_1888_7 := arithEq_of_rows h (row := 1888) (i := 7) rfl (by decide)
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
  have e_1888_8 := arithEq_of_rows h (row := 1888) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_8
  simp only [← c115296, ← c115297, ← c115298] at e_1888_8
  have e_1888_9 := arithEq_of_rows h (row := 1888) (i := 9) rfl (by decide)
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
  have e_1888_10 := arithEq_of_rows h (row := 1888) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_10
  simp only [← c115302, ← c115303, ← c115304] at e_1888_10
  have e_1888_11 := arithEq_of_rows h (row := 1888) (i := 11) rfl (by decide)
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
  have e_1888_12 := arithEq_of_rows h (row := 1888) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_12
  simp only [← c115308, ← c115309, ← c115310] at e_1888_12
  have e_1888_13 := arithEq_of_rows h (row := 1888) (i := 13) rfl (by decide)
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
  have e_1888_14 := arithEq_of_rows h (row := 1888) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_14
  simp only [← c115314, ← c115315, ← c115316] at e_1888_14
  have e_1888_15 := arithEq_of_rows h (row := 1888) (i := 15) rfl (by decide)
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
  have e_1888_16 := arithEq_of_rows h (row := 1888) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_16
  simp only [← c115320, ← c115321, ← c115322] at e_1888_16
  have e_1888_17 := arithEq_of_rows h (row := 1888) (i := 17) rfl (by decide)
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
  have e_1888_18 := arithEq_of_rows h (row := 1888) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1888_18
  simp only [← c115326, ← c115327, ← c115328] at e_1888_18
  have e_1888_19 := arithEq_of_rows h (row := 1888) (i := 19) rfl (by decide)
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
  have e_1889_0 := arithEq_of_rows h (row := 1889) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_0
  simp only [← c115332, ← c115333, ← c115334] at e_1889_0
  have e_1889_1 := arithEq_of_rows h (row := 1889) (i := 1) rfl (by decide)
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
  have e_1889_2 := arithEq_of_rows h (row := 1889) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_2
  simp only [← c115338, ← c115339, ← c115340] at e_1889_2
  have e_1889_3 := arithEq_of_rows h (row := 1889) (i := 3) rfl (by decide)
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
  have e_1889_4 := arithEq_of_rows h (row := 1889) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_4
  simp only [← c115344, ← c115345, ← c115346] at e_1889_4
  have e_1889_5 := arithEq_of_rows h (row := 1889) (i := 5) rfl (by decide)
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
  have e_1889_6 := arithEq_of_rows h (row := 1889) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_6
  simp only [← c115350, ← c115351, ← c115352] at e_1889_6
  have e_1889_7 := arithEq_of_rows h (row := 1889) (i := 7) rfl (by decide)
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
  have e_1889_8 := arithEq_of_rows h (row := 1889) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_8
  simp only [← c115356, ← c115357, ← c115358] at e_1889_8
  have e_1889_9 := arithEq_of_rows h (row := 1889) (i := 9) rfl (by decide)
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
  have e_1889_10 := arithEq_of_rows h (row := 1889) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_10
  simp only [← c115362, ← c115363, ← c115364] at e_1889_10
  have e_1889_11 := arithEq_of_rows h (row := 1889) (i := 11) rfl (by decide)
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
  have e_1889_12 := arithEq_of_rows h (row := 1889) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_12
  simp only [← c115368, ← c115369, ← c115370] at e_1889_12
  have e_1889_13 := arithEq_of_rows h (row := 1889) (i := 13) rfl (by decide)
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
  have e_1889_14 := arithEq_of_rows h (row := 1889) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_14
  simp only [← c115374, ← c115375, ← c115376] at e_1889_14
  have e_1889_15 := arithEq_of_rows h (row := 1889) (i := 15) rfl (by decide)
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
  have e_1889_16 := arithEq_of_rows h (row := 1889) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_16
  simp only [← c115380, ← c115381, ← c115382] at e_1889_16
  have e_1889_17 := arithEq_of_rows h (row := 1889) (i := 17) rfl (by decide)
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
  have e_1889_18 := arithEq_of_rows h (row := 1889) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1889_18
  simp only [← c115386, ← c115387, ← c115388] at e_1889_18
  have e_1889_19 := arithEq_of_rows h (row := 1889) (i := 19) rfl (by decide)
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
  have e_1890_0 := arithEq_of_rows h (row := 1890) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_0
  simp only [← c115392, ← c115393, ← c115394] at e_1890_0
  have e_1890_1 := arithEq_of_rows h (row := 1890) (i := 1) rfl (by decide)
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
  have e_1890_2 := arithEq_of_rows h (row := 1890) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_2
  simp only [← c115398, ← c115399, ← c115400] at e_1890_2
  have e_1890_3 := arithEq_of_rows h (row := 1890) (i := 3) rfl (by decide)
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
  have e_1890_4 := arithEq_of_rows h (row := 1890) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_4
  simp only [← c115404, ← c115405, ← c115406] at e_1890_4
  have e_1890_5 := arithEq_of_rows h (row := 1890) (i := 5) rfl (by decide)
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
  have e_1890_6 := arithEq_of_rows h (row := 1890) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_6
  simp only [← c115410, ← c115411, ← c115412] at e_1890_6
  have e_1890_7 := arithEq_of_rows h (row := 1890) (i := 7) rfl (by decide)
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
  have e_1890_8 := arithEq_of_rows h (row := 1890) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_8
  simp only [← c115416, ← c115417, ← c115418] at e_1890_8
  have e_1890_9 := arithEq_of_rows h (row := 1890) (i := 9) rfl (by decide)
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
  have e_1890_10 := arithEq_of_rows h (row := 1890) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_10
  simp only [← c115422, ← c115423, ← c115424] at e_1890_10
  have e_1890_11 := arithEq_of_rows h (row := 1890) (i := 11) rfl (by decide)
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
  have e_1890_12 := arithEq_of_rows h (row := 1890) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_12
  simp only [← c115428, ← c115429, ← c115430] at e_1890_12
  have e_1890_13 := arithEq_of_rows h (row := 1890) (i := 13) rfl (by decide)
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
  have e_1890_14 := arithEq_of_rows h (row := 1890) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_14
  simp only [← c115434, ← c115435, ← c115436] at e_1890_14
  have e_1890_15 := arithEq_of_rows h (row := 1890) (i := 15) rfl (by decide)
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
  have e_1890_16 := arithEq_of_rows h (row := 1890) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_16
  simp only [← c115440, ← c115441, ← c115442] at e_1890_16
  have e_1890_17 := arithEq_of_rows h (row := 1890) (i := 17) rfl (by decide)
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
  have e_1890_18 := arithEq_of_rows h (row := 1890) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1890_18
  simp only [← c115446, ← c115447, ← c115448] at e_1890_18
  have e_1890_19 := arithEq_of_rows h (row := 1890) (i := 19) rfl (by decide)
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
  have e_1891_0 := arithEq_of_rows h (row := 1891) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_0
  simp only [← c115452, ← c115453, ← c115454] at e_1891_0
  have e_1891_1 := arithEq_of_rows h (row := 1891) (i := 1) rfl (by decide)
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
  have e_1891_2 := arithEq_of_rows h (row := 1891) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_2
  simp only [← c115458, ← c115459, ← c115460] at e_1891_2
  have e_1891_3 := arithEq_of_rows h (row := 1891) (i := 3) rfl (by decide)
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
  have e_1891_4 := arithEq_of_rows h (row := 1891) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_4
  simp only [← c115464, ← c115465, ← c115466] at e_1891_4
  have e_1891_5 := arithEq_of_rows h (row := 1891) (i := 5) rfl (by decide)
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
  have e_1891_6 := arithEq_of_rows h (row := 1891) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_6
  simp only [← c115470, ← c115471, ← c115472] at e_1891_6
  have e_1891_7 := arithEq_of_rows h (row := 1891) (i := 7) rfl (by decide)
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
  have e_1891_8 := arithEq_of_rows h (row := 1891) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_8
  simp only [← c115476, ← c115477, ← c115478] at e_1891_8
  have e_1891_9 := arithEq_of_rows h (row := 1891) (i := 9) rfl (by decide)
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
  have e_1891_10 := arithEq_of_rows h (row := 1891) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_10
  simp only [← c115482, ← c115483, ← c115484] at e_1891_10
  have e_1891_11 := arithEq_of_rows h (row := 1891) (i := 11) rfl (by decide)
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
  have e_1891_12 := arithEq_of_rows h (row := 1891) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_12
  simp only [← c115488, ← c115489, ← c115490] at e_1891_12
  have e_1891_13 := arithEq_of_rows h (row := 1891) (i := 13) rfl (by decide)
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
  have e_1891_14 := arithEq_of_rows h (row := 1891) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_14
  simp only [← c115494, ← c115495, ← c115496] at e_1891_14
  have e_1891_15 := arithEq_of_rows h (row := 1891) (i := 15) rfl (by decide)
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
  have e_1891_16 := arithEq_of_rows h (row := 1891) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_16
  simp only [← c115500, ← c115501, ← c115502] at e_1891_16
  have e_1891_17 := arithEq_of_rows h (row := 1891) (i := 17) rfl (by decide)
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
  have e_1891_18 := arithEq_of_rows h (row := 1891) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1891_18
  simp only [← c115506, ← c115507, ← c115508] at e_1891_18
  have e_1891_19 := arithEq_of_rows h (row := 1891) (i := 19) rfl (by decide)
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
  have e_1892_0 := arithEq_of_rows h (row := 1892) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_0
  simp only [← c115512, ← c115513, ← c115514] at e_1892_0
  have e_1892_1 := arithEq_of_rows h (row := 1892) (i := 1) rfl (by decide)
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
  have e_1892_2 := arithEq_of_rows h (row := 1892) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_2
  simp only [← c115518, ← c115519, ← c115520] at e_1892_2
  have e_1892_3 := arithEq_of_rows h (row := 1892) (i := 3) rfl (by decide)
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
  have e_1892_4 := arithEq_of_rows h (row := 1892) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_4
  simp only [← c115524, ← c115525, ← c115526] at e_1892_4
  have e_1892_5 := arithEq_of_rows h (row := 1892) (i := 5) rfl (by decide)
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
  have e_1892_6 := arithEq_of_rows h (row := 1892) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_6
  simp only [← c115530, ← c115531, ← c115532] at e_1892_6
  have e_1892_7 := arithEq_of_rows h (row := 1892) (i := 7) rfl (by decide)
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
  have e_1892_8 := arithEq_of_rows h (row := 1892) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_8
  simp only [← c115536, ← c115537, ← c115538] at e_1892_8
  have e_1892_9 := arithEq_of_rows h (row := 1892) (i := 9) rfl (by decide)
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
  have e_1892_10 := arithEq_of_rows h (row := 1892) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_10
  simp only [← c115542, ← c115543, ← c115544] at e_1892_10
  have e_1892_11 := arithEq_of_rows h (row := 1892) (i := 11) rfl (by decide)
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
  have e_1892_12 := arithEq_of_rows h (row := 1892) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_12
  simp only [← c115548, ← c115549, ← c115550] at e_1892_12
  have e_1892_13 := arithEq_of_rows h (row := 1892) (i := 13) rfl (by decide)
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
  have e_1892_14 := arithEq_of_rows h (row := 1892) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_14
  simp only [← c115554, ← c115555, ← c115556] at e_1892_14
  have e_1892_15 := arithEq_of_rows h (row := 1892) (i := 15) rfl (by decide)
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
  have e_1892_16 := arithEq_of_rows h (row := 1892) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_16
  simp only [← c115560, ← c115561, ← c115562] at e_1892_16
  have e_1892_17 := arithEq_of_rows h (row := 1892) (i := 17) rfl (by decide)
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
  have e_1892_18 := arithEq_of_rows h (row := 1892) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1892_18
  simp only [← c115566, ← c115567, ← c115568] at e_1892_18
  have e_1892_19 := arithEq_of_rows h (row := 1892) (i := 19) rfl (by decide)
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
  have e_1893_0 := arithEq_of_rows h (row := 1893) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_0
  simp only [← c115572, ← c115573, ← c115574] at e_1893_0
  have e_1893_1 := arithEq_of_rows h (row := 1893) (i := 1) rfl (by decide)
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
  have e_1893_2 := arithEq_of_rows h (row := 1893) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_2
  simp only [← c115578, ← c115579, ← c115580] at e_1893_2
  have e_1893_3 := arithEq_of_rows h (row := 1893) (i := 3) rfl (by decide)
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
  have e_1893_4 := arithEq_of_rows h (row := 1893) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_4
  simp only [← c115584, ← c115585, ← c115586] at e_1893_4
  have e_1893_5 := arithEq_of_rows h (row := 1893) (i := 5) rfl (by decide)
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
  have e_1893_6 := arithEq_of_rows h (row := 1893) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_6
  simp only [← c115590, ← c115591, ← c115592] at e_1893_6
  have e_1893_7 := arithEq_of_rows h (row := 1893) (i := 7) rfl (by decide)
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
  have e_1893_8 := arithEq_of_rows h (row := 1893) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_8
  simp only [← c115596, ← c115597, ← c115598] at e_1893_8
  have e_1893_9 := arithEq_of_rows h (row := 1893) (i := 9) rfl (by decide)
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
  have e_1893_10 := arithEq_of_rows h (row := 1893) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_10
  simp only [← c115602, ← c115603, ← c115604] at e_1893_10
  have e_1893_11 := arithEq_of_rows h (row := 1893) (i := 11) rfl (by decide)
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
  have e_1893_12 := arithEq_of_rows h (row := 1893) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_12
  simp only [← c115608, ← c115609, ← c115610] at e_1893_12
  have e_1893_13 := arithEq_of_rows h (row := 1893) (i := 13) rfl (by decide)
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
  have e_1893_14 := arithEq_of_rows h (row := 1893) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_14
  simp only [← c115614, ← c115615, ← c115616] at e_1893_14
  have e_1893_15 := arithEq_of_rows h (row := 1893) (i := 15) rfl (by decide)
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
  have e_1893_16 := arithEq_of_rows h (row := 1893) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_16
  simp only [← c115620, ← c115621, ← c115622] at e_1893_16
  have e_1893_17 := arithEq_of_rows h (row := 1893) (i := 17) rfl (by decide)
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
  have e_1893_18 := arithEq_of_rows h (row := 1893) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1893_18
  simp only [← c115626, ← c115627, ← c115628] at e_1893_18
  have e_1893_19 := arithEq_of_rows h (row := 1893) (i := 19) rfl (by decide)
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
  have e_1894_0 := arithEq_of_rows h (row := 1894) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_0
  simp only [← c115632, ← c115633, ← c115634] at e_1894_0
  have e_1894_1 := arithEq_of_rows h (row := 1894) (i := 1) rfl (by decide)
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
  have e_1894_2 := arithEq_of_rows h (row := 1894) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_2
  simp only [← c115638, ← c115639, ← c115640] at e_1894_2
  have e_1894_3 := arithEq_of_rows h (row := 1894) (i := 3) rfl (by decide)
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
  have e_1894_4 := arithEq_of_rows h (row := 1894) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_4
  simp only [← c115644, ← c115645, ← c115646] at e_1894_4
  have e_1894_5 := arithEq_of_rows h (row := 1894) (i := 5) rfl (by decide)
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
  have e_1894_6 := arithEq_of_rows h (row := 1894) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_6
  simp only [← c115650, ← c115651, ← c115652] at e_1894_6
  have e_1894_7 := arithEq_of_rows h (row := 1894) (i := 7) rfl (by decide)
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
  have e_1894_8 := arithEq_of_rows h (row := 1894) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_8
  simp only [← c115656, ← c115657, ← c115658] at e_1894_8
  have e_1894_9 := arithEq_of_rows h (row := 1894) (i := 9) rfl (by decide)
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
  have e_1894_10 := arithEq_of_rows h (row := 1894) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_10
  simp only [← c115662, ← c115663, ← c115664] at e_1894_10
  have e_1894_11 := arithEq_of_rows h (row := 1894) (i := 11) rfl (by decide)
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
  have e_1894_12 := arithEq_of_rows h (row := 1894) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_12
  simp only [← c115668, ← c115669, ← c115670] at e_1894_12
  have e_1894_13 := arithEq_of_rows h (row := 1894) (i := 13) rfl (by decide)
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
  have e_1894_14 := arithEq_of_rows h (row := 1894) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_14
  simp only [← c115674, ← c115675, ← c115676] at e_1894_14
  have e_1894_15 := arithEq_of_rows h (row := 1894) (i := 15) rfl (by decide)
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
  have e_1894_16 := arithEq_of_rows h (row := 1894) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_16
  simp only [← c115680, ← c115681, ← c115682] at e_1894_16
  have e_1894_17 := arithEq_of_rows h (row := 1894) (i := 17) rfl (by decide)
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
  have e_1894_18 := arithEq_of_rows h (row := 1894) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1894_18
  simp only [← c115686, ← c115687, ← c115688] at e_1894_18
  have e_1894_19 := arithEq_of_rows h (row := 1894) (i := 19) rfl (by decide)
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
  have e_1895_0 := arithEq_of_rows h (row := 1895) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_0
  simp only [← c115692, ← c115693, ← c115694] at e_1895_0
  have e_1895_1 := arithEq_of_rows h (row := 1895) (i := 1) rfl (by decide)
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
  have e_1895_2 := arithEq_of_rows h (row := 1895) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_2
  simp only [← c115698, ← c115699, ← c115700] at e_1895_2
  have e_1895_3 := arithEq_of_rows h (row := 1895) (i := 3) rfl (by decide)
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
  have e_1895_4 := arithEq_of_rows h (row := 1895) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_4
  simp only [← c115704, ← c115705, ← c115706] at e_1895_4
  have e_1895_5 := arithEq_of_rows h (row := 1895) (i := 5) rfl (by decide)
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
  have e_1895_6 := arithEq_of_rows h (row := 1895) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_6
  simp only [← c115710, ← c115711, ← c115712] at e_1895_6
  have e_1895_7 := arithEq_of_rows h (row := 1895) (i := 7) rfl (by decide)
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
  have e_1895_8 := arithEq_of_rows h (row := 1895) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_8
  simp only [← c115716, ← c115717, ← c115718] at e_1895_8
  have e_1895_9 := arithEq_of_rows h (row := 1895) (i := 9) rfl (by decide)
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
  have e_1895_10 := arithEq_of_rows h (row := 1895) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_10
  simp only [← c115722, ← c115723, ← c115724] at e_1895_10
  have e_1895_11 := arithEq_of_rows h (row := 1895) (i := 11) rfl (by decide)
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
  have e_1895_12 := arithEq_of_rows h (row := 1895) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_12
  simp only [← c115728, ← c115729, ← c115730] at e_1895_12
  have e_1895_13 := arithEq_of_rows h (row := 1895) (i := 13) rfl (by decide)
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
  have e_1895_14 := arithEq_of_rows h (row := 1895) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_14
  simp only [← c115734, ← c115735, ← c115736] at e_1895_14
  have e_1895_15 := arithEq_of_rows h (row := 1895) (i := 15) rfl (by decide)
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
  have e_1895_16 := arithEq_of_rows h (row := 1895) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_16
  simp only [← c115740, ← c115741, ← c115742] at e_1895_16
  have e_1895_17 := arithEq_of_rows h (row := 1895) (i := 17) rfl (by decide)
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
  have e_1895_18 := arithEq_of_rows h (row := 1895) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1895_18
  simp only [← c115746, ← c115747, ← c115748] at e_1895_18
  have e_1895_19 := arithEq_of_rows h (row := 1895) (i := 19) rfl (by decide)
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
  have e_1896_0 := arithEq_of_rows h (row := 1896) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_0
  simp only [← c115752, ← c115753, ← c115754] at e_1896_0
  have e_1896_1 := arithEq_of_rows h (row := 1896) (i := 1) rfl (by decide)
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
  have e_1896_2 := arithEq_of_rows h (row := 1896) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_2
  simp only [← c115758, ← c115759, ← c115760] at e_1896_2
  have e_1896_3 := arithEq_of_rows h (row := 1896) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_3
  simp only [← c115761, ← c115762, k2, ← c115763] at e_1896_3
  have hr := e_1896_3
  simp only [e_1896_2] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
