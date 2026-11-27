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

theorem publicBatchWrapper89_f15942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1636 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16126)) := by
  have c100158 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100159 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100160 := (publicBatchWrapper89_copies3130 a h).1
  have c100161 := (publicBatchWrapper89_copies3130 a h).2.1
  have c100162 := (publicBatchWrapper89_copies3130 a h).2.2.1
  have c100163 := (publicBatchWrapper89_copies3130 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1636_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_2
  simp only [← c100158, ← c100159, ← c100160] at e_1636_2
  have e_1636_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_3
  simp only [← c100161, ← c100162, k2, ← c100163] at e_1636_3
  have hr := e_1636_3
  simp only [e_1636_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1636 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16127)) := by
  have c100164 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.1
  have c100165 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.1
  have c100166 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.1
  have c100167 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.1
  have c100168 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.1
  have c100169 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1636_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_4
  simp only [← c100164, ← c100165, ← c100166] at e_1636_4
  have e_1636_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_5
  simp only [← c100167, ← c100168, k2, ← c100169] at e_1636_5
  have hr := e_1636_5
  simp only [e_1636_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1636 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16128)) := by
  have c100170 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100171 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100172 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100173 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100174 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100175 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1636_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_6
  simp only [← c100170, ← c100171, ← c100172] at e_1636_6
  have e_1636_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_7
  simp only [← c100173, ← c100174, k2, ← c100175] at e_1636_7
  have hr := e_1636_7
  simp only [e_1636_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1636 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16129)) := by
  have c100176 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100177 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100178 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100179 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100180 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100181 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1636_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_8
  simp only [← c100176, ← c100177, ← c100178] at e_1636_8
  have e_1636_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_9
  simp only [← c100179, ← c100180, k2, ← c100181] at e_1636_9
  have hr := e_1636_9
  simp only [e_1636_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1636 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16130)) := by
  have c100182 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100183 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100184 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100185 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100186 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100187 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1636_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_10
  simp only [← c100182, ← c100183, ← c100184] at e_1636_10
  have e_1636_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_11
  simp only [← c100185, ← c100186, k2, ← c100187] at e_1636_11
  have hr := e_1636_11
  simp only [e_1636_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1636 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16131)) := by
  have c100188 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100189 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100190 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100191 := (publicBatchWrapper89_copies3130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100192 := (publicBatchWrapper89_copies3131 a h).1
  have c100193 := (publicBatchWrapper89_copies3131 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1636_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_12
  simp only [← c100188, ← c100189, ← c100190] at e_1636_12
  have e_1636_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_13
  simp only [← c100191, ← c100192, k2, ← c100193] at e_1636_13
  have hr := e_1636_13
  simp only [e_1636_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1636 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16132)) := by
  have c100194 := (publicBatchWrapper89_copies3131 a h).2.2.1
  have c100195 := (publicBatchWrapper89_copies3131 a h).2.2.2.1
  have c100196 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.1
  have c100197 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.1
  have c100198 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.1
  have c100199 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1636_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_14
  simp only [← c100194, ← c100195, ← c100196] at e_1636_14
  have e_1636_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_15
  simp only [← c100197, ← c100198, k2, ← c100199] at e_1636_15
  have hr := e_1636_15
  simp only [e_1636_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1636 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16133)) := by
  have c100200 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.1
  have c100201 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.1
  have c100202 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100203 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100204 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100205 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1636_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_16
  simp only [← c100200, ← c100201, ← c100202] at e_1636_16
  have e_1636_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_17
  simp only [← c100203, ← c100204, k2, ← c100205] at e_1636_17
  have hr := e_1636_17
  simp only [e_1636_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1636 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16134)) := by
  have c100206 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100207 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100208 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100209 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100210 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100211 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1636_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_18
  simp only [← c100206, ← c100207, ← c100208] at e_1636_18
  have e_1636_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_19
  simp only [← c100209, ← c100210, k2, ← c100211] at e_1636_19
  have hr := e_1636_19
  simp only [e_1636_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1637 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16135)) := by
  have c100212 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100213 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100214 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100215 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100216 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100217 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1637_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_0
  simp only [← c100212, ← c100213, ← c100214] at e_1637_0
  have e_1637_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_1
  simp only [← c100215, ← c100216, k2, ← c100217] at e_1637_1
  have hr := e_1637_1
  simp only [e_1637_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1637 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16136)) := by
  have c100218 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100219 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100220 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100221 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100222 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100223 := (publicBatchWrapper89_copies3131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1637_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_2
  simp only [← c100218, ← c100219, ← c100220] at e_1637_2
  have e_1637_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_3
  simp only [← c100221, ← c100222, k2, ← c100223] at e_1637_3
  have hr := e_1637_3
  simp only [e_1637_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1637 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16137)) := by
  have c100224 := (publicBatchWrapper89_copies3132 a h).1
  have c100225 := (publicBatchWrapper89_copies3132 a h).2.1
  have c100226 := (publicBatchWrapper89_copies3132 a h).2.2.1
  have c100227 := (publicBatchWrapper89_copies3132 a h).2.2.2.1
  have c100228 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.1
  have c100229 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1637_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_4
  simp only [← c100224, ← c100225, ← c100226] at e_1637_4
  have e_1637_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_5
  simp only [← c100227, ← c100228, k2, ← c100229] at e_1637_5
  have hr := e_1637_5
  simp only [e_1637_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1637 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16138)) := by
  have c100230 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.1
  have c100231 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.1
  have c100232 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.1
  have c100233 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.1
  have c100234 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100235 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1637_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_6
  simp only [← c100230, ← c100231, ← c100232] at e_1637_6
  have e_1637_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_7
  simp only [← c100233, ← c100234, k2, ← c100235] at e_1637_7
  have hr := e_1637_7
  simp only [e_1637_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1637 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16139)) := by
  have c100236 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100237 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100238 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100239 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100240 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100241 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1637_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_8
  simp only [← c100236, ← c100237, ← c100238] at e_1637_8
  have e_1637_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_9
  simp only [← c100239, ← c100240, k2, ← c100241] at e_1637_9
  have hr := e_1637_9
  simp only [e_1637_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1637 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16140)) := by
  have c100242 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100243 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100244 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100245 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100246 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100247 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1637_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_10
  simp only [← c100242, ← c100243, ← c100244] at e_1637_10
  have e_1637_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_11
  simp only [← c100245, ← c100246, k2, ← c100247] at e_1637_11
  have hr := e_1637_11
  simp only [e_1637_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1637 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16141)) := by
  have c100248 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100249 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100250 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100251 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100252 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100253 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1637_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_12
  simp only [← c100248, ← c100249, ← c100250] at e_1637_12
  have e_1637_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_13
  simp only [← c100251, ← c100252, k2, ← c100253] at e_1637_13
  have hr := e_1637_13
  simp only [e_1637_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1637 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16142)) := by
  have c100254 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100255 := (publicBatchWrapper89_copies3132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100256 := (publicBatchWrapper89_copies3133 a h).1
  have c100257 := (publicBatchWrapper89_copies3133 a h).2.1
  have c100258 := (publicBatchWrapper89_copies3133 a h).2.2.1
  have c100259 := (publicBatchWrapper89_copies3133 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1637_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_14
  simp only [← c100254, ← c100255, ← c100256] at e_1637_14
  have e_1637_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_15
  simp only [← c100257, ← c100258, k2, ← c100259] at e_1637_15
  have hr := e_1637_15
  simp only [e_1637_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1637 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16143)) := by
  have c100260 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.1
  have c100261 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.1
  have c100262 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.1
  have c100263 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.1
  have c100264 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.1
  have c100265 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1637_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_16
  simp only [← c100260, ← c100261, ← c100262] at e_1637_16
  have e_1637_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_17
  simp only [← c100263, ← c100264, k2, ← c100265] at e_1637_17
  have hr := e_1637_17
  simp only [e_1637_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1637 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16144)) := by
  have c100266 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100267 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100268 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100269 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100270 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100271 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1637_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_18
  simp only [← c100266, ← c100267, ← c100268] at e_1637_18
  have e_1637_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1637 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1637_19
  simp only [← c100269, ← c100270, k2, ← c100271] at e_1637_19
  have hr := e_1637_19
  simp only [e_1637_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1638 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16145)) := by
  have c100272 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100273 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100274 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100275 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100276 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100277 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1638_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_0
  simp only [← c100272, ← c100273, ← c100274] at e_1638_0
  have e_1638_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_1
  simp only [← c100275, ← c100276, k2, ← c100277] at e_1638_1
  have hr := e_1638_1
  simp only [e_1638_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1638 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16146)) := by
  have c100278 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100279 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100280 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100281 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100282 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100283 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1638_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_2
  simp only [← c100278, ← c100279, ← c100280] at e_1638_2
  have e_1638_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_3
  simp only [← c100281, ← c100282, k2, ← c100283] at e_1638_3
  have hr := e_1638_3
  simp only [e_1638_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1638 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16147)) := by
  have c100284 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100285 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100286 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100287 := (publicBatchWrapper89_copies3133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100288 := (publicBatchWrapper89_copies3134 a h).1
  have c100289 := (publicBatchWrapper89_copies3134 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1638_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_4
  simp only [← c100284, ← c100285, ← c100286] at e_1638_4
  have e_1638_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_5
  simp only [← c100287, ← c100288, k2, ← c100289] at e_1638_5
  have hr := e_1638_5
  simp only [e_1638_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1638 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16148)) := by
  have c100290 := (publicBatchWrapper89_copies3134 a h).2.2.1
  have c100291 := (publicBatchWrapper89_copies3134 a h).2.2.2.1
  have c100292 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.1
  have c100293 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.1
  have c100294 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.1
  have c100295 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1638_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_6
  simp only [← c100290, ← c100291, ← c100292] at e_1638_6
  have e_1638_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_7
  simp only [← c100293, ← c100294, k2, ← c100295] at e_1638_7
  have hr := e_1638_7
  simp only [e_1638_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1638 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16149)) := by
  have c100296 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.1
  have c100297 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.1
  have c100298 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100299 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100300 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100301 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1638_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_8
  simp only [← c100296, ← c100297, ← c100298] at e_1638_8
  have e_1638_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_9
  simp only [← c100299, ← c100300, k2, ← c100301] at e_1638_9
  have hr := e_1638_9
  simp only [e_1638_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1638 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16150)) := by
  have c100302 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100303 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100304 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100305 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100306 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100307 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1638_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_10
  simp only [← c100302, ← c100303, ← c100304] at e_1638_10
  have e_1638_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_11
  simp only [← c100305, ← c100306, k2, ← c100307] at e_1638_11
  have hr := e_1638_11
  simp only [e_1638_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1638 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16151)) := by
  have c100308 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100309 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100310 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100311 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100312 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100313 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1638_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_12
  simp only [← c100308, ← c100309, ← c100310] at e_1638_12
  have e_1638_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_13
  simp only [← c100311, ← c100312, k2, ← c100313] at e_1638_13
  have hr := e_1638_13
  simp only [e_1638_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1638 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16152)) := by
  have c100314 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100315 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100316 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100317 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100318 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100319 := (publicBatchWrapper89_copies3134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1638_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_14
  simp only [← c100314, ← c100315, ← c100316] at e_1638_14
  have e_1638_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_15
  simp only [← c100317, ← c100318, k2, ← c100319] at e_1638_15
  have hr := e_1638_15
  simp only [e_1638_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1638 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16153)) := by
  have c100320 := (publicBatchWrapper89_copies3135 a h).1
  have c100321 := (publicBatchWrapper89_copies3135 a h).2.1
  have c100322 := (publicBatchWrapper89_copies3135 a h).2.2.1
  have c100323 := (publicBatchWrapper89_copies3135 a h).2.2.2.1
  have c100324 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.1
  have c100325 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1638_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_16
  simp only [← c100320, ← c100321, ← c100322] at e_1638_16
  have e_1638_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_17
  simp only [← c100323, ← c100324, k2, ← c100325] at e_1638_17
  have hr := e_1638_17
  simp only [e_1638_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1638 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16154)) := by
  have c100326 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.1
  have c100327 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.1
  have c100328 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.1
  have c100329 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.1
  have c100330 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100331 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1638_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_18
  simp only [← c100326, ← c100327, ← c100328] at e_1638_18
  have e_1638_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1638 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1638_19
  simp only [← c100329, ← c100330, k2, ← c100331] at e_1638_19
  have hr := e_1638_19
  simp only [e_1638_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1639 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16155)) := by
  have c100332 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100333 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100334 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100335 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100336 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100337 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1639_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_0
  simp only [← c100332, ← c100333, ← c100334] at e_1639_0
  have e_1639_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_1
  simp only [← c100335, ← c100336, k2, ← c100337] at e_1639_1
  have hr := e_1639_1
  simp only [e_1639_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1639 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16156)) := by
  have c100338 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100339 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100340 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100341 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100342 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100343 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1639_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_2
  simp only [← c100338, ← c100339, ← c100340] at e_1639_2
  have e_1639_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_3
  simp only [← c100341, ← c100342, k2, ← c100343] at e_1639_3
  have hr := e_1639_3
  simp only [e_1639_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1639 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16157)) := by
  have c100344 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100345 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100346 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100347 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100348 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100349 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1639_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_4
  simp only [← c100344, ← c100345, ← c100346] at e_1639_4
  have e_1639_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_5
  simp only [← c100347, ← c100348, k2, ← c100349] at e_1639_5
  have hr := e_1639_5
  simp only [e_1639_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1639 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16158)) := by
  have c100350 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100351 := (publicBatchWrapper89_copies3135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100352 := (publicBatchWrapper89_copies3136 a h).1
  have c100353 := (publicBatchWrapper89_copies3136 a h).2.1
  have c100354 := (publicBatchWrapper89_copies3136 a h).2.2.1
  have c100355 := (publicBatchWrapper89_copies3136 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1639_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_6
  simp only [← c100350, ← c100351, ← c100352] at e_1639_6
  have e_1639_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_7
  simp only [← c100353, ← c100354, k2, ← c100355] at e_1639_7
  have hr := e_1639_7
  simp only [e_1639_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1639 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16159)) := by
  have c100356 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.1
  have c100357 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.1
  have c100358 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.1
  have c100359 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.1
  have c100360 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.1
  have c100361 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1639_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_8
  simp only [← c100356, ← c100357, ← c100358] at e_1639_8
  have e_1639_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_9
  simp only [← c100359, ← c100360, k2, ← c100361] at e_1639_9
  have hr := e_1639_9
  simp only [e_1639_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1639 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16160)) := by
  have c100362 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100363 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100364 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100365 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100366 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100367 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1639_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_10
  simp only [← c100362, ← c100363, ← c100364] at e_1639_10
  have e_1639_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_11
  simp only [← c100365, ← c100366, k2, ← c100367] at e_1639_11
  have hr := e_1639_11
  simp only [e_1639_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1639 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16161)) := by
  have c100368 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100369 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100370 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100371 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100372 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100373 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1639_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_12
  simp only [← c100368, ← c100369, ← c100370] at e_1639_12
  have e_1639_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_13
  simp only [← c100371, ← c100372, k2, ← c100373] at e_1639_13
  have hr := e_1639_13
  simp only [e_1639_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1639 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16162)) := by
  have c100374 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100375 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100376 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100377 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100378 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100379 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1639_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_14
  simp only [← c100374, ← c100375, ← c100376] at e_1639_14
  have e_1639_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_15
  simp only [← c100377, ← c100378, k2, ← c100379] at e_1639_15
  have hr := e_1639_15
  simp only [e_1639_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1639 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16163)) := by
  have c100380 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100381 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100382 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100383 := (publicBatchWrapper89_copies3136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100384 := (publicBatchWrapper89_copies3137 a h).1
  have c100385 := (publicBatchWrapper89_copies3137 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1639_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_16
  simp only [← c100380, ← c100381, ← c100382] at e_1639_16
  have e_1639_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_17
  simp only [← c100383, ← c100384, k2, ← c100385] at e_1639_17
  have hr := e_1639_17
  simp only [e_1639_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1639 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16164)) := by
  have c100386 := (publicBatchWrapper89_copies3137 a h).2.2.1
  have c100387 := (publicBatchWrapper89_copies3137 a h).2.2.2.1
  have c100388 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.1
  have c100389 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.1
  have c100390 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.1
  have c100391 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1639_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_18
  simp only [← c100386, ← c100387, ← c100388] at e_1639_18
  have e_1639_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1639 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1639_19
  simp only [← c100389, ← c100390, k2, ← c100391] at e_1639_19
  have hr := e_1639_19
  simp only [e_1639_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1640 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16165)) := by
  have c100392 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.1
  have c100393 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.1
  have c100394 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100395 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100396 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100397 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1640_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_0
  simp only [← c100392, ← c100393, ← c100394] at e_1640_0
  have e_1640_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_1
  simp only [← c100395, ← c100396, k2, ← c100397] at e_1640_1
  have hr := e_1640_1
  simp only [e_1640_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1640 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16166)) := by
  have c100398 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100399 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100400 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100401 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100402 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100403 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1640_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_2
  simp only [← c100398, ← c100399, ← c100400] at e_1640_2
  have e_1640_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_3
  simp only [← c100401, ← c100402, k2, ← c100403] at e_1640_3
  have hr := e_1640_3
  simp only [e_1640_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1640 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16167)) := by
  have c100404 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100405 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100406 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100407 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100408 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100409 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1640_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_4
  simp only [← c100404, ← c100405, ← c100406] at e_1640_4
  have e_1640_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_5
  simp only [← c100407, ← c100408, k2, ← c100409] at e_1640_5
  have hr := e_1640_5
  simp only [e_1640_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1640 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16168)) := by
  have c100410 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100411 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100412 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100413 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100414 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100415 := (publicBatchWrapper89_copies3137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1640_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_6
  simp only [← c100410, ← c100411, ← c100412] at e_1640_6
  have e_1640_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_7
  simp only [← c100413, ← c100414, k2, ← c100415] at e_1640_7
  have hr := e_1640_7
  simp only [e_1640_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1640 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16169)) := by
  have c100416 := (publicBatchWrapper89_copies3138 a h).1
  have c100417 := (publicBatchWrapper89_copies3138 a h).2.1
  have c100418 := (publicBatchWrapper89_copies3138 a h).2.2.1
  have c100419 := (publicBatchWrapper89_copies3138 a h).2.2.2.1
  have c100420 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.1
  have c100421 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1640_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_8
  simp only [← c100416, ← c100417, ← c100418] at e_1640_8
  have e_1640_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_9
  simp only [← c100419, ← c100420, k2, ← c100421] at e_1640_9
  have hr := e_1640_9
  simp only [e_1640_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1640 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16210)) := by
  have c100422 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.1
  have c100423 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.1
  have c100424 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.1
  have c100425 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.1
  have c100426 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100427 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1640_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_10
  simp only [← c100422, ← c100423, ← c100424] at e_1640_10
  have e_1640_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_11
  simp only [← c100425, ← c100426, k2, ← c100427] at e_1640_11
  have hr := e_1640_11
  simp only [e_1640_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1640 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16211)) := by
  have c100428 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100429 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100430 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100431 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100432 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100433 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1640_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_12
  simp only [← c100428, ← c100429, ← c100430] at e_1640_12
  have e_1640_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_13
  simp only [← c100431, ← c100432, k2, ← c100433] at e_1640_13
  have hr := e_1640_13
  simp only [e_1640_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1640 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16212)) := by
  have c100434 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100435 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100436 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100437 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100438 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100439 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1640_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_14
  simp only [← c100434, ← c100435, ← c100436] at e_1640_14
  have e_1640_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_15
  simp only [← c100437, ← c100438, k2, ← c100439] at e_1640_15
  have hr := e_1640_15
  simp only [e_1640_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1640 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16213)) := by
  have c100440 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100441 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100442 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100443 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100444 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100445 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1640_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_16
  simp only [← c100440, ← c100441, ← c100442] at e_1640_16
  have e_1640_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_17
  simp only [← c100443, ← c100444, k2, ← c100445] at e_1640_17
  have hr := e_1640_17
  simp only [e_1640_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1640 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16214)) := by
  have c100446 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100447 := (publicBatchWrapper89_copies3138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100448 := (publicBatchWrapper89_copies3139 a h).1
  have c100449 := (publicBatchWrapper89_copies3139 a h).2.1
  have c100450 := (publicBatchWrapper89_copies3139 a h).2.2.1
  have c100451 := (publicBatchWrapper89_copies3139 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1640_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_18
  simp only [← c100446, ← c100447, ← c100448] at e_1640_18
  have e_1640_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1640 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1640_19
  simp only [← c100449, ← c100450, k2, ← c100451] at e_1640_19
  have hr := e_1640_19
  simp only [e_1640_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1641 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16215)) := by
  have c100452 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.1
  have c100453 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.1
  have c100454 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.1
  have c100455 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.1
  have c100456 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.1
  have c100457 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1641_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_0
  simp only [← c100452, ← c100453, ← c100454] at e_1641_0
  have e_1641_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_1
  simp only [← c100455, ← c100456, k2, ← c100457] at e_1641_1
  have hr := e_1641_1
  simp only [e_1641_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1641 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16216)) := by
  have c100458 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100459 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100460 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100461 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100462 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100463 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1641_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_2
  simp only [← c100458, ← c100459, ← c100460] at e_1641_2
  have e_1641_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_3
  simp only [← c100461, ← c100462, k2, ← c100463] at e_1641_3
  have hr := e_1641_3
  simp only [e_1641_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1641 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16217)) := by
  have c100464 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100465 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100466 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100467 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100468 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100469 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1641_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_4
  simp only [← c100464, ← c100465, ← c100466] at e_1641_4
  have e_1641_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_5
  simp only [← c100467, ← c100468, k2, ← c100469] at e_1641_5
  have hr := e_1641_5
  simp only [e_1641_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1641 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16218)) := by
  have c100470 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100471 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100472 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100473 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100474 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100475 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1641_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_6
  simp only [← c100470, ← c100471, ← c100472] at e_1641_6
  have e_1641_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_7
  simp only [← c100473, ← c100474, k2, ← c100475] at e_1641_7
  have hr := e_1641_7
  simp only [e_1641_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1641 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16219)) := by
  have c100476 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100477 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100478 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100479 := (publicBatchWrapper89_copies3139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100480 := (publicBatchWrapper89_copies3140 a h).1
  have c100481 := (publicBatchWrapper89_copies3140 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1641_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_8
  simp only [← c100476, ← c100477, ← c100478] at e_1641_8
  have e_1641_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_9
  simp only [← c100479, ← c100480, k2, ← c100481] at e_1641_9
  have hr := e_1641_9
  simp only [e_1641_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1641 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16220)) := by
  have c100482 := (publicBatchWrapper89_copies3140 a h).2.2.1
  have c100483 := (publicBatchWrapper89_copies3140 a h).2.2.2.1
  have c100484 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.1
  have c100485 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.1
  have c100486 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.1
  have c100487 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1641_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_10
  simp only [← c100482, ← c100483, ← c100484] at e_1641_10
  have e_1641_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_11
  simp only [← c100485, ← c100486, k2, ← c100487] at e_1641_11
  have hr := e_1641_11
  simp only [e_1641_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1641 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16221)) := by
  have c100488 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.1
  have c100489 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.1
  have c100490 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100491 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100492 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100493 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1641_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_12
  simp only [← c100488, ← c100489, ← c100490] at e_1641_12
  have e_1641_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_13
  simp only [← c100491, ← c100492, k2, ← c100493] at e_1641_13
  have hr := e_1641_13
  simp only [e_1641_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1641 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16222)) := by
  have c100494 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100495 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100496 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100497 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100498 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100499 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1641_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_14
  simp only [← c100494, ← c100495, ← c100496] at e_1641_14
  have e_1641_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_15
  simp only [← c100497, ← c100498, k2, ← c100499] at e_1641_15
  have hr := e_1641_15
  simp only [e_1641_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1641 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16223)) := by
  have c100500 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100501 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100502 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100503 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100504 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100505 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1641_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_16
  simp only [← c100500, ← c100501, ← c100502] at e_1641_16
  have e_1641_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_17
  simp only [← c100503, ← c100504, k2, ← c100505] at e_1641_17
  have hr := e_1641_17
  simp only [e_1641_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1641 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16224)) := by
  have c100506 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100507 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100508 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100509 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100510 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100511 := (publicBatchWrapper89_copies3140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1641_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_18
  simp only [← c100506, ← c100507, ← c100508] at e_1641_18
  have e_1641_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1641 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1641_19
  simp only [← c100509, ← c100510, k2, ← c100511] at e_1641_19
  have hr := e_1641_19
  simp only [e_1641_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1642 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16225)) := by
  have c100512 := (publicBatchWrapper89_copies3141 a h).1
  have c100513 := (publicBatchWrapper89_copies3141 a h).2.1
  have c100514 := (publicBatchWrapper89_copies3141 a h).2.2.1
  have c100515 := (publicBatchWrapper89_copies3141 a h).2.2.2.1
  have c100516 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.1
  have c100517 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1642_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_0
  simp only [← c100512, ← c100513, ← c100514] at e_1642_0
  have e_1642_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_1
  simp only [← c100515, ← c100516, k2, ← c100517] at e_1642_1
  have hr := e_1642_1
  simp only [e_1642_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1642 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16226)) := by
  have c100518 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.1
  have c100519 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.1
  have c100520 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.1
  have c100521 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.1
  have c100522 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100523 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1642_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_2
  simp only [← c100518, ← c100519, ← c100520] at e_1642_2
  have e_1642_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_3
  simp only [← c100521, ← c100522, k2, ← c100523] at e_1642_3
  have hr := e_1642_3
  simp only [e_1642_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1642 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16227)) := by
  have c100524 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100525 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100526 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100527 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100528 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100529 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1642_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_4
  simp only [← c100524, ← c100525, ← c100526] at e_1642_4
  have e_1642_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_5
  simp only [← c100527, ← c100528, k2, ← c100529] at e_1642_5
  have hr := e_1642_5
  simp only [e_1642_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1642 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16228)) := by
  have c100530 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100531 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100532 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100533 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100534 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100535 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1642_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_6
  simp only [← c100530, ← c100531, ← c100532] at e_1642_6
  have e_1642_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_7
  simp only [← c100533, ← c100534, k2, ← c100535] at e_1642_7
  have hr := e_1642_7
  simp only [e_1642_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1642 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16229)) := by
  have c100536 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100537 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100538 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100539 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100540 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100541 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1642_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_8
  simp only [← c100536, ← c100537, ← c100538] at e_1642_8
  have e_1642_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_9
  simp only [← c100539, ← c100540, k2, ← c100541] at e_1642_9
  have hr := e_1642_9
  simp only [e_1642_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1642 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16230)) := by
  have c100542 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100543 := (publicBatchWrapper89_copies3141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100544 := (publicBatchWrapper89_copies3142 a h).1
  have c100545 := (publicBatchWrapper89_copies3142 a h).2.1
  have c100546 := (publicBatchWrapper89_copies3142 a h).2.2.1
  have c100547 := (publicBatchWrapper89_copies3142 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1642_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_10
  simp only [← c100542, ← c100543, ← c100544] at e_1642_10
  have e_1642_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_11
  simp only [← c100545, ← c100546, k2, ← c100547] at e_1642_11
  have hr := e_1642_11
  simp only [e_1642_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1642 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16231)) := by
  have c100548 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.1
  have c100549 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.1
  have c100550 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.1
  have c100551 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.1
  have c100552 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.1
  have c100553 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1642_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_12
  simp only [← c100548, ← c100549, ← c100550] at e_1642_12
  have e_1642_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_13
  simp only [← c100551, ← c100552, k2, ← c100553] at e_1642_13
  have hr := e_1642_13
  simp only [e_1642_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1642 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16232)) := by
  have c100554 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100555 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100556 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100557 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100558 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100559 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1642_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_14
  simp only [← c100554, ← c100555, ← c100556] at e_1642_14
  have e_1642_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_15
  simp only [← c100557, ← c100558, k2, ← c100559] at e_1642_15
  have hr := e_1642_15
  simp only [e_1642_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1642 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16233)) := by
  have c100560 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100561 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100562 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100563 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100564 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100565 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1642_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_16
  simp only [← c100560, ← c100561, ← c100562] at e_1642_16
  have e_1642_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_17
  simp only [← c100563, ← c100564, k2, ← c100565] at e_1642_17
  have hr := e_1642_17
  simp only [e_1642_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1642 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16234)) := by
  have c100566 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100567 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100568 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100569 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100570 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100571 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1642_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_18
  simp only [← c100566, ← c100567, ← c100568] at e_1642_18
  have e_1642_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1642 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1642_19
  simp only [← c100569, ← c100570, k2, ← c100571] at e_1642_19
  have hr := e_1642_19
  simp only [e_1642_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1643 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16235)) := by
  have c100572 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100573 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100574 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100575 := (publicBatchWrapper89_copies3142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100576 := (publicBatchWrapper89_copies3143 a h).1
  have c100577 := (publicBatchWrapper89_copies3143 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1643_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_0
  simp only [← c100572, ← c100573, ← c100574] at e_1643_0
  have e_1643_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_1
  simp only [← c100575, ← c100576, k2, ← c100577] at e_1643_1
  have hr := e_1643_1
  simp only [e_1643_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1643 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16236)) := by
  have c100578 := (publicBatchWrapper89_copies3143 a h).2.2.1
  have c100579 := (publicBatchWrapper89_copies3143 a h).2.2.2.1
  have c100580 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.1
  have c100581 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.1
  have c100582 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.1
  have c100583 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1643_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_2
  simp only [← c100578, ← c100579, ← c100580] at e_1643_2
  have e_1643_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_3
  simp only [← c100581, ← c100582, k2, ← c100583] at e_1643_3
  have hr := e_1643_3
  simp only [e_1643_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1643 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16237)) := by
  have c100584 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.1
  have c100585 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.1
  have c100586 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100587 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100588 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100589 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1643_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_4
  simp only [← c100584, ← c100585, ← c100586] at e_1643_4
  have e_1643_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_5
  simp only [← c100587, ← c100588, k2, ← c100589] at e_1643_5
  have hr := e_1643_5
  simp only [e_1643_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1643 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16238)) := by
  have c100590 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100591 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100592 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100593 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100594 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100595 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1643_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_6
  simp only [← c100590, ← c100591, ← c100592] at e_1643_6
  have e_1643_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_7
  simp only [← c100593, ← c100594, k2, ← c100595] at e_1643_7
  have hr := e_1643_7
  simp only [e_1643_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1643 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16239)) := by
  have c100596 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100597 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100598 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100599 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100600 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100601 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1643_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_8
  simp only [← c100596, ← c100597, ← c100598] at e_1643_8
  have e_1643_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_9
  simp only [← c100599, ← c100600, k2, ← c100601] at e_1643_9
  have hr := e_1643_9
  simp only [e_1643_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1643 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16240)) := by
  have c100602 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100603 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100604 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100605 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100606 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100607 := (publicBatchWrapper89_copies3143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1643_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_10
  simp only [← c100602, ← c100603, ← c100604] at e_1643_10
  have e_1643_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_11
  simp only [← c100605, ← c100606, k2, ← c100607] at e_1643_11
  have hr := e_1643_11
  simp only [e_1643_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1643 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16241)) := by
  have c100608 := (publicBatchWrapper89_copies3144 a h).1
  have c100609 := (publicBatchWrapper89_copies3144 a h).2.1
  have c100610 := (publicBatchWrapper89_copies3144 a h).2.2.1
  have c100611 := (publicBatchWrapper89_copies3144 a h).2.2.2.1
  have c100612 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.1
  have c100613 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1643_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_12
  simp only [← c100608, ← c100609, ← c100610] at e_1643_12
  have e_1643_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_13
  simp only [← c100611, ← c100612, k2, ← c100613] at e_1643_13
  have hr := e_1643_13
  simp only [e_1643_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1643 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16242)) := by
  have c100614 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.1
  have c100615 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.1
  have c100616 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.1
  have c100617 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.1
  have c100618 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100619 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1643_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_14
  simp only [← c100614, ← c100615, ← c100616] at e_1643_14
  have e_1643_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_15
  simp only [← c100617, ← c100618, k2, ← c100619] at e_1643_15
  have hr := e_1643_15
  simp only [e_1643_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1643 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16243)) := by
  have c100620 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100621 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100622 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100623 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100624 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100625 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1643_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_16
  simp only [← c100620, ← c100621, ← c100622] at e_1643_16
  have e_1643_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_17
  simp only [← c100623, ← c100624, k2, ← c100625] at e_1643_17
  have hr := e_1643_17
  simp only [e_1643_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1643 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16244)) := by
  have c100626 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100627 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100628 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100629 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100630 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100631 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1643_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_18
  simp only [← c100626, ← c100627, ← c100628] at e_1643_18
  have e_1643_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1643 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1643_19
  simp only [← c100629, ← c100630, k2, ← c100631] at e_1643_19
  have hr := e_1643_19
  simp only [e_1643_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1644 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16245)) := by
  have c100632 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100633 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100634 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100635 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100636 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100637 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1644_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_0
  simp only [← c100632, ← c100633, ← c100634] at e_1644_0
  have e_1644_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_1
  simp only [← c100635, ← c100636, k2, ← c100637] at e_1644_1
  have hr := e_1644_1
  simp only [e_1644_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1644 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16246)) := by
  have c100638 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100639 := (publicBatchWrapper89_copies3144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100640 := (publicBatchWrapper89_copies3145 a h).1
  have c100641 := (publicBatchWrapper89_copies3145 a h).2.1
  have c100642 := (publicBatchWrapper89_copies3145 a h).2.2.1
  have c100643 := (publicBatchWrapper89_copies3145 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1644_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_2
  simp only [← c100638, ← c100639, ← c100640] at e_1644_2
  have e_1644_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_3
  simp only [← c100641, ← c100642, k2, ← c100643] at e_1644_3
  have hr := e_1644_3
  simp only [e_1644_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1644 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16247)) := by
  have c100644 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.1
  have c100645 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.1
  have c100646 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.1
  have c100647 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.1
  have c100648 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.1
  have c100649 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1644_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_4
  simp only [← c100644, ← c100645, ← c100646] at e_1644_4
  have e_1644_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_5
  simp only [← c100647, ← c100648, k2, ← c100649] at e_1644_5
  have hr := e_1644_5
  simp only [e_1644_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1644 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16248)) := by
  have c100650 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100651 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100652 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100653 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100654 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100655 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1644_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_6
  simp only [← c100650, ← c100651, ← c100652] at e_1644_6
  have e_1644_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_7
  simp only [← c100653, ← c100654, k2, ← c100655] at e_1644_7
  have hr := e_1644_7
  simp only [e_1644_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1644 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16249)) := by
  have c100656 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100657 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100658 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100659 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100660 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100661 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1644_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_8
  simp only [← c100656, ← c100657, ← c100658] at e_1644_8
  have e_1644_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_9
  simp only [← c100659, ← c100660, k2, ← c100661] at e_1644_9
  have hr := e_1644_9
  simp only [e_1644_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1644 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16250)) := by
  have c100662 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100663 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100664 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100665 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100666 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100667 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1644_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_10
  simp only [← c100662, ← c100663, ← c100664] at e_1644_10
  have e_1644_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_11
  simp only [← c100665, ← c100666, k2, ← c100667] at e_1644_11
  have hr := e_1644_11
  simp only [e_1644_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1644 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16251)) := by
  have c100668 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100669 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100670 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100671 := (publicBatchWrapper89_copies3145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100672 := (publicBatchWrapper89_copies3146 a h).1
  have c100673 := (publicBatchWrapper89_copies3146 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1644_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_12
  simp only [← c100668, ← c100669, ← c100670] at e_1644_12
  have e_1644_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_13
  simp only [← c100671, ← c100672, k2, ← c100673] at e_1644_13
  have hr := e_1644_13
  simp only [e_1644_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1644 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16252)) := by
  have c100674 := (publicBatchWrapper89_copies3146 a h).2.2.1
  have c100675 := (publicBatchWrapper89_copies3146 a h).2.2.2.1
  have c100676 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.1
  have c100677 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.1
  have c100678 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.1
  have c100679 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1644_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_14
  simp only [← c100674, ← c100675, ← c100676] at e_1644_14
  have e_1644_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_15
  simp only [← c100677, ← c100678, k2, ← c100679] at e_1644_15
  have hr := e_1644_15
  simp only [e_1644_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1644 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16253)) := by
  have c100680 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.1
  have c100681 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.1
  have c100682 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100683 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100684 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100685 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1644_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_16
  simp only [← c100680, ← c100681, ← c100682] at e_1644_16
  have e_1644_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_17
  simp only [← c100683, ← c100684, k2, ← c100685] at e_1644_17
  have hr := e_1644_17
  simp only [e_1644_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1644 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16254)) := by
  have c100686 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100687 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100688 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100689 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100690 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100691 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1644_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_18
  simp only [← c100686, ← c100687, ← c100688] at e_1644_18
  have e_1644_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1644 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1644_19
  simp only [← c100689, ← c100690, k2, ← c100691] at e_1644_19
  have hr := e_1644_19
  simp only [e_1644_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1645 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16255)) := by
  have c100692 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100693 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100694 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100695 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100696 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100697 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1645_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_0
  simp only [← c100692, ← c100693, ← c100694] at e_1645_0
  have e_1645_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_1
  simp only [← c100695, ← c100696, k2, ← c100697] at e_1645_1
  have hr := e_1645_1
  simp only [e_1645_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1645 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16256)) := by
  have c100698 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100699 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100700 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100701 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100702 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100703 := (publicBatchWrapper89_copies3146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1645_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_2
  simp only [← c100698, ← c100699, ← c100700] at e_1645_2
  have e_1645_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_3
  simp only [← c100701, ← c100702, k2, ← c100703] at e_1645_3
  have hr := e_1645_3
  simp only [e_1645_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1645 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16257)) := by
  have c100704 := (publicBatchWrapper89_copies3147 a h).1
  have c100705 := (publicBatchWrapper89_copies3147 a h).2.1
  have c100706 := (publicBatchWrapper89_copies3147 a h).2.2.1
  have c100707 := (publicBatchWrapper89_copies3147 a h).2.2.2.1
  have c100708 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.1
  have c100709 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1645_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_4
  simp only [← c100704, ← c100705, ← c100706] at e_1645_4
  have e_1645_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_5
  simp only [← c100707, ← c100708, k2, ← c100709] at e_1645_5
  have hr := e_1645_5
  simp only [e_1645_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1645 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16258)) := by
  have c100710 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.1
  have c100711 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.1
  have c100712 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.1
  have c100713 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.1
  have c100714 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100715 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1645_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_6
  simp only [← c100710, ← c100711, ← c100712] at e_1645_6
  have e_1645_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_7
  simp only [← c100713, ← c100714, k2, ← c100715] at e_1645_7
  have hr := e_1645_7
  simp only [e_1645_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1645 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16259)) := by
  have c100716 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100717 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100718 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100719 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100720 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100721 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1645_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_8
  simp only [← c100716, ← c100717, ← c100718] at e_1645_8
  have e_1645_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_9
  simp only [← c100719, ← c100720, k2, ← c100721] at e_1645_9
  have hr := e_1645_9
  simp only [e_1645_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1645 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16260)) := by
  have c100722 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100723 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100724 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100725 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100726 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100727 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1645_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_10
  simp only [← c100722, ← c100723, ← c100724] at e_1645_10
  have e_1645_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_11
  simp only [← c100725, ← c100726, k2, ← c100727] at e_1645_11
  have hr := e_1645_11
  simp only [e_1645_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1645 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16261)) := by
  have c100728 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100729 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100730 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100731 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100732 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100733 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1645_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_12
  simp only [← c100728, ← c100729, ← c100730] at e_1645_12
  have e_1645_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_13
  simp only [← c100731, ← c100732, k2, ← c100733] at e_1645_13
  have hr := e_1645_13
  simp only [e_1645_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1645 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16262)) := by
  have c100734 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100735 := (publicBatchWrapper89_copies3147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100736 := (publicBatchWrapper89_copies3148 a h).1
  have c100737 := (publicBatchWrapper89_copies3148 a h).2.1
  have c100738 := (publicBatchWrapper89_copies3148 a h).2.2.1
  have c100739 := (publicBatchWrapper89_copies3148 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1645_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_14
  simp only [← c100734, ← c100735, ← c100736] at e_1645_14
  have e_1645_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_15
  simp only [← c100737, ← c100738, k2, ← c100739] at e_1645_15
  have hr := e_1645_15
  simp only [e_1645_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1645 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16263)) := by
  have c100740 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.1
  have c100741 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.1
  have c100742 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.1
  have c100743 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.1
  have c100744 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.1
  have c100745 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1645_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_16
  simp only [← c100740, ← c100741, ← c100742] at e_1645_16
  have e_1645_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_17
  simp only [← c100743, ← c100744, k2, ← c100745] at e_1645_17
  have hr := e_1645_17
  simp only [e_1645_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1645 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16264)) := by
  have c100746 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100747 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100748 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100749 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100750 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100751 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1645_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_18
  simp only [← c100746, ← c100747, ← c100748] at e_1645_18
  have e_1645_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1645 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1645_19
  simp only [← c100749, ← c100750, k2, ← c100751] at e_1645_19
  have hr := e_1645_19
  simp only [e_1645_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1646 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16265)) := by
  have c100752 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100753 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100754 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100755 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100756 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100757 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1646_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_0
  simp only [← c100752, ← c100753, ← c100754] at e_1646_0
  have e_1646_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_1
  simp only [← c100755, ← c100756, k2, ← c100757] at e_1646_1
  have hr := e_1646_1
  simp only [e_1646_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1646 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16266)) := by
  have c100758 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100759 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100760 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100761 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100762 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100763 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1646_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_2
  simp only [← c100758, ← c100759, ← c100760] at e_1646_2
  have e_1646_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_3
  simp only [← c100761, ← c100762, k2, ← c100763] at e_1646_3
  have hr := e_1646_3
  simp only [e_1646_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1646 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16267)) := by
  have c100764 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100765 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100766 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100767 := (publicBatchWrapper89_copies3148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100768 := (publicBatchWrapper89_copies3149 a h).1
  have c100769 := (publicBatchWrapper89_copies3149 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1646_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_4
  simp only [← c100764, ← c100765, ← c100766] at e_1646_4
  have e_1646_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_5
  simp only [← c100767, ← c100768, k2, ← c100769] at e_1646_5
  have hr := e_1646_5
  simp only [e_1646_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1646 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16268)) := by
  have c100770 := (publicBatchWrapper89_copies3149 a h).2.2.1
  have c100771 := (publicBatchWrapper89_copies3149 a h).2.2.2.1
  have c100772 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.1
  have c100773 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.1
  have c100774 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.1
  have c100775 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1646_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_6
  simp only [← c100770, ← c100771, ← c100772] at e_1646_6
  have e_1646_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_7
  simp only [← c100773, ← c100774, k2, ← c100775] at e_1646_7
  have hr := e_1646_7
  simp only [e_1646_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1646 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16269)) := by
  have c100776 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.1
  have c100777 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.1
  have c100778 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100779 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100780 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100781 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1646_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_8
  simp only [← c100776, ← c100777, ← c100778] at e_1646_8
  have e_1646_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_9
  simp only [← c100779, ← c100780, k2, ← c100781] at e_1646_9
  have hr := e_1646_9
  simp only [e_1646_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1646 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16270)) := by
  have c100782 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100783 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100784 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100785 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100786 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100787 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1646_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_10
  simp only [← c100782, ← c100783, ← c100784] at e_1646_10
  have e_1646_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_11
  simp only [← c100785, ← c100786, k2, ← c100787] at e_1646_11
  have hr := e_1646_11
  simp only [e_1646_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1646 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16271)) := by
  have c100788 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100789 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100790 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100791 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100792 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100793 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1646_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_12
  simp only [← c100788, ← c100789, ← c100790] at e_1646_12
  have e_1646_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_13
  simp only [← c100791, ← c100792, k2, ← c100793] at e_1646_13
  have hr := e_1646_13
  simp only [e_1646_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1646 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16272)) := by
  have c100794 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100795 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100796 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100797 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100798 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100799 := (publicBatchWrapper89_copies3149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1646_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_14
  simp only [← c100794, ← c100795, ← c100796] at e_1646_14
  have e_1646_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_15
  simp only [← c100797, ← c100798, k2, ← c100799] at e_1646_15
  have hr := e_1646_15
  simp only [e_1646_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1646 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16273)) := by
  have c100800 := (publicBatchWrapper89_copies3150 a h).1
  have c100801 := (publicBatchWrapper89_copies3150 a h).2.1
  have c100802 := (publicBatchWrapper89_copies3150 a h).2.2.1
  have c100803 := (publicBatchWrapper89_copies3150 a h).2.2.2.1
  have c100804 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.1
  have c100805 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1646_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_16
  simp only [← c100800, ← c100801, ← c100802] at e_1646_16
  have e_1646_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_17
  simp only [← c100803, ← c100804, k2, ← c100805] at e_1646_17
  have hr := e_1646_17
  simp only [e_1646_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1646 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16274)) := by
  have c100806 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.1
  have c100807 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.1
  have c100808 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.1
  have c100809 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.1
  have c100810 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100811 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1646_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_18
  simp only [← c100806, ← c100807, ← c100808] at e_1646_18
  have e_1646_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1646 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1646_19
  simp only [← c100809, ← c100810, k2, ← c100811] at e_1646_19
  have hr := e_1646_19
  simp only [e_1646_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1647 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16275)) := by
  have c100812 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100813 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100814 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100815 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100816 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100817 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1647_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_0
  simp only [← c100812, ← c100813, ← c100814] at e_1647_0
  have e_1647_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_1
  simp only [← c100815, ← c100816, k2, ← c100817] at e_1647_1
  have hr := e_1647_1
  simp only [e_1647_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1647 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16276)) := by
  have c100818 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100819 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100820 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100821 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100822 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100823 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1647_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_2
  simp only [← c100818, ← c100819, ← c100820] at e_1647_2
  have e_1647_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_3
  simp only [← c100821, ← c100822, k2, ← c100823] at e_1647_3
  have hr := e_1647_3
  simp only [e_1647_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1647 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16277)) := by
  have c100824 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100825 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100826 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100827 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100828 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100829 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1647_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_4
  simp only [← c100824, ← c100825, ← c100826] at e_1647_4
  have e_1647_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_5
  simp only [← c100827, ← c100828, k2, ← c100829] at e_1647_5
  have hr := e_1647_5
  simp only [e_1647_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1647 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16278)) := by
  have c100830 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100831 := (publicBatchWrapper89_copies3150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100832 := (publicBatchWrapper89_copies3151 a h).1
  have c100833 := (publicBatchWrapper89_copies3151 a h).2.1
  have c100834 := (publicBatchWrapper89_copies3151 a h).2.2.1
  have c100835 := (publicBatchWrapper89_copies3151 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1647_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_6
  simp only [← c100830, ← c100831, ← c100832] at e_1647_6
  have e_1647_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_7
  simp only [← c100833, ← c100834, k2, ← c100835] at e_1647_7
  have hr := e_1647_7
  simp only [e_1647_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1647 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16279)) := by
  have c100836 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.1
  have c100837 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.1
  have c100838 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.1
  have c100839 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.1
  have c100840 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.1
  have c100841 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1647_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_8
  simp only [← c100836, ← c100837, ← c100838] at e_1647_8
  have e_1647_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_9
  simp only [← c100839, ← c100840, k2, ← c100841] at e_1647_9
  have hr := e_1647_9
  simp only [e_1647_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1647 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16280)) := by
  have c100842 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100843 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100844 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100845 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100846 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100847 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1647_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_10
  simp only [← c100842, ← c100843, ← c100844] at e_1647_10
  have e_1647_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_11
  simp only [← c100845, ← c100846, k2, ← c100847] at e_1647_11
  have hr := e_1647_11
  simp only [e_1647_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1647 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16281)) := by
  have c100848 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100849 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100850 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100851 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100852 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100853 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1647_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_12
  simp only [← c100848, ← c100849, ← c100850] at e_1647_12
  have e_1647_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_13
  simp only [← c100851, ← c100852, k2, ← c100853] at e_1647_13
  have hr := e_1647_13
  simp only [e_1647_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1647 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16282)) := by
  have c100854 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100855 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100856 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100857 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100858 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100859 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1647_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_14
  simp only [← c100854, ← c100855, ← c100856] at e_1647_14
  have e_1647_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_15
  simp only [← c100857, ← c100858, k2, ← c100859] at e_1647_15
  have hr := e_1647_15
  simp only [e_1647_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1647 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16283)) := by
  have c100860 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100861 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100862 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100863 := (publicBatchWrapper89_copies3151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100864 := (publicBatchWrapper89_copies3152 a h).1
  have c100865 := (publicBatchWrapper89_copies3152 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1647_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_16
  simp only [← c100860, ← c100861, ← c100862] at e_1647_16
  have e_1647_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_17
  simp only [← c100863, ← c100864, k2, ← c100865] at e_1647_17
  have hr := e_1647_17
  simp only [e_1647_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1647 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16284)) := by
  have c100866 := (publicBatchWrapper89_copies3152 a h).2.2.1
  have c100867 := (publicBatchWrapper89_copies3152 a h).2.2.2.1
  have c100868 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.1
  have c100869 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.1
  have c100870 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.1
  have c100871 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1647_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_18
  simp only [← c100866, ← c100867, ← c100868] at e_1647_18
  have e_1647_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1647 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1647_19
  simp only [← c100869, ← c100870, k2, ← c100871] at e_1647_19
  have hr := e_1647_19
  simp only [e_1647_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1648 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16285)) := by
  have c100872 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.1
  have c100873 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.1
  have c100874 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100875 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100876 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100877 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1648_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_0
  simp only [← c100872, ← c100873, ← c100874] at e_1648_0
  have e_1648_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_1
  simp only [← c100875, ← c100876, k2, ← c100877] at e_1648_1
  have hr := e_1648_1
  simp only [e_1648_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1648 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16286)) := by
  have c100878 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100879 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100880 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100881 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100882 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100883 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1648_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_2
  simp only [← c100878, ← c100879, ← c100880] at e_1648_2
  have e_1648_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_3
  simp only [← c100881, ← c100882, k2, ← c100883] at e_1648_3
  have hr := e_1648_3
  simp only [e_1648_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1648 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16287)) := by
  have c100884 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100885 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100886 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100887 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100888 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100889 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1648_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_4
  simp only [← c100884, ← c100885, ← c100886] at e_1648_4
  have e_1648_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_5
  simp only [← c100887, ← c100888, k2, ← c100889] at e_1648_5
  have hr := e_1648_5
  simp only [e_1648_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1648 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16288)) := by
  have c100890 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100891 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100892 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100893 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100894 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100895 := (publicBatchWrapper89_copies3152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1648_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_6
  simp only [← c100890, ← c100891, ← c100892] at e_1648_6
  have e_1648_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_7
  simp only [← c100893, ← c100894, k2, ← c100895] at e_1648_7
  have hr := e_1648_7
  simp only [e_1648_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1648 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16289)) := by
  have c100896 := (publicBatchWrapper89_copies3153 a h).1
  have c100897 := (publicBatchWrapper89_copies3153 a h).2.1
  have c100898 := (publicBatchWrapper89_copies3153 a h).2.2.1
  have c100899 := (publicBatchWrapper89_copies3153 a h).2.2.2.1
  have c100900 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.1
  have c100901 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1648_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_8
  simp only [← c100896, ← c100897, ← c100898] at e_1648_8
  have e_1648_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_9
  simp only [← c100899, ← c100900, k2, ← c100901] at e_1648_9
  have hr := e_1648_9
  simp only [e_1648_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1648 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16290)) := by
  have c100902 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.1
  have c100903 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.1
  have c100904 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.1
  have c100905 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.1
  have c100906 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100907 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1648_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_10
  simp only [← c100902, ← c100903, ← c100904] at e_1648_10
  have e_1648_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_11
  simp only [← c100905, ← c100906, k2, ← c100907] at e_1648_11
  have hr := e_1648_11
  simp only [e_1648_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1648 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16291)) := by
  have c100908 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100909 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100910 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100911 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100912 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100913 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1648_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_12
  simp only [← c100908, ← c100909, ← c100910] at e_1648_12
  have e_1648_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_13
  simp only [← c100911, ← c100912, k2, ← c100913] at e_1648_13
  have hr := e_1648_13
  simp only [e_1648_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1648 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16292)) := by
  have c100914 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100915 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100916 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100917 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100918 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100919 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1648_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_14
  simp only [← c100914, ← c100915, ← c100916] at e_1648_14
  have e_1648_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_15
  simp only [← c100917, ← c100918, k2, ← c100919] at e_1648_15
  have hr := e_1648_15
  simp only [e_1648_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1648 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16293)) := by
  have c100920 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100921 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100922 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100923 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100924 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100925 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1648_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_16
  simp only [← c100920, ← c100921, ← c100922] at e_1648_16
  have e_1648_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_17
  simp only [← c100923, ← c100924, k2, ← c100925] at e_1648_17
  have hr := e_1648_17
  simp only [e_1648_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1648 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16294)) := by
  have c100926 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100927 := (publicBatchWrapper89_copies3153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100928 := (publicBatchWrapper89_copies3154 a h).1
  have c100929 := (publicBatchWrapper89_copies3154 a h).2.1
  have c100930 := (publicBatchWrapper89_copies3154 a h).2.2.1
  have c100931 := (publicBatchWrapper89_copies3154 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1648_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_18
  simp only [← c100926, ← c100927, ← c100928] at e_1648_18
  have e_1648_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1648 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1648_19
  simp only [← c100929, ← c100930, k2, ← c100931] at e_1648_19
  have hr := e_1648_19
  simp only [e_1648_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1649 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16295)) := by
  have c100932 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.1
  have c100933 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.1
  have c100934 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.1
  have c100935 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.1
  have c100936 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.1
  have c100937 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1649_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_0
  simp only [← c100932, ← c100933, ← c100934] at e_1649_0
  have e_1649_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_1
  simp only [← c100935, ← c100936, k2, ← c100937] at e_1649_1
  have hr := e_1649_1
  simp only [e_1649_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1649 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16296)) := by
  have c100938 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100939 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100940 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100941 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100942 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100943 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1649_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_2
  simp only [← c100938, ← c100939, ← c100940] at e_1649_2
  have e_1649_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_3
  simp only [← c100941, ← c100942, k2, ← c100943] at e_1649_3
  have hr := e_1649_3
  simp only [e_1649_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1649 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16297)) := by
  have c100944 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100945 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100946 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100947 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100948 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100949 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1649_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_4
  simp only [← c100944, ← c100945, ← c100946] at e_1649_4
  have e_1649_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_5
  simp only [← c100947, ← c100948, k2, ← c100949] at e_1649_5
  have hr := e_1649_5
  simp only [e_1649_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1649 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16298)) := by
  have c100950 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100951 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100952 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100953 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100954 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100955 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1649_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_6
  simp only [← c100950, ← c100951, ← c100952] at e_1649_6
  have e_1649_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_7
  simp only [← c100953, ← c100954, k2, ← c100955] at e_1649_7
  have hr := e_1649_7
  simp only [e_1649_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1649 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16299)) := by
  have c100956 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100957 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100958 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100959 := (publicBatchWrapper89_copies3154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100960 := (publicBatchWrapper89_copies3155 a h).1
  have c100961 := (publicBatchWrapper89_copies3155 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1649_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_8
  simp only [← c100956, ← c100957, ← c100958] at e_1649_8
  have e_1649_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_9
  simp only [← c100959, ← c100960, k2, ← c100961] at e_1649_9
  have hr := e_1649_9
  simp only [e_1649_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1649 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16300)) := by
  have c100962 := (publicBatchWrapper89_copies3155 a h).2.2.1
  have c100963 := (publicBatchWrapper89_copies3155 a h).2.2.2.1
  have c100964 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.1
  have c100965 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.1
  have c100966 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.1
  have c100967 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1649_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_10
  simp only [← c100962, ← c100963, ← c100964] at e_1649_10
  have e_1649_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_11
  simp only [← c100965, ← c100966, k2, ← c100967] at e_1649_11
  have hr := e_1649_11
  simp only [e_1649_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1649 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16301)) := by
  have c100968 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.1
  have c100969 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.1
  have c100970 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100971 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100972 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100973 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1649_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_12
  simp only [← c100968, ← c100969, ← c100970] at e_1649_12
  have e_1649_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_13
  simp only [← c100971, ← c100972, k2, ← c100973] at e_1649_13
  have hr := e_1649_13
  simp only [e_1649_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1649 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16302)) := by
  have c100974 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100975 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100976 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100977 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100978 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100979 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1649_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_14
  simp only [← c100974, ← c100975, ← c100976] at e_1649_14
  have e_1649_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_15
  simp only [← c100977, ← c100978, k2, ← c100979] at e_1649_15
  have hr := e_1649_15
  simp only [e_1649_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1649 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16303)) := by
  have c100980 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100981 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100982 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100983 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100984 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100985 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1649_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_16
  simp only [← c100980, ← c100981, ← c100982] at e_1649_16
  have e_1649_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_17
  simp only [← c100983, ← c100984, k2, ← c100985] at e_1649_17
  have hr := e_1649_17
  simp only [e_1649_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1649 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16304)) := by
  have c100986 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100987 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100988 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100989 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100990 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100991 := (publicBatchWrapper89_copies3155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1649_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_18
  simp only [← c100986, ← c100987, ← c100988] at e_1649_18
  have e_1649_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1649 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1649_19
  simp only [← c100989, ← c100990, k2, ← c100991] at e_1649_19
  have hr := e_1649_19
  simp only [e_1649_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1650 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16305)) := by
  have c100992 := (publicBatchWrapper89_copies3156 a h).1
  have c100993 := (publicBatchWrapper89_copies3156 a h).2.1
  have c100994 := (publicBatchWrapper89_copies3156 a h).2.2.1
  have c100995 := (publicBatchWrapper89_copies3156 a h).2.2.2.1
  have c100996 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.1
  have c100997 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1650_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_0
  simp only [← c100992, ← c100993, ← c100994] at e_1650_0
  have e_1650_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_1
  simp only [← c100995, ← c100996, k2, ← c100997] at e_1650_1
  have hr := e_1650_1
  simp only [e_1650_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1650 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16306)) := by
  have c100998 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.1
  have c100999 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.1
  have c101000 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.1
  have c101001 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.1
  have c101002 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101003 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1650_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_2
  simp only [← c100998, ← c100999, ← c101000] at e_1650_2
  have e_1650_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_3
  simp only [← c101001, ← c101002, k2, ← c101003] at e_1650_3
  have hr := e_1650_3
  simp only [e_1650_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1650 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16307)) := by
  have c101004 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101005 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101006 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101007 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101008 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101009 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1650_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_4
  simp only [← c101004, ← c101005, ← c101006] at e_1650_4
  have e_1650_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_5
  simp only [← c101007, ← c101008, k2, ← c101009] at e_1650_5
  have hr := e_1650_5
  simp only [e_1650_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1650 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16308)) := by
  have c101010 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101011 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101012 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101013 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101014 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101015 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1650_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_6
  simp only [← c101010, ← c101011, ← c101012] at e_1650_6
  have e_1650_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_7
  simp only [← c101013, ← c101014, k2, ← c101015] at e_1650_7
  have hr := e_1650_7
  simp only [e_1650_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1650 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16309)) := by
  have c101016 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101017 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101018 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101019 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101020 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101021 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1650_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_8
  simp only [← c101016, ← c101017, ← c101018] at e_1650_8
  have e_1650_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_9
  simp only [← c101019, ← c101020, k2, ← c101021] at e_1650_9
  have hr := e_1650_9
  simp only [e_1650_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1650 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16310)) := by
  have c101022 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101023 := (publicBatchWrapper89_copies3156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101024 := (publicBatchWrapper89_copies3157 a h).1
  have c101025 := (publicBatchWrapper89_copies3157 a h).2.1
  have c101026 := (publicBatchWrapper89_copies3157 a h).2.2.1
  have c101027 := (publicBatchWrapper89_copies3157 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1650_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_10
  simp only [← c101022, ← c101023, ← c101024] at e_1650_10
  have e_1650_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_11
  simp only [← c101025, ← c101026, k2, ← c101027] at e_1650_11
  have hr := e_1650_11
  simp only [e_1650_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1650 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16311)) := by
  have c101028 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.1
  have c101029 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.1
  have c101030 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.1
  have c101031 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.1
  have c101032 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.1
  have c101033 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1650_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_12
  simp only [← c101028, ← c101029, ← c101030] at e_1650_12
  have e_1650_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_13
  simp only [← c101031, ← c101032, k2, ← c101033] at e_1650_13
  have hr := e_1650_13
  simp only [e_1650_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1650 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16312)) := by
  have c101034 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101035 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101036 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101037 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101038 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101039 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1650_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_14
  simp only [← c101034, ← c101035, ← c101036] at e_1650_14
  have e_1650_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_15
  simp only [← c101037, ← c101038, k2, ← c101039] at e_1650_15
  have hr := e_1650_15
  simp only [e_1650_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1650 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16313)) := by
  have c101040 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101041 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101042 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101043 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101044 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101045 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1650_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_16
  simp only [← c101040, ← c101041, ← c101042] at e_1650_16
  have e_1650_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_17
  simp only [← c101043, ← c101044, k2, ← c101045] at e_1650_17
  have hr := e_1650_17
  simp only [e_1650_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1650 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16314)) := by
  have c101046 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101047 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101048 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101049 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101050 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101051 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1650_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_18
  simp only [← c101046, ← c101047, ← c101048] at e_1650_18
  have e_1650_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1650 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1650_19
  simp only [← c101049, ← c101050, k2, ← c101051] at e_1650_19
  have hr := e_1650_19
  simp only [e_1650_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1651 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16315)) := by
  have c101052 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101053 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101054 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101055 := (publicBatchWrapper89_copies3157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101056 := (publicBatchWrapper89_copies3158 a h).1
  have c101057 := (publicBatchWrapper89_copies3158 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1651_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_0
  simp only [← c101052, ← c101053, ← c101054] at e_1651_0
  have e_1651_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_1
  simp only [← c101055, ← c101056, k2, ← c101057] at e_1651_1
  have hr := e_1651_1
  simp only [e_1651_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1651 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16316)) := by
  have c101058 := (publicBatchWrapper89_copies3158 a h).2.2.1
  have c101059 := (publicBatchWrapper89_copies3158 a h).2.2.2.1
  have c101060 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.1
  have c101061 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.1
  have c101062 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.1
  have c101063 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1651_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_2
  simp only [← c101058, ← c101059, ← c101060] at e_1651_2
  have e_1651_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_3
  simp only [← c101061, ← c101062, k2, ← c101063] at e_1651_3
  have hr := e_1651_3
  simp only [e_1651_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1651 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16317)) := by
  have c101064 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.1
  have c101065 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.1
  have c101066 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101067 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101068 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101069 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1651_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_4
  simp only [← c101064, ← c101065, ← c101066] at e_1651_4
  have e_1651_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_5
  simp only [← c101067, ← c101068, k2, ← c101069] at e_1651_5
  have hr := e_1651_5
  simp only [e_1651_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1651 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16318)) := by
  have c101070 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101071 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101072 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101073 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101074 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101075 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1651_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_6
  simp only [← c101070, ← c101071, ← c101072] at e_1651_6
  have e_1651_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_7
  simp only [← c101073, ← c101074, k2, ← c101075] at e_1651_7
  have hr := e_1651_7
  simp only [e_1651_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1651 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16319)) := by
  have c101076 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101077 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101078 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101079 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101080 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101081 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1651_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_8
  simp only [← c101076, ← c101077, ← c101078] at e_1651_8
  have e_1651_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_9
  simp only [← c101079, ← c101080, k2, ← c101081] at e_1651_9
  have hr := e_1651_9
  simp only [e_1651_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1651 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16320)) := by
  have c101082 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101083 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101084 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101085 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101086 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101087 := (publicBatchWrapper89_copies3158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1651_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_10
  simp only [← c101082, ← c101083, ← c101084] at e_1651_10
  have e_1651_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_11
  simp only [← c101085, ← c101086, k2, ← c101087] at e_1651_11
  have hr := e_1651_11
  simp only [e_1651_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1651 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16321)) := by
  have c101088 := (publicBatchWrapper89_copies3159 a h).1
  have c101089 := (publicBatchWrapper89_copies3159 a h).2.1
  have c101090 := (publicBatchWrapper89_copies3159 a h).2.2.1
  have c101091 := (publicBatchWrapper89_copies3159 a h).2.2.2.1
  have c101092 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.1
  have c101093 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1651_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_12
  simp only [← c101088, ← c101089, ← c101090] at e_1651_12
  have e_1651_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_13
  simp only [← c101091, ← c101092, k2, ← c101093] at e_1651_13
  have hr := e_1651_13
  simp only [e_1651_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1651 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16322)) := by
  have c101094 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.1
  have c101095 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.1
  have c101096 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.1
  have c101097 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.1
  have c101098 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101099 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1651_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_14
  simp only [← c101094, ← c101095, ← c101096] at e_1651_14
  have e_1651_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_15
  simp only [← c101097, ← c101098, k2, ← c101099] at e_1651_15
  have hr := e_1651_15
  simp only [e_1651_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1651 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16323)) := by
  have c101100 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101101 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101102 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101103 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101104 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101105 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1651_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_16
  simp only [← c101100, ← c101101, ← c101102] at e_1651_16
  have e_1651_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_17
  simp only [← c101103, ← c101104, k2, ← c101105] at e_1651_17
  have hr := e_1651_17
  simp only [e_1651_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1651 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16324)) := by
  have c101106 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101107 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101108 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101109 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101110 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101111 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1651_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_18
  simp only [← c101106, ← c101107, ← c101108] at e_1651_18
  have e_1651_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1651 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1651_19
  simp only [← c101109, ← c101110, k2, ← c101111] at e_1651_19
  have hr := e_1651_19
  simp only [e_1651_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1652 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16325)) := by
  have c101112 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101113 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101114 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101115 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101116 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101117 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1652_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_0
  simp only [← c101112, ← c101113, ← c101114] at e_1652_0
  have e_1652_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_1
  simp only [← c101115, ← c101116, k2, ← c101117] at e_1652_1
  have hr := e_1652_1
  simp only [e_1652_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1652 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16326)) := by
  have c101118 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101119 := (publicBatchWrapper89_copies3159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101120 := (publicBatchWrapper89_copies3160 a h).1
  have c101121 := (publicBatchWrapper89_copies3160 a h).2.1
  have c101122 := (publicBatchWrapper89_copies3160 a h).2.2.1
  have c101123 := (publicBatchWrapper89_copies3160 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1652_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_2
  simp only [← c101118, ← c101119, ← c101120] at e_1652_2
  have e_1652_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_3
  simp only [← c101121, ← c101122, k2, ← c101123] at e_1652_3
  have hr := e_1652_3
  simp only [e_1652_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1652 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16327)) := by
  have c101124 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.1
  have c101125 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.1
  have c101126 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.1
  have c101127 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.1
  have c101128 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.1
  have c101129 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1652_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_4
  simp only [← c101124, ← c101125, ← c101126] at e_1652_4
  have e_1652_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_5
  simp only [← c101127, ← c101128, k2, ← c101129] at e_1652_5
  have hr := e_1652_5
  simp only [e_1652_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1652 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16328)) := by
  have c101130 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101131 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101132 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101133 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101134 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101135 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1652_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_6
  simp only [← c101130, ← c101131, ← c101132] at e_1652_6
  have e_1652_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_7
  simp only [← c101133, ← c101134, k2, ← c101135] at e_1652_7
  have hr := e_1652_7
  simp only [e_1652_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1652 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16329)) := by
  have c101136 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101137 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101138 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101139 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101140 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101141 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1652_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_8
  simp only [← c101136, ← c101137, ← c101138] at e_1652_8
  have e_1652_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_9
  simp only [← c101139, ← c101140, k2, ← c101141] at e_1652_9
  have hr := e_1652_9
  simp only [e_1652_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1652 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16330)) := by
  have c101142 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101143 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101144 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101145 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101146 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101147 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1652_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_10
  simp only [← c101142, ← c101143, ← c101144] at e_1652_10
  have e_1652_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_11
  simp only [← c101145, ← c101146, k2, ← c101147] at e_1652_11
  have hr := e_1652_11
  simp only [e_1652_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1652 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16331)) := by
  have c101148 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101149 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101150 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101151 := (publicBatchWrapper89_copies3160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101152 := (publicBatchWrapper89_copies3161 a h).1
  have c101153 := (publicBatchWrapper89_copies3161 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1652_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_12
  simp only [← c101148, ← c101149, ← c101150] at e_1652_12
  have e_1652_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_13
  simp only [← c101151, ← c101152, k2, ← c101153] at e_1652_13
  have hr := e_1652_13
  simp only [e_1652_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1652 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16332)) := by
  have c101154 := (publicBatchWrapper89_copies3161 a h).2.2.1
  have c101155 := (publicBatchWrapper89_copies3161 a h).2.2.2.1
  have c101156 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.1
  have c101157 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.1
  have c101158 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.1
  have c101159 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1652_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_14
  simp only [← c101154, ← c101155, ← c101156] at e_1652_14
  have e_1652_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_15
  simp only [← c101157, ← c101158, k2, ← c101159] at e_1652_15
  have hr := e_1652_15
  simp only [e_1652_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1652 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16333)) := by
  have c101160 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.1
  have c101161 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.1
  have c101162 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101163 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101164 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101165 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1652_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_16
  simp only [← c101160, ← c101161, ← c101162] at e_1652_16
  have e_1652_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_17
  simp only [← c101163, ← c101164, k2, ← c101165] at e_1652_17
  have hr := e_1652_17
  simp only [e_1652_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1652 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16334)) := by
  have c101166 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101167 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101168 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101169 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101170 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101171 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1652_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_18
  simp only [← c101166, ← c101167, ← c101168] at e_1652_18
  have e_1652_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1652 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1652_19
  simp only [← c101169, ← c101170, k2, ← c101171] at e_1652_19
  have hr := e_1652_19
  simp only [e_1652_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1653 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16335)) := by
  have c101172 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101173 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101174 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101175 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101176 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101177 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1653_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_0
  simp only [← c101172, ← c101173, ← c101174] at e_1653_0
  have e_1653_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_1
  simp only [← c101175, ← c101176, k2, ← c101177] at e_1653_1
  have hr := e_1653_1
  simp only [e_1653_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1653 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16336)) := by
  have c101178 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101179 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101180 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101181 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101182 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101183 := (publicBatchWrapper89_copies3161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1653_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_2
  simp only [← c101178, ← c101179, ← c101180] at e_1653_2
  have e_1653_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_3
  simp only [← c101181, ← c101182, k2, ← c101183] at e_1653_3
  have hr := e_1653_3
  simp only [e_1653_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1653 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16337)) := by
  have c101184 := (publicBatchWrapper89_copies3162 a h).1
  have c101185 := (publicBatchWrapper89_copies3162 a h).2.1
  have c101186 := (publicBatchWrapper89_copies3162 a h).2.2.1
  have c101187 := (publicBatchWrapper89_copies3162 a h).2.2.2.1
  have c101188 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.1
  have c101189 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1653_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_4
  simp only [← c101184, ← c101185, ← c101186] at e_1653_4
  have e_1653_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_5
  simp only [← c101187, ← c101188, k2, ← c101189] at e_1653_5
  have hr := e_1653_5
  simp only [e_1653_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1653 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16338)) := by
  have c101190 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.1
  have c101191 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.1
  have c101192 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.1
  have c101193 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.1
  have c101194 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101195 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1653_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_6
  simp only [← c101190, ← c101191, ← c101192] at e_1653_6
  have e_1653_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_7
  simp only [← c101193, ← c101194, k2, ← c101195] at e_1653_7
  have hr := e_1653_7
  simp only [e_1653_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1653 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16339)) := by
  have c101196 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101197 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101198 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101199 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101200 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101201 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1653_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_8
  simp only [← c101196, ← c101197, ← c101198] at e_1653_8
  have e_1653_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_9
  simp only [← c101199, ← c101200, k2, ← c101201] at e_1653_9
  have hr := e_1653_9
  simp only [e_1653_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1653 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16340)) := by
  have c101202 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101203 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101204 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101205 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101206 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101207 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1653_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_10
  simp only [← c101202, ← c101203, ← c101204] at e_1653_10
  have e_1653_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_11
  simp only [← c101205, ← c101206, k2, ← c101207] at e_1653_11
  have hr := e_1653_11
  simp only [e_1653_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1653 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16341)) := by
  have c101208 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101209 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101210 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101211 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101212 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101213 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1653_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_12
  simp only [← c101208, ← c101209, ← c101210] at e_1653_12
  have e_1653_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_13
  simp only [← c101211, ← c101212, k2, ← c101213] at e_1653_13
  have hr := e_1653_13
  simp only [e_1653_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1653 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16342)) := by
  have c101214 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101215 := (publicBatchWrapper89_copies3162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101216 := (publicBatchWrapper89_copies3163 a h).1
  have c101217 := (publicBatchWrapper89_copies3163 a h).2.1
  have c101218 := (publicBatchWrapper89_copies3163 a h).2.2.1
  have c101219 := (publicBatchWrapper89_copies3163 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1653_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_14
  simp only [← c101214, ← c101215, ← c101216] at e_1653_14
  have e_1653_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_15
  simp only [← c101217, ← c101218, k2, ← c101219] at e_1653_15
  have hr := e_1653_15
  simp only [e_1653_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1653 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16343)) := by
  have c101220 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.1
  have c101221 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.1
  have c101222 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.1
  have c101223 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.1
  have c101224 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.1
  have c101225 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1653_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_16
  simp only [← c101220, ← c101221, ← c101222] at e_1653_16
  have e_1653_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_17
  simp only [← c101223, ← c101224, k2, ← c101225] at e_1653_17
  have hr := e_1653_17
  simp only [e_1653_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1653 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16344)) := by
  have c101226 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101227 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101228 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101229 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101230 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101231 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1653_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_18
  simp only [← c101226, ← c101227, ← c101228] at e_1653_18
  have e_1653_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1653 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1653_19
  simp only [← c101229, ← c101230, k2, ← c101231] at e_1653_19
  have hr := e_1653_19
  simp only [e_1653_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1654 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16345)) := by
  have c101232 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101233 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101234 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101235 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101236 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101237 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1654_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_0
  simp only [← c101232, ← c101233, ← c101234] at e_1654_0
  have e_1654_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_1
  simp only [← c101235, ← c101236, k2, ← c101237] at e_1654_1
  have hr := e_1654_1
  simp only [e_1654_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1654 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16346)) := by
  have c101238 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101239 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101240 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101241 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101242 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101243 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1654_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_2
  simp only [← c101238, ← c101239, ← c101240] at e_1654_2
  have e_1654_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_3
  simp only [← c101241, ← c101242, k2, ← c101243] at e_1654_3
  have hr := e_1654_3
  simp only [e_1654_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1654 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16347)) := by
  have c101244 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101245 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101246 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101247 := (publicBatchWrapper89_copies3163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101248 := (publicBatchWrapper89_copies3164 a h).1
  have c101249 := (publicBatchWrapper89_copies3164 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1654_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_4
  simp only [← c101244, ← c101245, ← c101246] at e_1654_4
  have e_1654_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_5
  simp only [← c101247, ← c101248, k2, ← c101249] at e_1654_5
  have hr := e_1654_5
  simp only [e_1654_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1654 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16348)) := by
  have c101250 := (publicBatchWrapper89_copies3164 a h).2.2.1
  have c101251 := (publicBatchWrapper89_copies3164 a h).2.2.2.1
  have c101252 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.1
  have c101253 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.1
  have c101254 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.1
  have c101255 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1654_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_6
  simp only [← c101250, ← c101251, ← c101252] at e_1654_6
  have e_1654_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_7
  simp only [← c101253, ← c101254, k2, ← c101255] at e_1654_7
  have hr := e_1654_7
  simp only [e_1654_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1654 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16349)) := by
  have c101256 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.1
  have c101257 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.1
  have c101258 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101259 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101260 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101261 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1654_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_8
  simp only [← c101256, ← c101257, ← c101258] at e_1654_8
  have e_1654_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_9
  simp only [← c101259, ← c101260, k2, ← c101261] at e_1654_9
  have hr := e_1654_9
  simp only [e_1654_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1654 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16350)) := by
  have c101262 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101263 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101264 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101265 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101266 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101267 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1654_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_10
  simp only [← c101262, ← c101263, ← c101264] at e_1654_10
  have e_1654_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_11
  simp only [← c101265, ← c101266, k2, ← c101267] at e_1654_11
  have hr := e_1654_11
  simp only [e_1654_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1654 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16351)) := by
  have c101268 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101269 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101270 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101271 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101272 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101273 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1654_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_12
  simp only [← c101268, ← c101269, ← c101270] at e_1654_12
  have e_1654_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_13
  simp only [← c101271, ← c101272, k2, ← c101273] at e_1654_13
  have hr := e_1654_13
  simp only [e_1654_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1654 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16352)) := by
  have c101274 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101275 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101276 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101277 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101278 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101279 := (publicBatchWrapper89_copies3164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1654_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_14
  simp only [← c101274, ← c101275, ← c101276] at e_1654_14
  have e_1654_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_15
  simp only [← c101277, ← c101278, k2, ← c101279] at e_1654_15
  have hr := e_1654_15
  simp only [e_1654_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1654 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16353)) := by
  have c101280 := (publicBatchWrapper89_copies3165 a h).1
  have c101281 := (publicBatchWrapper89_copies3165 a h).2.1
  have c101282 := (publicBatchWrapper89_copies3165 a h).2.2.1
  have c101283 := (publicBatchWrapper89_copies3165 a h).2.2.2.1
  have c101284 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.1
  have c101285 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1654_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_16
  simp only [← c101280, ← c101281, ← c101282] at e_1654_16
  have e_1654_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_17
  simp only [← c101283, ← c101284, k2, ← c101285] at e_1654_17
  have hr := e_1654_17
  simp only [e_1654_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1654 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16354)) := by
  have c101286 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.1
  have c101287 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.1
  have c101288 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.1
  have c101289 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.1
  have c101290 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101291 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1654_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_18
  simp only [← c101286, ← c101287, ← c101288] at e_1654_18
  have e_1654_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1654 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1654_19
  simp only [← c101289, ← c101290, k2, ← c101291] at e_1654_19
  have hr := e_1654_19
  simp only [e_1654_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1655 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16355)) := by
  have c101292 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101293 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101294 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101295 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101296 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101297 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1655_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_0
  simp only [← c101292, ← c101293, ← c101294] at e_1655_0
  have e_1655_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_1
  simp only [← c101295, ← c101296, k2, ← c101297] at e_1655_1
  have hr := e_1655_1
  simp only [e_1655_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1655 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16356)) := by
  have c101298 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101299 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101300 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101301 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101302 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101303 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1655_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_2
  simp only [← c101298, ← c101299, ← c101300] at e_1655_2
  have e_1655_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_3
  simp only [← c101301, ← c101302, k2, ← c101303] at e_1655_3
  have hr := e_1655_3
  simp only [e_1655_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1655 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16357)) := by
  have c101304 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101305 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101306 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101307 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101308 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101309 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1655_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_4
  simp only [← c101304, ← c101305, ← c101306] at e_1655_4
  have e_1655_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_5
  simp only [← c101307, ← c101308, k2, ← c101309] at e_1655_5
  have hr := e_1655_5
  simp only [e_1655_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1655 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16358)) := by
  have c101310 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101311 := (publicBatchWrapper89_copies3165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101312 := (publicBatchWrapper89_copies3166 a h).1
  have c101313 := (publicBatchWrapper89_copies3166 a h).2.1
  have c101314 := (publicBatchWrapper89_copies3166 a h).2.2.1
  have c101315 := (publicBatchWrapper89_copies3166 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1655_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_6
  simp only [← c101310, ← c101311, ← c101312] at e_1655_6
  have e_1655_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_7
  simp only [← c101313, ← c101314, k2, ← c101315] at e_1655_7
  have hr := e_1655_7
  simp only [e_1655_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1655 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16359)) := by
  have c101316 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.1
  have c101317 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.1
  have c101318 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.1
  have c101319 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.1
  have c101320 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.1
  have c101321 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1655_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_8
  simp only [← c101316, ← c101317, ← c101318] at e_1655_8
  have e_1655_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_9
  simp only [← c101319, ← c101320, k2, ← c101321] at e_1655_9
  have hr := e_1655_9
  simp only [e_1655_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1655 47) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16360)) := by
  have c101322 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101323 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101324 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101325 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101326 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101327 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1655_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_10
  simp only [← c101322, ← c101323, ← c101324] at e_1655_10
  have e_1655_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_11
  simp only [← c101325, ← c101326, k2, ← c101327] at e_1655_11
  have hr := e_1655_11
  simp only [e_1655_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1655 55) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16361)) := by
  have c101328 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101329 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101330 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101331 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101332 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101333 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1655_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_12
  simp only [← c101328, ← c101329, ← c101330] at e_1655_12
  have e_1655_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_13
  simp only [← c101331, ← c101332, k2, ← c101333] at e_1655_13
  have hr := e_1655_13
  simp only [e_1655_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1655 63) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16362)) := by
  have c101334 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101335 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101336 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101337 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101338 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101339 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1655_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_14
  simp only [← c101334, ← c101335, ← c101336] at e_1655_14
  have e_1655_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_15
  simp only [← c101337, ← c101338, k2, ← c101339] at e_1655_15
  have hr := e_1655_15
  simp only [e_1655_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1655 71) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16363)) := by
  have c101340 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101341 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101342 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101343 := (publicBatchWrapper89_copies3166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101344 := (publicBatchWrapper89_copies3167 a h).1
  have c101345 := (publicBatchWrapper89_copies3167 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1655_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_16
  simp only [← c101340, ← c101341, ← c101342] at e_1655_16
  have e_1655_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_17
  simp only [← c101343, ← c101344, k2, ← c101345] at e_1655_17
  have hr := e_1655_17
  simp only [e_1655_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1655 79) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16364)) := by
  have c101346 := (publicBatchWrapper89_copies3167 a h).2.2.1
  have c101347 := (publicBatchWrapper89_copies3167 a h).2.2.2.1
  have c101348 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.1
  have c101349 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.1
  have c101350 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.1
  have c101351 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1655_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_18
  simp only [← c101346, ← c101347, ← c101348] at e_1655_18
  have e_1655_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1655 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1655_19
  simp only [← c101349, ← c101350, k2, ← c101351] at e_1655_19
  have hr := e_1655_19
  simp only [e_1655_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1656 7) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16365)) := by
  have c101352 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.1
  have c101353 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.1
  have c101354 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101355 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101356 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101357 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1656_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_0
  simp only [← c101352, ← c101353, ← c101354] at e_1656_0
  have e_1656_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_1
  simp only [← c101355, ← c101356, k2, ← c101357] at e_1656_1
  have hr := e_1656_1
  simp only [e_1656_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1656 15) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16366)) := by
  have c101358 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101359 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101360 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101361 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101362 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101363 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1656_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_2
  simp only [← c101358, ← c101359, ← c101360] at e_1656_2
  have e_1656_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_3
  simp only [← c101361, ← c101362, k2, ← c101363] at e_1656_3
  have hr := e_1656_3
  simp only [e_1656_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1656 23) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16367)) := by
  have c101364 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101365 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101366 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101367 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101368 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101369 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1656_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_4
  simp only [← c101364, ← c101365, ← c101366] at e_1656_4
  have e_1656_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_5
  simp only [← c101367, ← c101368, k2, ← c101369] at e_1656_5
  have hr := e_1656_5
  simp only [e_1656_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1656 31) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16368)) := by
  have c101370 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101371 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101372 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101373 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101374 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101375 := (publicBatchWrapper89_copies3167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1656_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_6
  simp only [← c101370, ← c101371, ← c101372] at e_1656_6
  have e_1656_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_7
  simp only [← c101373, ← c101374, k2, ← c101375] at e_1656_7
  have hr := e_1656_7
  simp only [e_1656_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1656 39) = bselect (a (.wire 79 23)) (a (.virt 200)) (a (.virt 16369)) := by
  have c101376 := (publicBatchWrapper89_copies3168 a h).1
  have c101377 := (publicBatchWrapper89_copies3168 a h).2.1
  have c101378 := (publicBatchWrapper89_copies3168 a h).2.2.1
  have c101379 := (publicBatchWrapper89_copies3168 a h).2.2.2.1
  have c101380 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.1
  have c101381 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1656_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_8
  simp only [← c101376, ← c101377, ← c101378] at e_1656_8
  have e_1656_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_9
  simp only [← c101379, ← c101380, k2, ← c101381] at e_1656_9
  have hr := e_1656_9
  simp only [e_1656_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1656 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16410)) := by
  have c101382 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.1
  have c101383 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.1
  have c101384 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.1
  have c101385 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.1
  have c101386 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101387 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1656_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_10
  simp only [← c101382, ← c101383, ← c101384] at e_1656_10
  have e_1656_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_11
  simp only [← c101385, ← c101386, k2, ← c101387] at e_1656_11
  have hr := e_1656_11
  simp only [e_1656_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1656 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16411)) := by
  have c101388 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101389 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101390 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101391 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101392 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101393 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1656_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_12
  simp only [← c101388, ← c101389, ← c101390] at e_1656_12
  have e_1656_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_13
  simp only [← c101391, ← c101392, k2, ← c101393] at e_1656_13
  have hr := e_1656_13
  simp only [e_1656_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1656 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16412)) := by
  have c101394 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101395 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101396 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101397 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101398 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101399 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1656_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_14
  simp only [← c101394, ← c101395, ← c101396] at e_1656_14
  have e_1656_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_15
  simp only [← c101397, ← c101398, k2, ← c101399] at e_1656_15
  have hr := e_1656_15
  simp only [e_1656_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1656 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16413)) := by
  have c101400 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101401 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101402 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101403 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101404 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101405 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1656_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_16
  simp only [← c101400, ← c101401, ← c101402] at e_1656_16
  have e_1656_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_17
  simp only [← c101403, ← c101404, k2, ← c101405] at e_1656_17
  have hr := e_1656_17
  simp only [e_1656_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1656 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16414)) := by
  have c101406 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101407 := (publicBatchWrapper89_copies3168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101408 := (publicBatchWrapper89_copies3169 a h).1
  have c101409 := (publicBatchWrapper89_copies3169 a h).2.1
  have c101410 := (publicBatchWrapper89_copies3169 a h).2.2.1
  have c101411 := (publicBatchWrapper89_copies3169 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1656_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_18
  simp only [← c101406, ← c101407, ← c101408] at e_1656_18
  have e_1656_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1656 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1656_19
  simp only [← c101409, ← c101410, k2, ← c101411] at e_1656_19
  have hr := e_1656_19
  simp only [e_1656_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1657 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16415)) := by
  have c101412 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.1
  have c101413 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.1
  have c101414 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.1
  have c101415 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.1
  have c101416 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.1
  have c101417 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1657_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_0
  simp only [← c101412, ← c101413, ← c101414] at e_1657_0
  have e_1657_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_1
  simp only [← c101415, ← c101416, k2, ← c101417] at e_1657_1
  have hr := e_1657_1
  simp only [e_1657_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1657 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16416)) := by
  have c101418 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101419 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101420 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101421 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101422 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101423 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1657_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_2
  simp only [← c101418, ← c101419, ← c101420] at e_1657_2
  have e_1657_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_3
  simp only [← c101421, ← c101422, k2, ← c101423] at e_1657_3
  have hr := e_1657_3
  simp only [e_1657_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1657 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16417)) := by
  have c101424 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101425 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101426 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101427 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101428 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101429 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1657_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_4
  simp only [← c101424, ← c101425, ← c101426] at e_1657_4
  have e_1657_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_5
  simp only [← c101427, ← c101428, k2, ← c101429] at e_1657_5
  have hr := e_1657_5
  simp only [e_1657_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1657 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16418)) := by
  have c101430 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101431 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101432 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101433 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101434 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101435 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1657_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_6
  simp only [← c101430, ← c101431, ← c101432] at e_1657_6
  have e_1657_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_7
  simp only [← c101433, ← c101434, k2, ← c101435] at e_1657_7
  have hr := e_1657_7
  simp only [e_1657_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1657 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16419)) := by
  have c101436 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101437 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101438 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101439 := (publicBatchWrapper89_copies3169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101440 := (publicBatchWrapper89_copies3170 a h).1
  have c101441 := (publicBatchWrapper89_copies3170 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1657_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_8
  simp only [← c101436, ← c101437, ← c101438] at e_1657_8
  have e_1657_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_9
  simp only [← c101439, ← c101440, k2, ← c101441] at e_1657_9
  have hr := e_1657_9
  simp only [e_1657_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1657 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16420)) := by
  have c101442 := (publicBatchWrapper89_copies3170 a h).2.2.1
  have c101443 := (publicBatchWrapper89_copies3170 a h).2.2.2.1
  have c101444 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.1
  have c101445 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.1
  have c101446 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.1
  have c101447 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1657_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_10
  simp only [← c101442, ← c101443, ← c101444] at e_1657_10
  have e_1657_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_11
  simp only [← c101445, ← c101446, k2, ← c101447] at e_1657_11
  have hr := e_1657_11
  simp only [e_1657_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1657 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16421)) := by
  have c101448 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.1
  have c101449 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.1
  have c101450 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101451 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101452 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101453 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1657_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_12
  simp only [← c101448, ← c101449, ← c101450] at e_1657_12
  have e_1657_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_13
  simp only [← c101451, ← c101452, k2, ← c101453] at e_1657_13
  have hr := e_1657_13
  simp only [e_1657_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1657 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16422)) := by
  have c101454 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101455 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101456 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101457 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101458 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101459 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1657_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_14
  simp only [← c101454, ← c101455, ← c101456] at e_1657_14
  have e_1657_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_15
  simp only [← c101457, ← c101458, k2, ← c101459] at e_1657_15
  have hr := e_1657_15
  simp only [e_1657_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1657 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16423)) := by
  have c101460 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101461 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101462 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101463 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101464 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101465 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1657_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_16
  simp only [← c101460, ← c101461, ← c101462] at e_1657_16
  have e_1657_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_17
  simp only [← c101463, ← c101464, k2, ← c101465] at e_1657_17
  have hr := e_1657_17
  simp only [e_1657_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1657 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16424)) := by
  have c101466 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101467 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101468 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101469 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101470 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101471 := (publicBatchWrapper89_copies3170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1657_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_18
  simp only [← c101466, ← c101467, ← c101468] at e_1657_18
  have e_1657_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1657 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1657_19
  simp only [← c101469, ← c101470, k2, ← c101471] at e_1657_19
  have hr := e_1657_19
  simp only [e_1657_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1658 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16425)) := by
  have c101472 := (publicBatchWrapper89_copies3171 a h).1
  have c101473 := (publicBatchWrapper89_copies3171 a h).2.1
  have c101474 := (publicBatchWrapper89_copies3171 a h).2.2.1
  have c101475 := (publicBatchWrapper89_copies3171 a h).2.2.2.1
  have c101476 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.1
  have c101477 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1658_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_0
  simp only [← c101472, ← c101473, ← c101474] at e_1658_0
  have e_1658_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_1
  simp only [← c101475, ← c101476, k2, ← c101477] at e_1658_1
  have hr := e_1658_1
  simp only [e_1658_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1658 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16426)) := by
  have c101478 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.1
  have c101479 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.1
  have c101480 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.1
  have c101481 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.1
  have c101482 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101483 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1658_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_2
  simp only [← c101478, ← c101479, ← c101480] at e_1658_2
  have e_1658_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_3
  simp only [← c101481, ← c101482, k2, ← c101483] at e_1658_3
  have hr := e_1658_3
  simp only [e_1658_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1658 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16427)) := by
  have c101484 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101485 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101486 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101487 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101488 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101489 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1658_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_4
  simp only [← c101484, ← c101485, ← c101486] at e_1658_4
  have e_1658_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_5
  simp only [← c101487, ← c101488, k2, ← c101489] at e_1658_5
  have hr := e_1658_5
  simp only [e_1658_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1658 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16428)) := by
  have c101490 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101491 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101492 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101493 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101494 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101495 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1658_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_6
  simp only [← c101490, ← c101491, ← c101492] at e_1658_6
  have e_1658_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_7
  simp only [← c101493, ← c101494, k2, ← c101495] at e_1658_7
  have hr := e_1658_7
  simp only [e_1658_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1658 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16429)) := by
  have c101496 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101497 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101498 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101499 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101500 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101501 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1658_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_8
  simp only [← c101496, ← c101497, ← c101498] at e_1658_8
  have e_1658_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_9
  simp only [← c101499, ← c101500, k2, ← c101501] at e_1658_9
  have hr := e_1658_9
  simp only [e_1658_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1658 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16430)) := by
  have c101502 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101503 := (publicBatchWrapper89_copies3171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101504 := (publicBatchWrapper89_copies3172 a h).1
  have c101505 := (publicBatchWrapper89_copies3172 a h).2.1
  have c101506 := (publicBatchWrapper89_copies3172 a h).2.2.1
  have c101507 := (publicBatchWrapper89_copies3172 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1658_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_10
  simp only [← c101502, ← c101503, ← c101504] at e_1658_10
  have e_1658_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_11
  simp only [← c101505, ← c101506, k2, ← c101507] at e_1658_11
  have hr := e_1658_11
  simp only [e_1658_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1658 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16431)) := by
  have c101508 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.1
  have c101509 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.1
  have c101510 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.1
  have c101511 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.1
  have c101512 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.1
  have c101513 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1658_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_12
  simp only [← c101508, ← c101509, ← c101510] at e_1658_12
  have e_1658_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_13
  simp only [← c101511, ← c101512, k2, ← c101513] at e_1658_13
  have hr := e_1658_13
  simp only [e_1658_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1658 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16432)) := by
  have c101514 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101515 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101516 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101517 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101518 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101519 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1658_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_14
  simp only [← c101514, ← c101515, ← c101516] at e_1658_14
  have e_1658_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_15
  simp only [← c101517, ← c101518, k2, ← c101519] at e_1658_15
  have hr := e_1658_15
  simp only [e_1658_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1658 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16433)) := by
  have c101520 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101521 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101522 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101523 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101524 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101525 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1658_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_16
  simp only [← c101520, ← c101521, ← c101522] at e_1658_16
  have e_1658_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_17
  simp only [← c101523, ← c101524, k2, ← c101525] at e_1658_17
  have hr := e_1658_17
  simp only [e_1658_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1658 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16434)) := by
  have c101526 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101527 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101528 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101529 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101530 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101531 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1658_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_18
  simp only [← c101526, ← c101527, ← c101528] at e_1658_18
  have e_1658_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1658 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1658_19
  simp only [← c101529, ← c101530, k2, ← c101531] at e_1658_19
  have hr := e_1658_19
  simp only [e_1658_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1659 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16435)) := by
  have c101532 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101533 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101534 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101535 := (publicBatchWrapper89_copies3172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101536 := (publicBatchWrapper89_copies3173 a h).1
  have c101537 := (publicBatchWrapper89_copies3173 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1659_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_0
  simp only [← c101532, ← c101533, ← c101534] at e_1659_0
  have e_1659_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_1
  simp only [← c101535, ← c101536, k2, ← c101537] at e_1659_1
  have hr := e_1659_1
  simp only [e_1659_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1659 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16436)) := by
  have c101538 := (publicBatchWrapper89_copies3173 a h).2.2.1
  have c101539 := (publicBatchWrapper89_copies3173 a h).2.2.2.1
  have c101540 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.1
  have c101541 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.1
  have c101542 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.1
  have c101543 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1659_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_2
  simp only [← c101538, ← c101539, ← c101540] at e_1659_2
  have e_1659_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_3
  simp only [← c101541, ← c101542, k2, ← c101543] at e_1659_3
  have hr := e_1659_3
  simp only [e_1659_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1659 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16437)) := by
  have c101544 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.1
  have c101545 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.1
  have c101546 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101547 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101548 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101549 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1659_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_4
  simp only [← c101544, ← c101545, ← c101546] at e_1659_4
  have e_1659_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_5
  simp only [← c101547, ← c101548, k2, ← c101549] at e_1659_5
  have hr := e_1659_5
  simp only [e_1659_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1659 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16438)) := by
  have c101550 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101551 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101552 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101553 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101554 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101555 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1659_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_6
  simp only [← c101550, ← c101551, ← c101552] at e_1659_6
  have e_1659_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_7
  simp only [← c101553, ← c101554, k2, ← c101555] at e_1659_7
  have hr := e_1659_7
  simp only [e_1659_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1659 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16439)) := by
  have c101556 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101557 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101558 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101559 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101560 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101561 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1659_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_8
  simp only [← c101556, ← c101557, ← c101558] at e_1659_8
  have e_1659_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_9
  simp only [← c101559, ← c101560, k2, ← c101561] at e_1659_9
  have hr := e_1659_9
  simp only [e_1659_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1659 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16440)) := by
  have c101562 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101563 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101564 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101565 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101566 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101567 := (publicBatchWrapper89_copies3173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1659_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_10
  simp only [← c101562, ← c101563, ← c101564] at e_1659_10
  have e_1659_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_11
  simp only [← c101565, ← c101566, k2, ← c101567] at e_1659_11
  have hr := e_1659_11
  simp only [e_1659_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1659 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16441)) := by
  have c101568 := (publicBatchWrapper89_copies3174 a h).1
  have c101569 := (publicBatchWrapper89_copies3174 a h).2.1
  have c101570 := (publicBatchWrapper89_copies3174 a h).2.2.1
  have c101571 := (publicBatchWrapper89_copies3174 a h).2.2.2.1
  have c101572 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.1
  have c101573 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1659_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_12
  simp only [← c101568, ← c101569, ← c101570] at e_1659_12
  have e_1659_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_13
  simp only [← c101571, ← c101572, k2, ← c101573] at e_1659_13
  have hr := e_1659_13
  simp only [e_1659_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1659 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16442)) := by
  have c101574 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.1
  have c101575 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.1
  have c101576 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.1
  have c101577 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.1
  have c101578 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101579 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1659_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_14
  simp only [← c101574, ← c101575, ← c101576] at e_1659_14
  have e_1659_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_15
  simp only [← c101577, ← c101578, k2, ← c101579] at e_1659_15
  have hr := e_1659_15
  simp only [e_1659_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1659 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16443)) := by
  have c101580 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101581 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101582 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101583 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101584 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101585 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1659_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_16
  simp only [← c101580, ← c101581, ← c101582] at e_1659_16
  have e_1659_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_17
  simp only [← c101583, ← c101584, k2, ← c101585] at e_1659_17
  have hr := e_1659_17
  simp only [e_1659_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1659 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16444)) := by
  have c101586 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101587 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101588 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101589 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101590 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101591 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1659_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_18
  simp only [← c101586, ← c101587, ← c101588] at e_1659_18
  have e_1659_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1659 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1659_19
  simp only [← c101589, ← c101590, k2, ← c101591] at e_1659_19
  have hr := e_1659_19
  simp only [e_1659_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1660 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16445)) := by
  have c101592 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101593 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101594 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101595 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101596 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101597 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1660_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_0
  simp only [← c101592, ← c101593, ← c101594] at e_1660_0
  have e_1660_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_1
  simp only [← c101595, ← c101596, k2, ← c101597] at e_1660_1
  have hr := e_1660_1
  simp only [e_1660_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1660 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16446)) := by
  have c101598 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101599 := (publicBatchWrapper89_copies3174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101600 := (publicBatchWrapper89_copies3175 a h).1
  have c101601 := (publicBatchWrapper89_copies3175 a h).2.1
  have c101602 := (publicBatchWrapper89_copies3175 a h).2.2.1
  have c101603 := (publicBatchWrapper89_copies3175 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1660_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_2
  simp only [← c101598, ← c101599, ← c101600] at e_1660_2
  have e_1660_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_3
  simp only [← c101601, ← c101602, k2, ← c101603] at e_1660_3
  have hr := e_1660_3
  simp only [e_1660_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1660 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16447)) := by
  have c101604 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.1
  have c101605 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.1
  have c101606 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.1
  have c101607 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.1
  have c101608 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.1
  have c101609 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1660_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_4
  simp only [← c101604, ← c101605, ← c101606] at e_1660_4
  have e_1660_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_5
  simp only [← c101607, ← c101608, k2, ← c101609] at e_1660_5
  have hr := e_1660_5
  simp only [e_1660_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1660 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16448)) := by
  have c101610 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101611 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101612 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101613 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101614 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101615 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1660_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_6
  simp only [← c101610, ← c101611, ← c101612] at e_1660_6
  have e_1660_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_7
  simp only [← c101613, ← c101614, k2, ← c101615] at e_1660_7
  have hr := e_1660_7
  simp only [e_1660_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1660 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16449)) := by
  have c101616 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101617 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101618 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101619 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101620 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101621 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1660_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_8
  simp only [← c101616, ← c101617, ← c101618] at e_1660_8
  have e_1660_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_9
  simp only [← c101619, ← c101620, k2, ← c101621] at e_1660_9
  have hr := e_1660_9
  simp only [e_1660_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1660 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16450)) := by
  have c101622 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101623 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101624 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101625 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101626 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101627 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1660_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_10
  simp only [← c101622, ← c101623, ← c101624] at e_1660_10
  have e_1660_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_11
  simp only [← c101625, ← c101626, k2, ← c101627] at e_1660_11
  have hr := e_1660_11
  simp only [e_1660_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1660 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16451)) := by
  have c101628 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101629 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101630 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101631 := (publicBatchWrapper89_copies3175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101632 := (publicBatchWrapper89_copies3176 a h).1
  have c101633 := (publicBatchWrapper89_copies3176 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1660_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_12
  simp only [← c101628, ← c101629, ← c101630] at e_1660_12
  have e_1660_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_13
  simp only [← c101631, ← c101632, k2, ← c101633] at e_1660_13
  have hr := e_1660_13
  simp only [e_1660_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1660 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16452)) := by
  have c101634 := (publicBatchWrapper89_copies3176 a h).2.2.1
  have c101635 := (publicBatchWrapper89_copies3176 a h).2.2.2.1
  have c101636 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.1
  have c101637 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.1
  have c101638 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.1
  have c101639 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1660_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_14
  simp only [← c101634, ← c101635, ← c101636] at e_1660_14
  have e_1660_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_15
  simp only [← c101637, ← c101638, k2, ← c101639] at e_1660_15
  have hr := e_1660_15
  simp only [e_1660_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1660 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16453)) := by
  have c101640 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.1
  have c101641 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.1
  have c101642 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101643 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101644 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101645 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1660_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_16
  simp only [← c101640, ← c101641, ← c101642] at e_1660_16
  have e_1660_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_17
  simp only [← c101643, ← c101644, k2, ← c101645] at e_1660_17
  have hr := e_1660_17
  simp only [e_1660_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1660 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16454)) := by
  have c101646 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101647 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101648 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101649 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101650 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101651 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1660_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_18
  simp only [← c101646, ← c101647, ← c101648] at e_1660_18
  have e_1660_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1660 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1660_19
  simp only [← c101649, ← c101650, k2, ← c101651] at e_1660_19
  have hr := e_1660_19
  simp only [e_1660_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1661 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16455)) := by
  have c101652 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101653 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101654 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101655 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101656 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101657 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1661_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_0
  simp only [← c101652, ← c101653, ← c101654] at e_1661_0
  have e_1661_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_1
  simp only [← c101655, ← c101656, k2, ← c101657] at e_1661_1
  have hr := e_1661_1
  simp only [e_1661_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1661 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16456)) := by
  have c101658 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101659 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101660 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101661 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101662 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101663 := (publicBatchWrapper89_copies3176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1661_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_2
  simp only [← c101658, ← c101659, ← c101660] at e_1661_2
  have e_1661_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_3
  simp only [← c101661, ← c101662, k2, ← c101663] at e_1661_3
  have hr := e_1661_3
  simp only [e_1661_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1661 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16457)) := by
  have c101664 := (publicBatchWrapper89_copies3177 a h).1
  have c101665 := (publicBatchWrapper89_copies3177 a h).2.1
  have c101666 := (publicBatchWrapper89_copies3177 a h).2.2.1
  have c101667 := (publicBatchWrapper89_copies3177 a h).2.2.2.1
  have c101668 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.1
  have c101669 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1661_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_4
  simp only [← c101664, ← c101665, ← c101666] at e_1661_4
  have e_1661_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_5
  simp only [← c101667, ← c101668, k2, ← c101669] at e_1661_5
  have hr := e_1661_5
  simp only [e_1661_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1661 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16458)) := by
  have c101670 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.1
  have c101671 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.1
  have c101672 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.1
  have c101673 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.1
  have c101674 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101675 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1661_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_6
  simp only [← c101670, ← c101671, ← c101672] at e_1661_6
  have e_1661_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_7
  simp only [← c101673, ← c101674, k2, ← c101675] at e_1661_7
  have hr := e_1661_7
  simp only [e_1661_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1661 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16459)) := by
  have c101676 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101677 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101678 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101679 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101680 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101681 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1661_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_8
  simp only [← c101676, ← c101677, ← c101678] at e_1661_8
  have e_1661_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_9
  simp only [← c101679, ← c101680, k2, ← c101681] at e_1661_9
  have hr := e_1661_9
  simp only [e_1661_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1661 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16460)) := by
  have c101682 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101683 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101684 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101685 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101686 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101687 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1661_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_10
  simp only [← c101682, ← c101683, ← c101684] at e_1661_10
  have e_1661_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_11
  simp only [← c101685, ← c101686, k2, ← c101687] at e_1661_11
  have hr := e_1661_11
  simp only [e_1661_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1661 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16461)) := by
  have c101688 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101689 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101690 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101691 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101692 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101693 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1661_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_12
  simp only [← c101688, ← c101689, ← c101690] at e_1661_12
  have e_1661_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_13
  simp only [← c101691, ← c101692, k2, ← c101693] at e_1661_13
  have hr := e_1661_13
  simp only [e_1661_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1661 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16462)) := by
  have c101694 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101695 := (publicBatchWrapper89_copies3177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101696 := (publicBatchWrapper89_copies3178 a h).1
  have c101697 := (publicBatchWrapper89_copies3178 a h).2.1
  have c101698 := (publicBatchWrapper89_copies3178 a h).2.2.1
  have c101699 := (publicBatchWrapper89_copies3178 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1661_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_14
  simp only [← c101694, ← c101695, ← c101696] at e_1661_14
  have e_1661_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_15
  simp only [← c101697, ← c101698, k2, ← c101699] at e_1661_15
  have hr := e_1661_15
  simp only [e_1661_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1661 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16463)) := by
  have c101700 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.1
  have c101701 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.1
  have c101702 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.1
  have c101703 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.1
  have c101704 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.1
  have c101705 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1661_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_16
  simp only [← c101700, ← c101701, ← c101702] at e_1661_16
  have e_1661_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_17
  simp only [← c101703, ← c101704, k2, ← c101705] at e_1661_17
  have hr := e_1661_17
  simp only [e_1661_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1661 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16464)) := by
  have c101706 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101707 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101708 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101709 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101710 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101711 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1661_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_18
  simp only [← c101706, ← c101707, ← c101708] at e_1661_18
  have e_1661_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1661 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1661_19
  simp only [← c101709, ← c101710, k2, ← c101711] at e_1661_19
  have hr := e_1661_19
  simp only [e_1661_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1662 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16465)) := by
  have c101712 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101713 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101714 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101715 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101716 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101717 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1662_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_0
  simp only [← c101712, ← c101713, ← c101714] at e_1662_0
  have e_1662_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_1
  simp only [← c101715, ← c101716, k2, ← c101717] at e_1662_1
  have hr := e_1662_1
  simp only [e_1662_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1662 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16466)) := by
  have c101718 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101719 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101720 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101721 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101722 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101723 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1662_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_2
  simp only [← c101718, ← c101719, ← c101720] at e_1662_2
  have e_1662_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_3
  simp only [← c101721, ← c101722, k2, ← c101723] at e_1662_3
  have hr := e_1662_3
  simp only [e_1662_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1662 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16467)) := by
  have c101724 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101725 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101726 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101727 := (publicBatchWrapper89_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101728 := (publicBatchWrapper89_copies3179 a h).1
  have c101729 := (publicBatchWrapper89_copies3179 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1662_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_4
  simp only [← c101724, ← c101725, ← c101726] at e_1662_4
  have e_1662_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_5
  simp only [← c101727, ← c101728, k2, ← c101729] at e_1662_5
  have hr := e_1662_5
  simp only [e_1662_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1662 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16468)) := by
  have c101730 := (publicBatchWrapper89_copies3179 a h).2.2.1
  have c101731 := (publicBatchWrapper89_copies3179 a h).2.2.2.1
  have c101732 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.1
  have c101733 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.1
  have c101734 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.1
  have c101735 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1662_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_6
  simp only [← c101730, ← c101731, ← c101732] at e_1662_6
  have e_1662_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_7
  simp only [← c101733, ← c101734, k2, ← c101735] at e_1662_7
  have hr := e_1662_7
  simp only [e_1662_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1662 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16469)) := by
  have c101736 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.1
  have c101737 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.1
  have c101738 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101739 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101740 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101741 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1662_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_8
  simp only [← c101736, ← c101737, ← c101738] at e_1662_8
  have e_1662_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_9
  simp only [← c101739, ← c101740, k2, ← c101741] at e_1662_9
  have hr := e_1662_9
  simp only [e_1662_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1662 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16470)) := by
  have c101742 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101743 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101744 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101745 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101746 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101747 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1662_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_10
  simp only [← c101742, ← c101743, ← c101744] at e_1662_10
  have e_1662_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_11
  simp only [← c101745, ← c101746, k2, ← c101747] at e_1662_11
  have hr := e_1662_11
  simp only [e_1662_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1662 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16471)) := by
  have c101748 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101749 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101750 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101751 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101752 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101753 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1662_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_12
  simp only [← c101748, ← c101749, ← c101750] at e_1662_12
  have e_1662_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_13
  simp only [← c101751, ← c101752, k2, ← c101753] at e_1662_13
  have hr := e_1662_13
  simp only [e_1662_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1662 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16472)) := by
  have c101754 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101755 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101756 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101757 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101758 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101759 := (publicBatchWrapper89_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1662_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_14
  simp only [← c101754, ← c101755, ← c101756] at e_1662_14
  have e_1662_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_15
  simp only [← c101757, ← c101758, k2, ← c101759] at e_1662_15
  have hr := e_1662_15
  simp only [e_1662_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1662 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16473)) := by
  have c101760 := (publicBatchWrapper89_copies3180 a h).1
  have c101761 := (publicBatchWrapper89_copies3180 a h).2.1
  have c101762 := (publicBatchWrapper89_copies3180 a h).2.2.1
  have c101763 := (publicBatchWrapper89_copies3180 a h).2.2.2.1
  have c101764 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.1
  have c101765 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1662_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_16
  simp only [← c101760, ← c101761, ← c101762] at e_1662_16
  have e_1662_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_17
  simp only [← c101763, ← c101764, k2, ← c101765] at e_1662_17
  have hr := e_1662_17
  simp only [e_1662_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1662 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16474)) := by
  have c101766 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.1
  have c101767 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.1
  have c101768 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.1
  have c101769 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.1
  have c101770 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101771 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1662_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_18
  simp only [← c101766, ← c101767, ← c101768] at e_1662_18
  have e_1662_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1662 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1662_19
  simp only [← c101769, ← c101770, k2, ← c101771] at e_1662_19
  have hr := e_1662_19
  simp only [e_1662_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1663 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16475)) := by
  have c101772 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101773 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101774 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101775 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101776 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101777 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1663_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_0
  simp only [← c101772, ← c101773, ← c101774] at e_1663_0
  have e_1663_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_1
  simp only [← c101775, ← c101776, k2, ← c101777] at e_1663_1
  have hr := e_1663_1
  simp only [e_1663_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1663 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16476)) := by
  have c101778 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101779 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101780 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101781 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101782 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101783 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1663_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_2
  simp only [← c101778, ← c101779, ← c101780] at e_1663_2
  have e_1663_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_3
  simp only [← c101781, ← c101782, k2, ← c101783] at e_1663_3
  have hr := e_1663_3
  simp only [e_1663_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1663 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16477)) := by
  have c101784 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101785 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101786 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101787 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101788 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101789 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1663_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_4
  simp only [← c101784, ← c101785, ← c101786] at e_1663_4
  have e_1663_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_5
  simp only [← c101787, ← c101788, k2, ← c101789] at e_1663_5
  have hr := e_1663_5
  simp only [e_1663_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1663 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16478)) := by
  have c101790 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101791 := (publicBatchWrapper89_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101792 := (publicBatchWrapper89_copies3181 a h).1
  have c101793 := (publicBatchWrapper89_copies3181 a h).2.1
  have c101794 := (publicBatchWrapper89_copies3181 a h).2.2.1
  have c101795 := (publicBatchWrapper89_copies3181 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1663_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_6
  simp only [← c101790, ← c101791, ← c101792] at e_1663_6
  have e_1663_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_7
  simp only [← c101793, ← c101794, k2, ← c101795] at e_1663_7
  have hr := e_1663_7
  simp only [e_1663_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1663 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16479)) := by
  have c101796 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.1
  have c101797 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.1
  have c101798 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.1
  have c101799 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.1
  have c101800 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.1
  have c101801 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1663_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_8
  simp only [← c101796, ← c101797, ← c101798] at e_1663_8
  have e_1663_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_9
  simp only [← c101799, ← c101800, k2, ← c101801] at e_1663_9
  have hr := e_1663_9
  simp only [e_1663_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1663 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16480)) := by
  have c101802 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101803 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101804 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101805 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101806 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101807 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1663_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_10
  simp only [← c101802, ← c101803, ← c101804] at e_1663_10
  have e_1663_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_11
  simp only [← c101805, ← c101806, k2, ← c101807] at e_1663_11
  have hr := e_1663_11
  simp only [e_1663_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1663 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16481)) := by
  have c101808 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101809 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101810 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101811 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101812 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101813 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1663_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_12
  simp only [← c101808, ← c101809, ← c101810] at e_1663_12
  have e_1663_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_13
  simp only [← c101811, ← c101812, k2, ← c101813] at e_1663_13
  have hr := e_1663_13
  simp only [e_1663_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1663 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16482)) := by
  have c101814 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101815 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101816 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101817 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101818 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101819 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1663_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_14
  simp only [← c101814, ← c101815, ← c101816] at e_1663_14
  have e_1663_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_15
  simp only [← c101817, ← c101818, k2, ← c101819] at e_1663_15
  have hr := e_1663_15
  simp only [e_1663_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1663 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16483)) := by
  have c101820 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101821 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101822 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101823 := (publicBatchWrapper89_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101824 := (publicBatchWrapper89_copies3182 a h).1
  have c101825 := (publicBatchWrapper89_copies3182 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1663_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_16
  simp only [← c101820, ← c101821, ← c101822] at e_1663_16
  have e_1663_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_17
  simp only [← c101823, ← c101824, k2, ← c101825] at e_1663_17
  have hr := e_1663_17
  simp only [e_1663_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1663 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16484)) := by
  have c101826 := (publicBatchWrapper89_copies3182 a h).2.2.1
  have c101827 := (publicBatchWrapper89_copies3182 a h).2.2.2.1
  have c101828 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.1
  have c101829 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.1
  have c101830 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.1
  have c101831 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1663_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_18
  simp only [← c101826, ← c101827, ← c101828] at e_1663_18
  have e_1663_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1663 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_19
  simp only [← c101829, ← c101830, k2, ← c101831] at e_1663_19
  have hr := e_1663_19
  simp only [e_1663_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1664 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16485)) := by
  have c101832 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.1
  have c101833 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.1
  have c101834 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101835 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101836 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101837 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1664_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_0
  simp only [← c101832, ← c101833, ← c101834] at e_1664_0
  have e_1664_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_1
  simp only [← c101835, ← c101836, k2, ← c101837] at e_1664_1
  have hr := e_1664_1
  simp only [e_1664_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1664 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16486)) := by
  have c101838 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101839 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101840 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101841 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101842 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101843 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1664_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_2
  simp only [← c101838, ← c101839, ← c101840] at e_1664_2
  have e_1664_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_3
  simp only [← c101841, ← c101842, k2, ← c101843] at e_1664_3
  have hr := e_1664_3
  simp only [e_1664_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1664 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16487)) := by
  have c101844 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101845 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101846 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101847 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101848 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101849 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1664_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_4
  simp only [← c101844, ← c101845, ← c101846] at e_1664_4
  have e_1664_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_5
  simp only [← c101847, ← c101848, k2, ← c101849] at e_1664_5
  have hr := e_1664_5
  simp only [e_1664_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1664 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16488)) := by
  have c101850 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101851 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101852 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101853 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101854 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101855 := (publicBatchWrapper89_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1664_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_6
  simp only [← c101850, ← c101851, ← c101852] at e_1664_6
  have e_1664_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_7
  simp only [← c101853, ← c101854, k2, ← c101855] at e_1664_7
  have hr := e_1664_7
  simp only [e_1664_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1664 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16489)) := by
  have c101856 := (publicBatchWrapper89_copies3183 a h).1
  have c101857 := (publicBatchWrapper89_copies3183 a h).2.1
  have c101858 := (publicBatchWrapper89_copies3183 a h).2.2.1
  have c101859 := (publicBatchWrapper89_copies3183 a h).2.2.2.1
  have c101860 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.1
  have c101861 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1664_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_8
  simp only [← c101856, ← c101857, ← c101858] at e_1664_8
  have e_1664_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_9
  simp only [← c101859, ← c101860, k2, ← c101861] at e_1664_9
  have hr := e_1664_9
  simp only [e_1664_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1664 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16490)) := by
  have c101862 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.1
  have c101863 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.1
  have c101864 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.1
  have c101865 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.1
  have c101866 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101867 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1664_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_10
  simp only [← c101862, ← c101863, ← c101864] at e_1664_10
  have e_1664_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_11
  simp only [← c101865, ← c101866, k2, ← c101867] at e_1664_11
  have hr := e_1664_11
  simp only [e_1664_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1664 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16491)) := by
  have c101868 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101869 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101870 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101871 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101872 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101873 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1664_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_12
  simp only [← c101868, ← c101869, ← c101870] at e_1664_12
  have e_1664_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_13
  simp only [← c101871, ← c101872, k2, ← c101873] at e_1664_13
  have hr := e_1664_13
  simp only [e_1664_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1664 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16492)) := by
  have c101874 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101875 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101876 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101877 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101878 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101879 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1664_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_14
  simp only [← c101874, ← c101875, ← c101876] at e_1664_14
  have e_1664_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_15
  simp only [← c101877, ← c101878, k2, ← c101879] at e_1664_15
  have hr := e_1664_15
  simp only [e_1664_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1664 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16493)) := by
  have c101880 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101881 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101882 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101883 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101884 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101885 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1664_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_16
  simp only [← c101880, ← c101881, ← c101882] at e_1664_16
  have e_1664_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_17
  simp only [← c101883, ← c101884, k2, ← c101885] at e_1664_17
  have hr := e_1664_17
  simp only [e_1664_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1664 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16494)) := by
  have c101886 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101887 := (publicBatchWrapper89_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101888 := (publicBatchWrapper89_copies3184 a h).1
  have c101889 := (publicBatchWrapper89_copies3184 a h).2.1
  have c101890 := (publicBatchWrapper89_copies3184 a h).2.2.1
  have c101891 := (publicBatchWrapper89_copies3184 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1664_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_18
  simp only [← c101886, ← c101887, ← c101888] at e_1664_18
  have e_1664_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1664 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_19
  simp only [← c101889, ← c101890, k2, ← c101891] at e_1664_19
  have hr := e_1664_19
  simp only [e_1664_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1665 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16495)) := by
  have c101892 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.1
  have c101893 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.1
  have c101894 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.1
  have c101895 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.1
  have c101896 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.1
  have c101897 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1665_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_0
  simp only [← c101892, ← c101893, ← c101894] at e_1665_0
  have e_1665_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_1
  simp only [← c101895, ← c101896, k2, ← c101897] at e_1665_1
  have hr := e_1665_1
  simp only [e_1665_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1665 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16496)) := by
  have c101898 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101899 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101900 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101901 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101902 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101903 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1665_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_2
  simp only [← c101898, ← c101899, ← c101900] at e_1665_2
  have e_1665_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_3
  simp only [← c101901, ← c101902, k2, ← c101903] at e_1665_3
  have hr := e_1665_3
  simp only [e_1665_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1665 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16497)) := by
  have c101904 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101905 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101906 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101907 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101908 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101909 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1665_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_4
  simp only [← c101904, ← c101905, ← c101906] at e_1665_4
  have e_1665_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_5
  simp only [← c101907, ← c101908, k2, ← c101909] at e_1665_5
  have hr := e_1665_5
  simp only [e_1665_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1665 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16498)) := by
  have c101910 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101911 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101912 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101913 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101914 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101915 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1665_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_6
  simp only [← c101910, ← c101911, ← c101912] at e_1665_6
  have e_1665_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_7
  simp only [← c101913, ← c101914, k2, ← c101915] at e_1665_7
  have hr := e_1665_7
  simp only [e_1665_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1665 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16499)) := by
  have c101916 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101917 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101918 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101919 := (publicBatchWrapper89_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101920 := (publicBatchWrapper89_copies3185 a h).1
  have c101921 := (publicBatchWrapper89_copies3185 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1665_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_8
  simp only [← c101916, ← c101917, ← c101918] at e_1665_8
  have e_1665_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_9
  simp only [← c101919, ← c101920, k2, ← c101921] at e_1665_9
  have hr := e_1665_9
  simp only [e_1665_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1665 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16500)) := by
  have c101922 := (publicBatchWrapper89_copies3185 a h).2.2.1
  have c101923 := (publicBatchWrapper89_copies3185 a h).2.2.2.1
  have c101924 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.1
  have c101925 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.1
  have c101926 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.1
  have c101927 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1665_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_10
  simp only [← c101922, ← c101923, ← c101924] at e_1665_10
  have e_1665_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_11
  simp only [← c101925, ← c101926, k2, ← c101927] at e_1665_11
  have hr := e_1665_11
  simp only [e_1665_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1665 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16501)) := by
  have c101928 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.1
  have c101929 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.1
  have c101930 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101931 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101932 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101933 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1665_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_12
  simp only [← c101928, ← c101929, ← c101930] at e_1665_12
  have e_1665_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_13
  simp only [← c101931, ← c101932, k2, ← c101933] at e_1665_13
  have hr := e_1665_13
  simp only [e_1665_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1665 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16502)) := by
  have c101934 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101935 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101936 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101937 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101938 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101939 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1665_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_14
  simp only [← c101934, ← c101935, ← c101936] at e_1665_14
  have e_1665_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_15
  simp only [← c101937, ← c101938, k2, ← c101939] at e_1665_15
  have hr := e_1665_15
  simp only [e_1665_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1665 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16503)) := by
  have c101940 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101941 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101942 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101943 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101944 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101945 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1665_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_16
  simp only [← c101940, ← c101941, ← c101942] at e_1665_16
  have e_1665_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_17
  simp only [← c101943, ← c101944, k2, ← c101945] at e_1665_17
  have hr := e_1665_17
  simp only [e_1665_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1665 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16504)) := by
  have c101946 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101947 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101948 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101949 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101950 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101951 := (publicBatchWrapper89_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1665_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_18
  simp only [← c101946, ← c101947, ← c101948] at e_1665_18
  have e_1665_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1665 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_19
  simp only [← c101949, ← c101950, k2, ← c101951] at e_1665_19
  have hr := e_1665_19
  simp only [e_1665_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1666 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16505)) := by
  have c101952 := (publicBatchWrapper89_copies3186 a h).1
  have c101953 := (publicBatchWrapper89_copies3186 a h).2.1
  have c101954 := (publicBatchWrapper89_copies3186 a h).2.2.1
  have c101955 := (publicBatchWrapper89_copies3186 a h).2.2.2.1
  have c101956 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.1
  have c101957 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1666_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_0
  simp only [← c101952, ← c101953, ← c101954] at e_1666_0
  have e_1666_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_1
  simp only [← c101955, ← c101956, k2, ← c101957] at e_1666_1
  have hr := e_1666_1
  simp only [e_1666_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1666 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16506)) := by
  have c101958 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.1
  have c101959 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.1
  have c101960 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.1
  have c101961 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.1
  have c101962 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101963 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1666_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_2
  simp only [← c101958, ← c101959, ← c101960] at e_1666_2
  have e_1666_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_3
  simp only [← c101961, ← c101962, k2, ← c101963] at e_1666_3
  have hr := e_1666_3
  simp only [e_1666_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1666 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16507)) := by
  have c101964 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101965 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101966 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101967 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101968 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101969 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1666_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_4
  simp only [← c101964, ← c101965, ← c101966] at e_1666_4
  have e_1666_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_5
  simp only [← c101967, ← c101968, k2, ← c101969] at e_1666_5
  have hr := e_1666_5
  simp only [e_1666_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1666 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16508)) := by
  have c101970 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101971 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101972 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101973 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101974 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101975 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1666_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_6
  simp only [← c101970, ← c101971, ← c101972] at e_1666_6
  have e_1666_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_7
  simp only [← c101973, ← c101974, k2, ← c101975] at e_1666_7
  have hr := e_1666_7
  simp only [e_1666_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1666 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16509)) := by
  have c101976 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101977 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101978 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101979 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101980 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101981 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1666_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_8
  simp only [← c101976, ← c101977, ← c101978] at e_1666_8
  have e_1666_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_9
  simp only [← c101979, ← c101980, k2, ← c101981] at e_1666_9
  have hr := e_1666_9
  simp only [e_1666_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1666 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16510)) := by
  have c101982 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101983 := (publicBatchWrapper89_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101984 := (publicBatchWrapper89_copies3187 a h).1
  have c101985 := (publicBatchWrapper89_copies3187 a h).2.1
  have c101986 := (publicBatchWrapper89_copies3187 a h).2.2.1
  have c101987 := (publicBatchWrapper89_copies3187 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1666_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_10
  simp only [← c101982, ← c101983, ← c101984] at e_1666_10
  have e_1666_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_11
  simp only [← c101985, ← c101986, k2, ← c101987] at e_1666_11
  have hr := e_1666_11
  simp only [e_1666_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1666 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16511)) := by
  have c101988 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.1
  have c101989 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.1
  have c101990 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.1
  have c101991 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.1
  have c101992 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.1
  have c101993 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1666_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_12
  simp only [← c101988, ← c101989, ← c101990] at e_1666_12
  have e_1666_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_13
  simp only [← c101991, ← c101992, k2, ← c101993] at e_1666_13
  have hr := e_1666_13
  simp only [e_1666_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1666 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16512)) := by
  have c101994 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101995 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101996 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101997 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101998 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101999 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1666_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_14
  simp only [← c101994, ← c101995, ← c101996] at e_1666_14
  have e_1666_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_15
  simp only [← c101997, ← c101998, k2, ← c101999] at e_1666_15
  have hr := e_1666_15
  simp only [e_1666_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1666 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16513)) := by
  have c102000 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102001 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102002 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102003 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102004 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102005 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1666_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_16
  simp only [← c102000, ← c102001, ← c102002] at e_1666_16
  have e_1666_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_17
  simp only [← c102003, ← c102004, k2, ← c102005] at e_1666_17
  have hr := e_1666_17
  simp only [e_1666_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1666 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16514)) := by
  have c102006 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102007 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102008 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102009 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102010 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102011 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1666_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_18
  simp only [← c102006, ← c102007, ← c102008] at e_1666_18
  have e_1666_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1666 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_19
  simp only [← c102009, ← c102010, k2, ← c102011] at e_1666_19
  have hr := e_1666_19
  simp only [e_1666_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1667 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16515)) := by
  have c102012 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102013 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102014 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102015 := (publicBatchWrapper89_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102016 := (publicBatchWrapper89_copies3188 a h).1
  have c102017 := (publicBatchWrapper89_copies3188 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1667_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_0
  simp only [← c102012, ← c102013, ← c102014] at e_1667_0
  have e_1667_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_1
  simp only [← c102015, ← c102016, k2, ← c102017] at e_1667_1
  have hr := e_1667_1
  simp only [e_1667_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1667 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16516)) := by
  have c102018 := (publicBatchWrapper89_copies3188 a h).2.2.1
  have c102019 := (publicBatchWrapper89_copies3188 a h).2.2.2.1
  have c102020 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.1
  have c102021 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.1
  have c102022 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.1
  have c102023 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1667_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_2
  simp only [← c102018, ← c102019, ← c102020] at e_1667_2
  have e_1667_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_3
  simp only [← c102021, ← c102022, k2, ← c102023] at e_1667_3
  have hr := e_1667_3
  simp only [e_1667_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1667 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16517)) := by
  have c102024 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.1
  have c102025 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.1
  have c102026 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102027 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102028 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102029 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1667_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_4
  simp only [← c102024, ← c102025, ← c102026] at e_1667_4
  have e_1667_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_5
  simp only [← c102027, ← c102028, k2, ← c102029] at e_1667_5
  have hr := e_1667_5
  simp only [e_1667_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1667 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16518)) := by
  have c102030 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102031 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102032 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102033 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102034 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102035 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1667_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_6
  simp only [← c102030, ← c102031, ← c102032] at e_1667_6
  have e_1667_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_7
  simp only [← c102033, ← c102034, k2, ← c102035] at e_1667_7
  have hr := e_1667_7
  simp only [e_1667_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1667 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16519)) := by
  have c102036 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102037 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102038 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102039 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102040 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102041 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1667_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_8
  simp only [← c102036, ← c102037, ← c102038] at e_1667_8
  have e_1667_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_9
  simp only [← c102039, ← c102040, k2, ← c102041] at e_1667_9
  have hr := e_1667_9
  simp only [e_1667_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1667 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16520)) := by
  have c102042 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102043 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102044 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102045 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102046 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102047 := (publicBatchWrapper89_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1667_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_10
  simp only [← c102042, ← c102043, ← c102044] at e_1667_10
  have e_1667_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_11
  simp only [← c102045, ← c102046, k2, ← c102047] at e_1667_11
  have hr := e_1667_11
  simp only [e_1667_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1667 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16521)) := by
  have c102048 := (publicBatchWrapper89_copies3189 a h).1
  have c102049 := (publicBatchWrapper89_copies3189 a h).2.1
  have c102050 := (publicBatchWrapper89_copies3189 a h).2.2.1
  have c102051 := (publicBatchWrapper89_copies3189 a h).2.2.2.1
  have c102052 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.1
  have c102053 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1667_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_12
  simp only [← c102048, ← c102049, ← c102050] at e_1667_12
  have e_1667_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_13
  simp only [← c102051, ← c102052, k2, ← c102053] at e_1667_13
  have hr := e_1667_13
  simp only [e_1667_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1667 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16522)) := by
  have c102054 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.1
  have c102055 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.1
  have c102056 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.1
  have c102057 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.1
  have c102058 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102059 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1667_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_14
  simp only [← c102054, ← c102055, ← c102056] at e_1667_14
  have e_1667_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_15
  simp only [← c102057, ← c102058, k2, ← c102059] at e_1667_15
  have hr := e_1667_15
  simp only [e_1667_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1667 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16523)) := by
  have c102060 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102061 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102062 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102063 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102064 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102065 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1667_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_16
  simp only [← c102060, ← c102061, ← c102062] at e_1667_16
  have e_1667_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_17
  simp only [← c102063, ← c102064, k2, ← c102065] at e_1667_17
  have hr := e_1667_17
  simp only [e_1667_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1667 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16524)) := by
  have c102066 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102067 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102068 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102069 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102070 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102071 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1667_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_18
  simp only [← c102066, ← c102067, ← c102068] at e_1667_18
  have e_1667_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1667 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_19
  simp only [← c102069, ← c102070, k2, ← c102071] at e_1667_19
  have hr := e_1667_19
  simp only [e_1667_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1668 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16525)) := by
  have c102072 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102073 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102074 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102075 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102076 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102077 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1668_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_0
  simp only [← c102072, ← c102073, ← c102074] at e_1668_0
  have e_1668_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_1
  simp only [← c102075, ← c102076, k2, ← c102077] at e_1668_1
  have hr := e_1668_1
  simp only [e_1668_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1668 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16526)) := by
  have c102078 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102079 := (publicBatchWrapper89_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102080 := (publicBatchWrapper89_copies3190 a h).1
  have c102081 := (publicBatchWrapper89_copies3190 a h).2.1
  have c102082 := (publicBatchWrapper89_copies3190 a h).2.2.1
  have c102083 := (publicBatchWrapper89_copies3190 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1668_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_2
  simp only [← c102078, ← c102079, ← c102080] at e_1668_2
  have e_1668_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_3
  simp only [← c102081, ← c102082, k2, ← c102083] at e_1668_3
  have hr := e_1668_3
  simp only [e_1668_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1668 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16527)) := by
  have c102084 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.1
  have c102085 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.1
  have c102086 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.1
  have c102087 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.1
  have c102088 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.1
  have c102089 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1668_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_4
  simp only [← c102084, ← c102085, ← c102086] at e_1668_4
  have e_1668_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_5
  simp only [← c102087, ← c102088, k2, ← c102089] at e_1668_5
  have hr := e_1668_5
  simp only [e_1668_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1668 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16528)) := by
  have c102090 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102091 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102092 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102093 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102094 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102095 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1668_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_6
  simp only [← c102090, ← c102091, ← c102092] at e_1668_6
  have e_1668_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_7
  simp only [← c102093, ← c102094, k2, ← c102095] at e_1668_7
  have hr := e_1668_7
  simp only [e_1668_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1668 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16529)) := by
  have c102096 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102097 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102098 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102099 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102100 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102101 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1668_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_8
  simp only [← c102096, ← c102097, ← c102098] at e_1668_8
  have e_1668_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_9
  simp only [← c102099, ← c102100, k2, ← c102101] at e_1668_9
  have hr := e_1668_9
  simp only [e_1668_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1668 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16530)) := by
  have c102102 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102103 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102104 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102105 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102106 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102107 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1668_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_10
  simp only [← c102102, ← c102103, ← c102104] at e_1668_10
  have e_1668_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_11
  simp only [← c102105, ← c102106, k2, ← c102107] at e_1668_11
  have hr := e_1668_11
  simp only [e_1668_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1668 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16531)) := by
  have c102108 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102109 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102110 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102111 := (publicBatchWrapper89_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102112 := (publicBatchWrapper89_copies3191 a h).1
  have c102113 := (publicBatchWrapper89_copies3191 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1668_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_12
  simp only [← c102108, ← c102109, ← c102110] at e_1668_12
  have e_1668_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_13
  simp only [← c102111, ← c102112, k2, ← c102113] at e_1668_13
  have hr := e_1668_13
  simp only [e_1668_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1668 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16532)) := by
  have c102114 := (publicBatchWrapper89_copies3191 a h).2.2.1
  have c102115 := (publicBatchWrapper89_copies3191 a h).2.2.2.1
  have c102116 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.1
  have c102117 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.1
  have c102118 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.1
  have c102119 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1668_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_14
  simp only [← c102114, ← c102115, ← c102116] at e_1668_14
  have e_1668_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_15
  simp only [← c102117, ← c102118, k2, ← c102119] at e_1668_15
  have hr := e_1668_15
  simp only [e_1668_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1668 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16533)) := by
  have c102120 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.1
  have c102121 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.1
  have c102122 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102123 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102124 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102125 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1668_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_16
  simp only [← c102120, ← c102121, ← c102122] at e_1668_16
  have e_1668_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_17
  simp only [← c102123, ← c102124, k2, ← c102125] at e_1668_17
  have hr := e_1668_17
  simp only [e_1668_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1668 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16534)) := by
  have c102126 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102127 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102128 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102129 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102130 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102131 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1668_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_18
  simp only [← c102126, ← c102127, ← c102128] at e_1668_18
  have e_1668_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1668 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_19
  simp only [← c102129, ← c102130, k2, ← c102131] at e_1668_19
  have hr := e_1668_19
  simp only [e_1668_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1669 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16535)) := by
  have c102132 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102133 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102134 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102135 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102136 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102137 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1669_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_0
  simp only [← c102132, ← c102133, ← c102134] at e_1669_0
  have e_1669_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_1
  simp only [← c102135, ← c102136, k2, ← c102137] at e_1669_1
  have hr := e_1669_1
  simp only [e_1669_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1669 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16536)) := by
  have c102138 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102139 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102140 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102141 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102142 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102143 := (publicBatchWrapper89_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1669_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_2
  simp only [← c102138, ← c102139, ← c102140] at e_1669_2
  have e_1669_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_3
  simp only [← c102141, ← c102142, k2, ← c102143] at e_1669_3
  have hr := e_1669_3
  simp only [e_1669_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1669 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16537)) := by
  have c102144 := (publicBatchWrapper89_copies3192 a h).1
  have c102145 := (publicBatchWrapper89_copies3192 a h).2.1
  have c102146 := (publicBatchWrapper89_copies3192 a h).2.2.1
  have c102147 := (publicBatchWrapper89_copies3192 a h).2.2.2.1
  have c102148 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.1
  have c102149 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1669_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_4
  simp only [← c102144, ← c102145, ← c102146] at e_1669_4
  have e_1669_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_5
  simp only [← c102147, ← c102148, k2, ← c102149] at e_1669_5
  have hr := e_1669_5
  simp only [e_1669_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1669 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16538)) := by
  have c102150 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.1
  have c102151 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.1
  have c102152 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.1
  have c102153 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.1
  have c102154 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102155 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1669_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_6
  simp only [← c102150, ← c102151, ← c102152] at e_1669_6
  have e_1669_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_7
  simp only [← c102153, ← c102154, k2, ← c102155] at e_1669_7
  have hr := e_1669_7
  simp only [e_1669_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1669 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16539)) := by
  have c102156 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102157 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102158 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102159 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102160 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102161 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1669_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_8
  simp only [← c102156, ← c102157, ← c102158] at e_1669_8
  have e_1669_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_9
  simp only [← c102159, ← c102160, k2, ← c102161] at e_1669_9
  have hr := e_1669_9
  simp only [e_1669_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1669 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16540)) := by
  have c102162 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102163 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102164 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102165 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102166 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102167 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1669_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_10
  simp only [← c102162, ← c102163, ← c102164] at e_1669_10
  have e_1669_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_11
  simp only [← c102165, ← c102166, k2, ← c102167] at e_1669_11
  have hr := e_1669_11
  simp only [e_1669_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1669 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16541)) := by
  have c102168 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102169 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102170 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102171 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102172 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102173 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1669_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_12
  simp only [← c102168, ← c102169, ← c102170] at e_1669_12
  have e_1669_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_13
  simp only [← c102171, ← c102172, k2, ← c102173] at e_1669_13
  have hr := e_1669_13
  simp only [e_1669_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1669 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16542)) := by
  have c102174 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102175 := (publicBatchWrapper89_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102176 := (publicBatchWrapper89_copies3193 a h).1
  have c102177 := (publicBatchWrapper89_copies3193 a h).2.1
  have c102178 := (publicBatchWrapper89_copies3193 a h).2.2.1
  have c102179 := (publicBatchWrapper89_copies3193 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1669_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_14
  simp only [← c102174, ← c102175, ← c102176] at e_1669_14
  have e_1669_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_15
  simp only [← c102177, ← c102178, k2, ← c102179] at e_1669_15
  have hr := e_1669_15
  simp only [e_1669_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1669 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16543)) := by
  have c102180 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.1
  have c102181 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.1
  have c102182 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.1
  have c102183 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.1
  have c102184 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.1
  have c102185 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1669_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_16
  simp only [← c102180, ← c102181, ← c102182] at e_1669_16
  have e_1669_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_17
  simp only [← c102183, ← c102184, k2, ← c102185] at e_1669_17
  have hr := e_1669_17
  simp only [e_1669_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1669 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16544)) := by
  have c102186 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102187 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102188 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102189 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102190 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102191 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1669_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_18
  simp only [← c102186, ← c102187, ← c102188] at e_1669_18
  have e_1669_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1669 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_19
  simp only [← c102189, ← c102190, k2, ← c102191] at e_1669_19
  have hr := e_1669_19
  simp only [e_1669_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1670 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16545)) := by
  have c102192 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102193 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102194 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102195 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102196 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102197 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1670_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_0
  simp only [← c102192, ← c102193, ← c102194] at e_1670_0
  have e_1670_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_1
  simp only [← c102195, ← c102196, k2, ← c102197] at e_1670_1
  have hr := e_1670_1
  simp only [e_1670_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1670 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16546)) := by
  have c102198 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102199 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102200 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102201 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102202 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102203 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1670_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_2
  simp only [← c102198, ← c102199, ← c102200] at e_1670_2
  have e_1670_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_3
  simp only [← c102201, ← c102202, k2, ← c102203] at e_1670_3
  have hr := e_1670_3
  simp only [e_1670_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1670 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16547)) := by
  have c102204 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102205 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102206 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102207 := (publicBatchWrapper89_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102208 := (publicBatchWrapper89_copies3194 a h).1
  have c102209 := (publicBatchWrapper89_copies3194 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1670_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_4
  simp only [← c102204, ← c102205, ← c102206] at e_1670_4
  have e_1670_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_5
  simp only [← c102207, ← c102208, k2, ← c102209] at e_1670_5
  have hr := e_1670_5
  simp only [e_1670_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1670 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16548)) := by
  have c102210 := (publicBatchWrapper89_copies3194 a h).2.2.1
  have c102211 := (publicBatchWrapper89_copies3194 a h).2.2.2.1
  have c102212 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.1
  have c102213 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.1
  have c102214 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.1
  have c102215 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1670_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_6
  simp only [← c102210, ← c102211, ← c102212] at e_1670_6
  have e_1670_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_7
  simp only [← c102213, ← c102214, k2, ← c102215] at e_1670_7
  have hr := e_1670_7
  simp only [e_1670_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1670 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16549)) := by
  have c102216 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.1
  have c102217 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.1
  have c102218 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102219 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102220 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102221 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1670_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_8
  simp only [← c102216, ← c102217, ← c102218] at e_1670_8
  have e_1670_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_9
  simp only [← c102219, ← c102220, k2, ← c102221] at e_1670_9
  have hr := e_1670_9
  simp only [e_1670_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1670 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16550)) := by
  have c102222 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102223 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102224 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102225 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102226 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102227 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1670_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_10
  simp only [← c102222, ← c102223, ← c102224] at e_1670_10
  have e_1670_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_11
  simp only [← c102225, ← c102226, k2, ← c102227] at e_1670_11
  have hr := e_1670_11
  simp only [e_1670_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1670 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16551)) := by
  have c102228 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102229 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102230 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102231 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102232 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102233 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1670_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_12
  simp only [← c102228, ← c102229, ← c102230] at e_1670_12
  have e_1670_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_13
  simp only [← c102231, ← c102232, k2, ← c102233] at e_1670_13
  have hr := e_1670_13
  simp only [e_1670_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1670 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16552)) := by
  have c102234 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102235 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102236 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102237 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102238 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102239 := (publicBatchWrapper89_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1670_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_14
  simp only [← c102234, ← c102235, ← c102236] at e_1670_14
  have e_1670_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_15
  simp only [← c102237, ← c102238, k2, ← c102239] at e_1670_15
  have hr := e_1670_15
  simp only [e_1670_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1670 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16553)) := by
  have c102240 := (publicBatchWrapper89_copies3195 a h).1
  have c102241 := (publicBatchWrapper89_copies3195 a h).2.1
  have c102242 := (publicBatchWrapper89_copies3195 a h).2.2.1
  have c102243 := (publicBatchWrapper89_copies3195 a h).2.2.2.1
  have c102244 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.1
  have c102245 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1670_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_16
  simp only [← c102240, ← c102241, ← c102242] at e_1670_16
  have e_1670_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_17
  simp only [← c102243, ← c102244, k2, ← c102245] at e_1670_17
  have hr := e_1670_17
  simp only [e_1670_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1670 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16554)) := by
  have c102246 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.1
  have c102247 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.1
  have c102248 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.1
  have c102249 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.1
  have c102250 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102251 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1670_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_18
  simp only [← c102246, ← c102247, ← c102248] at e_1670_18
  have e_1670_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1670 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_19
  simp only [← c102249, ← c102250, k2, ← c102251] at e_1670_19
  have hr := e_1670_19
  simp only [e_1670_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1671 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16555)) := by
  have c102252 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102253 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102254 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102255 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102256 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102257 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1671_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_0
  simp only [← c102252, ← c102253, ← c102254] at e_1671_0
  have e_1671_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_1
  simp only [← c102255, ← c102256, k2, ← c102257] at e_1671_1
  have hr := e_1671_1
  simp only [e_1671_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1671 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16556)) := by
  have c102258 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102259 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102260 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102261 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102262 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102263 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1671_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_2
  simp only [← c102258, ← c102259, ← c102260] at e_1671_2
  have e_1671_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_3
  simp only [← c102261, ← c102262, k2, ← c102263] at e_1671_3
  have hr := e_1671_3
  simp only [e_1671_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1671 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16557)) := by
  have c102264 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102265 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102266 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102267 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102268 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102269 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1671_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_4
  simp only [← c102264, ← c102265, ← c102266] at e_1671_4
  have e_1671_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_5
  simp only [← c102267, ← c102268, k2, ← c102269] at e_1671_5
  have hr := e_1671_5
  simp only [e_1671_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1671 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16558)) := by
  have c102270 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102271 := (publicBatchWrapper89_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102272 := (publicBatchWrapper89_copies3196 a h).1
  have c102273 := (publicBatchWrapper89_copies3196 a h).2.1
  have c102274 := (publicBatchWrapper89_copies3196 a h).2.2.1
  have c102275 := (publicBatchWrapper89_copies3196 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1671_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_6
  simp only [← c102270, ← c102271, ← c102272] at e_1671_6
  have e_1671_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_7
  simp only [← c102273, ← c102274, k2, ← c102275] at e_1671_7
  have hr := e_1671_7
  simp only [e_1671_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1671 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16559)) := by
  have c102276 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.1
  have c102277 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.1
  have c102278 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.1
  have c102279 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.1
  have c102280 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.1
  have c102281 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1671_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_8
  simp only [← c102276, ← c102277, ← c102278] at e_1671_8
  have e_1671_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_9
  simp only [← c102279, ← c102280, k2, ← c102281] at e_1671_9
  have hr := e_1671_9
  simp only [e_1671_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1671 47) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16560)) := by
  have c102282 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102283 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102284 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102285 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102286 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102287 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1671_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_10
  simp only [← c102282, ← c102283, ← c102284] at e_1671_10
  have e_1671_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_11
  simp only [← c102285, ← c102286, k2, ← c102287] at e_1671_11
  have hr := e_1671_11
  simp only [e_1671_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1671 55) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16561)) := by
  have c102288 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102289 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102290 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102291 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102292 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102293 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1671_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_12
  simp only [← c102288, ← c102289, ← c102290] at e_1671_12
  have e_1671_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_13
  simp only [← c102291, ← c102292, k2, ← c102293] at e_1671_13
  have hr := e_1671_13
  simp only [e_1671_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1671 63) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16562)) := by
  have c102294 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102295 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102296 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102297 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102298 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102299 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1671_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_14
  simp only [← c102294, ← c102295, ← c102296] at e_1671_14
  have e_1671_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_15
  simp only [← c102297, ← c102298, k2, ← c102299] at e_1671_15
  have hr := e_1671_15
  simp only [e_1671_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1671 71) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16563)) := by
  have c102300 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102301 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102302 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102303 := (publicBatchWrapper89_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102304 := (publicBatchWrapper89_copies3197 a h).1
  have c102305 := (publicBatchWrapper89_copies3197 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1671_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_16
  simp only [← c102300, ← c102301, ← c102302] at e_1671_16
  have e_1671_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_17
  simp only [← c102303, ← c102304, k2, ← c102305] at e_1671_17
  have hr := e_1671_17
  simp only [e_1671_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1671 79) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16564)) := by
  have c102306 := (publicBatchWrapper89_copies3197 a h).2.2.1
  have c102307 := (publicBatchWrapper89_copies3197 a h).2.2.2.1
  have c102308 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.1
  have c102309 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.1
  have c102310 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.1
  have c102311 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1671_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_18
  simp only [← c102306, ← c102307, ← c102308] at e_1671_18
  have e_1671_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1671 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_19
  simp only [← c102309, ← c102310, k2, ← c102311] at e_1671_19
  have hr := e_1671_19
  simp only [e_1671_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1672 7) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16565)) := by
  have c102312 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.1
  have c102313 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.1
  have c102314 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102315 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102316 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102317 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1672_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_0
  simp only [← c102312, ← c102313, ← c102314] at e_1672_0
  have e_1672_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_1
  simp only [← c102315, ← c102316, k2, ← c102317] at e_1672_1
  have hr := e_1672_1
  simp only [e_1672_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1672 15) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16566)) := by
  have c102318 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102319 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102320 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102321 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102322 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102323 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1672_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_2
  simp only [← c102318, ← c102319, ← c102320] at e_1672_2
  have e_1672_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_3
  simp only [← c102321, ← c102322, k2, ← c102323] at e_1672_3
  have hr := e_1672_3
  simp only [e_1672_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1672 23) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16567)) := by
  have c102324 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102325 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102326 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102327 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102328 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102329 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1672_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_4
  simp only [← c102324, ← c102325, ← c102326] at e_1672_4
  have e_1672_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_5
  simp only [← c102327, ← c102328, k2, ← c102329] at e_1672_5
  have hr := e_1672_5
  simp only [e_1672_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1672 31) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16568)) := by
  have c102330 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102331 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102332 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102333 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102334 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102335 := (publicBatchWrapper89_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1672_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_6
  simp only [← c102330, ← c102331, ← c102332] at e_1672_6
  have e_1672_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_7
  simp only [← c102333, ← c102334, k2, ← c102335] at e_1672_7
  have hr := e_1672_7
  simp only [e_1672_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1672 39) = bselect (a (.wire 79 67)) (a (.virt 200)) (a (.virt 16569)) := by
  have c102336 := (publicBatchWrapper89_copies3198 a h).1
  have c102337 := (publicBatchWrapper89_copies3198 a h).2.1
  have c102338 := (publicBatchWrapper89_copies3198 a h).2.2.1
  have c102339 := (publicBatchWrapper89_copies3198 a h).2.2.2.1
  have c102340 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.1
  have c102341 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1672_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_8
  simp only [← c102336, ← c102337, ← c102338] at e_1672_8
  have e_1672_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_9
  simp only [← c102339, ← c102340, k2, ← c102341] at e_1672_9
  have hr := e_1672_9
  simp only [e_1672_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1672 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16610)) := by
  have c102342 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.1
  have c102343 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.1
  have c102344 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.1
  have c102345 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.1
  have c102346 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102347 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1672_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_10
  simp only [← c102342, ← c102343, ← c102344] at e_1672_10
  have e_1672_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_11
  simp only [← c102345, ← c102346, k2, ← c102347] at e_1672_11
  have hr := e_1672_11
  simp only [e_1672_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1672 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16611)) := by
  have c102348 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102349 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102350 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102351 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102352 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102353 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1672_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_12
  simp only [← c102348, ← c102349, ← c102350] at e_1672_12
  have e_1672_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_13
  simp only [← c102351, ← c102352, k2, ← c102353] at e_1672_13
  have hr := e_1672_13
  simp only [e_1672_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1672 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16612)) := by
  have c102354 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102355 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102356 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102357 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102358 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102359 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1672_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_14
  simp only [← c102354, ← c102355, ← c102356] at e_1672_14
  have e_1672_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_15
  simp only [← c102357, ← c102358, k2, ← c102359] at e_1672_15
  have hr := e_1672_15
  simp only [e_1672_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1672 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16613)) := by
  have c102360 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102361 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102362 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102363 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102364 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102365 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1672_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_16
  simp only [← c102360, ← c102361, ← c102362] at e_1672_16
  have e_1672_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_17
  simp only [← c102363, ← c102364, k2, ← c102365] at e_1672_17
  have hr := e_1672_17
  simp only [e_1672_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1672 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16614)) := by
  have c102366 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102367 := (publicBatchWrapper89_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102368 := (publicBatchWrapper89_copies3199 a h).1
  have c102369 := (publicBatchWrapper89_copies3199 a h).2.1
  have c102370 := (publicBatchWrapper89_copies3199 a h).2.2.1
  have c102371 := (publicBatchWrapper89_copies3199 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1672_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_18
  simp only [← c102366, ← c102367, ← c102368] at e_1672_18
  have e_1672_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1672 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_19
  simp only [← c102369, ← c102370, k2, ← c102371] at e_1672_19
  have hr := e_1672_19
  simp only [e_1672_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1673 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16615)) := by
  have c102372 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.1
  have c102373 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.1
  have c102374 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.1
  have c102375 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.1
  have c102376 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.1
  have c102377 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1673_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_0
  simp only [← c102372, ← c102373, ← c102374] at e_1673_0
  have e_1673_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_1
  simp only [← c102375, ← c102376, k2, ← c102377] at e_1673_1
  have hr := e_1673_1
  simp only [e_1673_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1673 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16616)) := by
  have c102378 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102379 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102380 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102381 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102382 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102383 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1673_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_2
  simp only [← c102378, ← c102379, ← c102380] at e_1673_2
  have e_1673_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_3
  simp only [← c102381, ← c102382, k2, ← c102383] at e_1673_3
  have hr := e_1673_3
  simp only [e_1673_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1673 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16617)) := by
  have c102384 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102385 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102386 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102387 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102388 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102389 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1673_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_4
  simp only [← c102384, ← c102385, ← c102386] at e_1673_4
  have e_1673_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_5
  simp only [← c102387, ← c102388, k2, ← c102389] at e_1673_5
  have hr := e_1673_5
  simp only [e_1673_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1673 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16618)) := by
  have c102390 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102391 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102392 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102393 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102394 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102395 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1673_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_6
  simp only [← c102390, ← c102391, ← c102392] at e_1673_6
  have e_1673_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_7
  simp only [← c102393, ← c102394, k2, ← c102395] at e_1673_7
  have hr := e_1673_7
  simp only [e_1673_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1673 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16619)) := by
  have c102396 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102397 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102398 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102399 := (publicBatchWrapper89_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102400 := (publicBatchWrapper89_copies3200 a h).1
  have c102401 := (publicBatchWrapper89_copies3200 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1673_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_8
  simp only [← c102396, ← c102397, ← c102398] at e_1673_8
  have e_1673_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_9
  simp only [← c102399, ← c102400, k2, ← c102401] at e_1673_9
  have hr := e_1673_9
  simp only [e_1673_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1673 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16620)) := by
  have c102402 := (publicBatchWrapper89_copies3200 a h).2.2.1
  have c102403 := (publicBatchWrapper89_copies3200 a h).2.2.2.1
  have c102404 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.1
  have c102405 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.1
  have c102406 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.1
  have c102407 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1673_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_10
  simp only [← c102402, ← c102403, ← c102404] at e_1673_10
  have e_1673_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_11
  simp only [← c102405, ← c102406, k2, ← c102407] at e_1673_11
  have hr := e_1673_11
  simp only [e_1673_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1673 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16621)) := by
  have c102408 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.1
  have c102409 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.1
  have c102410 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102411 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102412 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102413 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1673_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_12
  simp only [← c102408, ← c102409, ← c102410] at e_1673_12
  have e_1673_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_13
  simp only [← c102411, ← c102412, k2, ← c102413] at e_1673_13
  have hr := e_1673_13
  simp only [e_1673_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1673 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16622)) := by
  have c102414 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102415 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102416 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102417 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102418 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102419 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1673_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_14
  simp only [← c102414, ← c102415, ← c102416] at e_1673_14
  have e_1673_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_15
  simp only [← c102417, ← c102418, k2, ← c102419] at e_1673_15
  have hr := e_1673_15
  simp only [e_1673_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1673 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16623)) := by
  have c102420 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102421 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102422 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102423 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102424 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102425 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1673_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_16
  simp only [← c102420, ← c102421, ← c102422] at e_1673_16
  have e_1673_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_17
  simp only [← c102423, ← c102424, k2, ← c102425] at e_1673_17
  have hr := e_1673_17
  simp only [e_1673_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1673 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16624)) := by
  have c102426 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102427 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102428 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102429 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102430 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102431 := (publicBatchWrapper89_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1673_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_18
  simp only [← c102426, ← c102427, ← c102428] at e_1673_18
  have e_1673_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1673 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_19
  simp only [← c102429, ← c102430, k2, ← c102431] at e_1673_19
  have hr := e_1673_19
  simp only [e_1673_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1674 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16625)) := by
  have c102432 := (publicBatchWrapper89_copies3201 a h).1
  have c102433 := (publicBatchWrapper89_copies3201 a h).2.1
  have c102434 := (publicBatchWrapper89_copies3201 a h).2.2.1
  have c102435 := (publicBatchWrapper89_copies3201 a h).2.2.2.1
  have c102436 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.1
  have c102437 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1674_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_0
  simp only [← c102432, ← c102433, ← c102434] at e_1674_0
  have e_1674_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_1
  simp only [← c102435, ← c102436, k2, ← c102437] at e_1674_1
  have hr := e_1674_1
  simp only [e_1674_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1674 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16626)) := by
  have c102438 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.1
  have c102439 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.1
  have c102440 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.1
  have c102441 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.1
  have c102442 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102443 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1674_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_2
  simp only [← c102438, ← c102439, ← c102440] at e_1674_2
  have e_1674_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_3
  simp only [← c102441, ← c102442, k2, ← c102443] at e_1674_3
  have hr := e_1674_3
  simp only [e_1674_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1674 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16627)) := by
  have c102444 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102445 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102446 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102447 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102448 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102449 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1674_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_4
  simp only [← c102444, ← c102445, ← c102446] at e_1674_4
  have e_1674_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_5
  simp only [← c102447, ← c102448, k2, ← c102449] at e_1674_5
  have hr := e_1674_5
  simp only [e_1674_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1674 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16628)) := by
  have c102450 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102451 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102452 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102453 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102454 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102455 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1674_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_6
  simp only [← c102450, ← c102451, ← c102452] at e_1674_6
  have e_1674_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_7
  simp only [← c102453, ← c102454, k2, ← c102455] at e_1674_7
  have hr := e_1674_7
  simp only [e_1674_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1674 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16629)) := by
  have c102456 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102457 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102458 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102459 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102460 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102461 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1674_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_8
  simp only [← c102456, ← c102457, ← c102458] at e_1674_8
  have e_1674_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_9
  simp only [← c102459, ← c102460, k2, ← c102461] at e_1674_9
  have hr := e_1674_9
  simp only [e_1674_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1674 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16630)) := by
  have c102462 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102463 := (publicBatchWrapper89_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102464 := (publicBatchWrapper89_copies3202 a h).1
  have c102465 := (publicBatchWrapper89_copies3202 a h).2.1
  have c102466 := (publicBatchWrapper89_copies3202 a h).2.2.1
  have c102467 := (publicBatchWrapper89_copies3202 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1674_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_10
  simp only [← c102462, ← c102463, ← c102464] at e_1674_10
  have e_1674_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_11
  simp only [← c102465, ← c102466, k2, ← c102467] at e_1674_11
  have hr := e_1674_11
  simp only [e_1674_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1674 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16631)) := by
  have c102468 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.1
  have c102469 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.1
  have c102470 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.1
  have c102471 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.1
  have c102472 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.1
  have c102473 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1674_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_12
  simp only [← c102468, ← c102469, ← c102470] at e_1674_12
  have e_1674_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_13
  simp only [← c102471, ← c102472, k2, ← c102473] at e_1674_13
  have hr := e_1674_13
  simp only [e_1674_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1674 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16632)) := by
  have c102474 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102475 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102476 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102477 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102478 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102479 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1674_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_14
  simp only [← c102474, ← c102475, ← c102476] at e_1674_14
  have e_1674_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_15
  simp only [← c102477, ← c102478, k2, ← c102479] at e_1674_15
  have hr := e_1674_15
  simp only [e_1674_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1674 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16633)) := by
  have c102480 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102481 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102482 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102483 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102484 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102485 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1674_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_16
  simp only [← c102480, ← c102481, ← c102482] at e_1674_16
  have e_1674_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_17
  simp only [← c102483, ← c102484, k2, ← c102485] at e_1674_17
  have hr := e_1674_17
  simp only [e_1674_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1674 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16634)) := by
  have c102486 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102487 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102488 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102489 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102490 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102491 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1674_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_18
  simp only [← c102486, ← c102487, ← c102488] at e_1674_18
  have e_1674_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1674 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_19
  simp only [← c102489, ← c102490, k2, ← c102491] at e_1674_19
  have hr := e_1674_19
  simp only [e_1674_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1675 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16635)) := by
  have c102492 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102493 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102494 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102495 := (publicBatchWrapper89_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102496 := (publicBatchWrapper89_copies3203 a h).1
  have c102497 := (publicBatchWrapper89_copies3203 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1675_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_0
  simp only [← c102492, ← c102493, ← c102494] at e_1675_0
  have e_1675_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_1
  simp only [← c102495, ← c102496, k2, ← c102497] at e_1675_1
  have hr := e_1675_1
  simp only [e_1675_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1675 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16636)) := by
  have c102498 := (publicBatchWrapper89_copies3203 a h).2.2.1
  have c102499 := (publicBatchWrapper89_copies3203 a h).2.2.2.1
  have c102500 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.1
  have c102501 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.1
  have c102502 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.1
  have c102503 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1675_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_2
  simp only [← c102498, ← c102499, ← c102500] at e_1675_2
  have e_1675_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_3
  simp only [← c102501, ← c102502, k2, ← c102503] at e_1675_3
  have hr := e_1675_3
  simp only [e_1675_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1675 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16637)) := by
  have c102504 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.1
  have c102505 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.1
  have c102506 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102507 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102508 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102509 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1675_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_4
  simp only [← c102504, ← c102505, ← c102506] at e_1675_4
  have e_1675_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_5
  simp only [← c102507, ← c102508, k2, ← c102509] at e_1675_5
  have hr := e_1675_5
  simp only [e_1675_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1675 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16638)) := by
  have c102510 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102511 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102512 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102513 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102514 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102515 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1675_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_6
  simp only [← c102510, ← c102511, ← c102512] at e_1675_6
  have e_1675_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_7
  simp only [← c102513, ← c102514, k2, ← c102515] at e_1675_7
  have hr := e_1675_7
  simp only [e_1675_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1675 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16639)) := by
  have c102516 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102517 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102518 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102519 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102520 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102521 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1675_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_8
  simp only [← c102516, ← c102517, ← c102518] at e_1675_8
  have e_1675_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_9
  simp only [← c102519, ← c102520, k2, ← c102521] at e_1675_9
  have hr := e_1675_9
  simp only [e_1675_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1675 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16640)) := by
  have c102522 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102523 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102524 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102525 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102526 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102527 := (publicBatchWrapper89_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1675_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_10
  simp only [← c102522, ← c102523, ← c102524] at e_1675_10
  have e_1675_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_11
  simp only [← c102525, ← c102526, k2, ← c102527] at e_1675_11
  have hr := e_1675_11
  simp only [e_1675_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1675 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16641)) := by
  have c102528 := (publicBatchWrapper89_copies3204 a h).1
  have c102529 := (publicBatchWrapper89_copies3204 a h).2.1
  have c102530 := (publicBatchWrapper89_copies3204 a h).2.2.1
  have c102531 := (publicBatchWrapper89_copies3204 a h).2.2.2.1
  have c102532 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.1
  have c102533 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1675_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_12
  simp only [← c102528, ← c102529, ← c102530] at e_1675_12
  have e_1675_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_13
  simp only [← c102531, ← c102532, k2, ← c102533] at e_1675_13
  have hr := e_1675_13
  simp only [e_1675_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1675 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16642)) := by
  have c102534 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.1
  have c102535 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.1
  have c102536 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.1
  have c102537 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.1
  have c102538 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102539 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1675_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_14
  simp only [← c102534, ← c102535, ← c102536] at e_1675_14
  have e_1675_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_15
  simp only [← c102537, ← c102538, k2, ← c102539] at e_1675_15
  have hr := e_1675_15
  simp only [e_1675_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1675 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16643)) := by
  have c102540 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102541 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102542 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102543 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102544 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102545 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1675_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_16
  simp only [← c102540, ← c102541, ← c102542] at e_1675_16
  have e_1675_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_17
  simp only [← c102543, ← c102544, k2, ← c102545] at e_1675_17
  have hr := e_1675_17
  simp only [e_1675_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1675 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16644)) := by
  have c102546 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102547 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102548 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102549 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102550 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102551 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1675_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_18
  simp only [← c102546, ← c102547, ← c102548] at e_1675_18
  have e_1675_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1675 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_19
  simp only [← c102549, ← c102550, k2, ← c102551] at e_1675_19
  have hr := e_1675_19
  simp only [e_1675_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1676 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16645)) := by
  have c102552 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102553 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102554 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102555 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102556 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102557 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1676_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_0
  simp only [← c102552, ← c102553, ← c102554] at e_1676_0
  have e_1676_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_1
  simp only [← c102555, ← c102556, k2, ← c102557] at e_1676_1
  have hr := e_1676_1
  simp only [e_1676_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1676 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16646)) := by
  have c102558 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102559 := (publicBatchWrapper89_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102560 := (publicBatchWrapper89_copies3205 a h).1
  have c102561 := (publicBatchWrapper89_copies3205 a h).2.1
  have c102562 := (publicBatchWrapper89_copies3205 a h).2.2.1
  have c102563 := (publicBatchWrapper89_copies3205 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1676_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_2
  simp only [← c102558, ← c102559, ← c102560] at e_1676_2
  have e_1676_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_3
  simp only [← c102561, ← c102562, k2, ← c102563] at e_1676_3
  have hr := e_1676_3
  simp only [e_1676_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1676 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16647)) := by
  have c102564 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.1
  have c102565 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.1
  have c102566 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.1
  have c102567 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.1
  have c102568 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.1
  have c102569 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1676_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_4
  simp only [← c102564, ← c102565, ← c102566] at e_1676_4
  have e_1676_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_5
  simp only [← c102567, ← c102568, k2, ← c102569] at e_1676_5
  have hr := e_1676_5
  simp only [e_1676_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1676 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16648)) := by
  have c102570 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102571 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102572 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102573 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102574 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102575 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1676_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_6
  simp only [← c102570, ← c102571, ← c102572] at e_1676_6
  have e_1676_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_7
  simp only [← c102573, ← c102574, k2, ← c102575] at e_1676_7
  have hr := e_1676_7
  simp only [e_1676_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1676 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16649)) := by
  have c102576 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102577 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102578 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102579 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102580 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102581 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1676_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_8
  simp only [← c102576, ← c102577, ← c102578] at e_1676_8
  have e_1676_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_9
  simp only [← c102579, ← c102580, k2, ← c102581] at e_1676_9
  have hr := e_1676_9
  simp only [e_1676_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1676 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16650)) := by
  have c102582 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102583 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102584 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102585 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102586 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102587 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1676_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_10
  simp only [← c102582, ← c102583, ← c102584] at e_1676_10
  have e_1676_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_11
  simp only [← c102585, ← c102586, k2, ← c102587] at e_1676_11
  have hr := e_1676_11
  simp only [e_1676_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1676 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16651)) := by
  have c102588 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102589 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102590 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102591 := (publicBatchWrapper89_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102592 := (publicBatchWrapper89_copies3206 a h).1
  have c102593 := (publicBatchWrapper89_copies3206 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1676_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_12
  simp only [← c102588, ← c102589, ← c102590] at e_1676_12
  have e_1676_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_13
  simp only [← c102591, ← c102592, k2, ← c102593] at e_1676_13
  have hr := e_1676_13
  simp only [e_1676_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1676 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16652)) := by
  have c102594 := (publicBatchWrapper89_copies3206 a h).2.2.1
  have c102595 := (publicBatchWrapper89_copies3206 a h).2.2.2.1
  have c102596 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.1
  have c102597 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.1
  have c102598 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.1
  have c102599 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1676_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_14
  simp only [← c102594, ← c102595, ← c102596] at e_1676_14
  have e_1676_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_15
  simp only [← c102597, ← c102598, k2, ← c102599] at e_1676_15
  have hr := e_1676_15
  simp only [e_1676_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1676 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16653)) := by
  have c102600 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.1
  have c102601 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.1
  have c102602 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102603 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102604 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102605 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1676_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_16
  simp only [← c102600, ← c102601, ← c102602] at e_1676_16
  have e_1676_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_17
  simp only [← c102603, ← c102604, k2, ← c102605] at e_1676_17
  have hr := e_1676_17
  simp only [e_1676_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1676 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16654)) := by
  have c102606 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102607 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102608 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102609 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102610 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102611 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1676_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_18
  simp only [← c102606, ← c102607, ← c102608] at e_1676_18
  have e_1676_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1676 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_19
  simp only [← c102609, ← c102610, k2, ← c102611] at e_1676_19
  have hr := e_1676_19
  simp only [e_1676_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1677 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16655)) := by
  have c102612 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102613 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102614 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102615 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102616 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102617 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1677_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_0
  simp only [← c102612, ← c102613, ← c102614] at e_1677_0
  have e_1677_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_1
  simp only [← c102615, ← c102616, k2, ← c102617] at e_1677_1
  have hr := e_1677_1
  simp only [e_1677_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1677 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16656)) := by
  have c102618 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102619 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102620 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102621 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102622 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102623 := (publicBatchWrapper89_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1677_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_2
  simp only [← c102618, ← c102619, ← c102620] at e_1677_2
  have e_1677_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_3
  simp only [← c102621, ← c102622, k2, ← c102623] at e_1677_3
  have hr := e_1677_3
  simp only [e_1677_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1677 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16657)) := by
  have c102624 := (publicBatchWrapper89_copies3207 a h).1
  have c102625 := (publicBatchWrapper89_copies3207 a h).2.1
  have c102626 := (publicBatchWrapper89_copies3207 a h).2.2.1
  have c102627 := (publicBatchWrapper89_copies3207 a h).2.2.2.1
  have c102628 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.1
  have c102629 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1677_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_4
  simp only [← c102624, ← c102625, ← c102626] at e_1677_4
  have e_1677_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_5
  simp only [← c102627, ← c102628, k2, ← c102629] at e_1677_5
  have hr := e_1677_5
  simp only [e_1677_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1677 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16658)) := by
  have c102630 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.1
  have c102631 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.1
  have c102632 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.1
  have c102633 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.1
  have c102634 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102635 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1677_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_6
  simp only [← c102630, ← c102631, ← c102632] at e_1677_6
  have e_1677_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_7
  simp only [← c102633, ← c102634, k2, ← c102635] at e_1677_7
  have hr := e_1677_7
  simp only [e_1677_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1677 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16659)) := by
  have c102636 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102637 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102638 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102639 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102640 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102641 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1677_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_8
  simp only [← c102636, ← c102637, ← c102638] at e_1677_8
  have e_1677_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_9
  simp only [← c102639, ← c102640, k2, ← c102641] at e_1677_9
  have hr := e_1677_9
  simp only [e_1677_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1677 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16660)) := by
  have c102642 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102643 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102644 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102645 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102646 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102647 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1677_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_10
  simp only [← c102642, ← c102643, ← c102644] at e_1677_10
  have e_1677_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_11
  simp only [← c102645, ← c102646, k2, ← c102647] at e_1677_11
  have hr := e_1677_11
  simp only [e_1677_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1677 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16661)) := by
  have c102648 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102649 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102650 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102651 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102652 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102653 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1677_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_12
  simp only [← c102648, ← c102649, ← c102650] at e_1677_12
  have e_1677_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_13
  simp only [← c102651, ← c102652, k2, ← c102653] at e_1677_13
  have hr := e_1677_13
  simp only [e_1677_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1677 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16662)) := by
  have c102654 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102655 := (publicBatchWrapper89_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102656 := (publicBatchWrapper89_copies3208 a h).1
  have c102657 := (publicBatchWrapper89_copies3208 a h).2.1
  have c102658 := (publicBatchWrapper89_copies3208 a h).2.2.1
  have c102659 := (publicBatchWrapper89_copies3208 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1677_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_14
  simp only [← c102654, ← c102655, ← c102656] at e_1677_14
  have e_1677_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_15
  simp only [← c102657, ← c102658, k2, ← c102659] at e_1677_15
  have hr := e_1677_15
  simp only [e_1677_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1677 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16663)) := by
  have c102660 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.1
  have c102661 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.1
  have c102662 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.1
  have c102663 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.1
  have c102664 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.1
  have c102665 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1677_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_16
  simp only [← c102660, ← c102661, ← c102662] at e_1677_16
  have e_1677_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_17
  simp only [← c102663, ← c102664, k2, ← c102665] at e_1677_17
  have hr := e_1677_17
  simp only [e_1677_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1677 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16664)) := by
  have c102666 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102667 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102668 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102669 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102670 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102671 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1677_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_18
  simp only [← c102666, ← c102667, ← c102668] at e_1677_18
  have e_1677_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1677 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_19
  simp only [← c102669, ← c102670, k2, ← c102671] at e_1677_19
  have hr := e_1677_19
  simp only [e_1677_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1678 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16665)) := by
  have c102672 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102673 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102674 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102675 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102676 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102677 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1678_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_0
  simp only [← c102672, ← c102673, ← c102674] at e_1678_0
  have e_1678_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_1
  simp only [← c102675, ← c102676, k2, ← c102677] at e_1678_1
  have hr := e_1678_1
  simp only [e_1678_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1678 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16666)) := by
  have c102678 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102679 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102680 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102681 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102682 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102683 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1678_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_2
  simp only [← c102678, ← c102679, ← c102680] at e_1678_2
  have e_1678_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_3
  simp only [← c102681, ← c102682, k2, ← c102683] at e_1678_3
  have hr := e_1678_3
  simp only [e_1678_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1678 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16667)) := by
  have c102684 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102685 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102686 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102687 := (publicBatchWrapper89_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102688 := (publicBatchWrapper89_copies3209 a h).1
  have c102689 := (publicBatchWrapper89_copies3209 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1678_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_4
  simp only [← c102684, ← c102685, ← c102686] at e_1678_4
  have e_1678_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_5
  simp only [← c102687, ← c102688, k2, ← c102689] at e_1678_5
  have hr := e_1678_5
  simp only [e_1678_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1678 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16668)) := by
  have c102690 := (publicBatchWrapper89_copies3209 a h).2.2.1
  have c102691 := (publicBatchWrapper89_copies3209 a h).2.2.2.1
  have c102692 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.1
  have c102693 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.1
  have c102694 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.1
  have c102695 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1678_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_6
  simp only [← c102690, ← c102691, ← c102692] at e_1678_6
  have e_1678_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_7
  simp only [← c102693, ← c102694, k2, ← c102695] at e_1678_7
  have hr := e_1678_7
  simp only [e_1678_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1678 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16669)) := by
  have c102696 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.1
  have c102697 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.1
  have c102698 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102699 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102700 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102701 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1678_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_8
  simp only [← c102696, ← c102697, ← c102698] at e_1678_8
  have e_1678_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_9
  simp only [← c102699, ← c102700, k2, ← c102701] at e_1678_9
  have hr := e_1678_9
  simp only [e_1678_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1678 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16670)) := by
  have c102702 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102703 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102704 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102705 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102706 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102707 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1678_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_10
  simp only [← c102702, ← c102703, ← c102704] at e_1678_10
  have e_1678_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_11
  simp only [← c102705, ← c102706, k2, ← c102707] at e_1678_11
  have hr := e_1678_11
  simp only [e_1678_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1678 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16671)) := by
  have c102708 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102709 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102710 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102711 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102712 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102713 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1678_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_12
  simp only [← c102708, ← c102709, ← c102710] at e_1678_12
  have e_1678_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_13
  simp only [← c102711, ← c102712, k2, ← c102713] at e_1678_13
  have hr := e_1678_13
  simp only [e_1678_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1678 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16672)) := by
  have c102714 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102715 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102716 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102717 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102718 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102719 := (publicBatchWrapper89_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1678_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_14
  simp only [← c102714, ← c102715, ← c102716] at e_1678_14
  have e_1678_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_15
  simp only [← c102717, ← c102718, k2, ← c102719] at e_1678_15
  have hr := e_1678_15
  simp only [e_1678_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1678 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16673)) := by
  have c102720 := (publicBatchWrapper89_copies3210 a h).1
  have c102721 := (publicBatchWrapper89_copies3210 a h).2.1
  have c102722 := (publicBatchWrapper89_copies3210 a h).2.2.1
  have c102723 := (publicBatchWrapper89_copies3210 a h).2.2.2.1
  have c102724 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.1
  have c102725 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1678_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_16
  simp only [← c102720, ← c102721, ← c102722] at e_1678_16
  have e_1678_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_17
  simp only [← c102723, ← c102724, k2, ← c102725] at e_1678_17
  have hr := e_1678_17
  simp only [e_1678_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1678 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16674)) := by
  have c102726 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.1
  have c102727 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.1
  have c102728 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.1
  have c102729 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.1
  have c102730 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102731 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1678_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_18
  simp only [← c102726, ← c102727, ← c102728] at e_1678_18
  have e_1678_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1678 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_19
  simp only [← c102729, ← c102730, k2, ← c102731] at e_1678_19
  have hr := e_1678_19
  simp only [e_1678_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1679 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16675)) := by
  have c102732 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102733 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102734 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102735 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102736 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102737 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1679_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_0
  simp only [← c102732, ← c102733, ← c102734] at e_1679_0
  have e_1679_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_1
  simp only [← c102735, ← c102736, k2, ← c102737] at e_1679_1
  have hr := e_1679_1
  simp only [e_1679_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1679 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16676)) := by
  have c102738 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102739 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102740 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102741 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102742 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102743 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1679_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_2
  simp only [← c102738, ← c102739, ← c102740] at e_1679_2
  have e_1679_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_3
  simp only [← c102741, ← c102742, k2, ← c102743] at e_1679_3
  have hr := e_1679_3
  simp only [e_1679_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1679 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16677)) := by
  have c102744 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102745 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102746 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102747 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102748 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102749 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1679_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_4
  simp only [← c102744, ← c102745, ← c102746] at e_1679_4
  have e_1679_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_5
  simp only [← c102747, ← c102748, k2, ← c102749] at e_1679_5
  have hr := e_1679_5
  simp only [e_1679_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1679 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16678)) := by
  have c102750 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102751 := (publicBatchWrapper89_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102752 := (publicBatchWrapper89_copies3211 a h).1
  have c102753 := (publicBatchWrapper89_copies3211 a h).2.1
  have c102754 := (publicBatchWrapper89_copies3211 a h).2.2.1
  have c102755 := (publicBatchWrapper89_copies3211 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1679_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_6
  simp only [← c102750, ← c102751, ← c102752] at e_1679_6
  have e_1679_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_7
  simp only [← c102753, ← c102754, k2, ← c102755] at e_1679_7
  have hr := e_1679_7
  simp only [e_1679_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1679 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16679)) := by
  have c102756 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.1
  have c102757 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.1
  have c102758 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.1
  have c102759 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.1
  have c102760 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.1
  have c102761 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1679_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_8
  simp only [← c102756, ← c102757, ← c102758] at e_1679_8
  have e_1679_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_9
  simp only [← c102759, ← c102760, k2, ← c102761] at e_1679_9
  have hr := e_1679_9
  simp only [e_1679_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1679 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16680)) := by
  have c102762 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102763 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102764 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102765 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102766 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102767 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1679_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_10
  simp only [← c102762, ← c102763, ← c102764] at e_1679_10
  have e_1679_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_11
  simp only [← c102765, ← c102766, k2, ← c102767] at e_1679_11
  have hr := e_1679_11
  simp only [e_1679_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1679 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16681)) := by
  have c102768 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102769 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102770 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102771 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102772 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102773 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1679_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_12
  simp only [← c102768, ← c102769, ← c102770] at e_1679_12
  have e_1679_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_13
  simp only [← c102771, ← c102772, k2, ← c102773] at e_1679_13
  have hr := e_1679_13
  simp only [e_1679_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1679 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16682)) := by
  have c102774 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102775 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102776 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102777 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102778 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102779 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1679_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_14
  simp only [← c102774, ← c102775, ← c102776] at e_1679_14
  have e_1679_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_15
  simp only [← c102777, ← c102778, k2, ← c102779] at e_1679_15
  have hr := e_1679_15
  simp only [e_1679_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1679 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16683)) := by
  have c102780 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102781 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102782 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102783 := (publicBatchWrapper89_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102784 := (publicBatchWrapper89_copies3212 a h).1
  have c102785 := (publicBatchWrapper89_copies3212 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1679_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_16
  simp only [← c102780, ← c102781, ← c102782] at e_1679_16
  have e_1679_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_17
  simp only [← c102783, ← c102784, k2, ← c102785] at e_1679_17
  have hr := e_1679_17
  simp only [e_1679_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1679 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16684)) := by
  have c102786 := (publicBatchWrapper89_copies3212 a h).2.2.1
  have c102787 := (publicBatchWrapper89_copies3212 a h).2.2.2.1
  have c102788 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.1
  have c102789 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.1
  have c102790 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.1
  have c102791 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1679_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_18
  simp only [← c102786, ← c102787, ← c102788] at e_1679_18
  have e_1679_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1679 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_19
  simp only [← c102789, ← c102790, k2, ← c102791] at e_1679_19
  have hr := e_1679_19
  simp only [e_1679_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1680 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16685)) := by
  have c102792 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.1
  have c102793 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.1
  have c102794 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102795 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102796 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102797 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1680_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_0
  simp only [← c102792, ← c102793, ← c102794] at e_1680_0
  have e_1680_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_1
  simp only [← c102795, ← c102796, k2, ← c102797] at e_1680_1
  have hr := e_1680_1
  simp only [e_1680_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1680 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16686)) := by
  have c102798 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102799 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102800 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102801 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102802 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102803 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1680_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_2
  simp only [← c102798, ← c102799, ← c102800] at e_1680_2
  have e_1680_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_3
  simp only [← c102801, ← c102802, k2, ← c102803] at e_1680_3
  have hr := e_1680_3
  simp only [e_1680_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1680 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16687)) := by
  have c102804 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102805 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102806 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102807 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102808 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102809 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1680_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_4
  simp only [← c102804, ← c102805, ← c102806] at e_1680_4
  have e_1680_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_5
  simp only [← c102807, ← c102808, k2, ← c102809] at e_1680_5
  have hr := e_1680_5
  simp only [e_1680_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1680 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16688)) := by
  have c102810 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102811 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102812 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102813 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102814 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102815 := (publicBatchWrapper89_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1680_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_6
  simp only [← c102810, ← c102811, ← c102812] at e_1680_6
  have e_1680_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_7
  simp only [← c102813, ← c102814, k2, ← c102815] at e_1680_7
  have hr := e_1680_7
  simp only [e_1680_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1680 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16689)) := by
  have c102816 := (publicBatchWrapper89_copies3213 a h).1
  have c102817 := (publicBatchWrapper89_copies3213 a h).2.1
  have c102818 := (publicBatchWrapper89_copies3213 a h).2.2.1
  have c102819 := (publicBatchWrapper89_copies3213 a h).2.2.2.1
  have c102820 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.1
  have c102821 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1680_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_8
  simp only [← c102816, ← c102817, ← c102818] at e_1680_8
  have e_1680_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_9
  simp only [← c102819, ← c102820, k2, ← c102821] at e_1680_9
  have hr := e_1680_9
  simp only [e_1680_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1680 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16690)) := by
  have c102822 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.1
  have c102823 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.1
  have c102824 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.1
  have c102825 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.1
  have c102826 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102827 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1680_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_10
  simp only [← c102822, ← c102823, ← c102824] at e_1680_10
  have e_1680_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_11
  simp only [← c102825, ← c102826, k2, ← c102827] at e_1680_11
  have hr := e_1680_11
  simp only [e_1680_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1680 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16691)) := by
  have c102828 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102829 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102830 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102831 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102832 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102833 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1680_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_12
  simp only [← c102828, ← c102829, ← c102830] at e_1680_12
  have e_1680_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_13
  simp only [← c102831, ← c102832, k2, ← c102833] at e_1680_13
  have hr := e_1680_13
  simp only [e_1680_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1680 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16692)) := by
  have c102834 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102835 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102836 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102837 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102838 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102839 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1680_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_14
  simp only [← c102834, ← c102835, ← c102836] at e_1680_14
  have e_1680_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_15
  simp only [← c102837, ← c102838, k2, ← c102839] at e_1680_15
  have hr := e_1680_15
  simp only [e_1680_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1680 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16693)) := by
  have c102840 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102841 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102842 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102843 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102844 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102845 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1680_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_16
  simp only [← c102840, ← c102841, ← c102842] at e_1680_16
  have e_1680_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_17
  simp only [← c102843, ← c102844, k2, ← c102845] at e_1680_17
  have hr := e_1680_17
  simp only [e_1680_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1680 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16694)) := by
  have c102846 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102847 := (publicBatchWrapper89_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102848 := (publicBatchWrapper89_copies3214 a h).1
  have c102849 := (publicBatchWrapper89_copies3214 a h).2.1
  have c102850 := (publicBatchWrapper89_copies3214 a h).2.2.1
  have c102851 := (publicBatchWrapper89_copies3214 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1680_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_18
  simp only [← c102846, ← c102847, ← c102848] at e_1680_18
  have e_1680_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1680 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_19
  simp only [← c102849, ← c102850, k2, ← c102851] at e_1680_19
  have hr := e_1680_19
  simp only [e_1680_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1681 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16695)) := by
  have c102852 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.1
  have c102853 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.1
  have c102854 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.1
  have c102855 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.1
  have c102856 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.1
  have c102857 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1681_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_0
  simp only [← c102852, ← c102853, ← c102854] at e_1681_0
  have e_1681_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_1
  simp only [← c102855, ← c102856, k2, ← c102857] at e_1681_1
  have hr := e_1681_1
  simp only [e_1681_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1681 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16696)) := by
  have c102858 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102859 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102860 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102861 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102862 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102863 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1681_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_2
  simp only [← c102858, ← c102859, ← c102860] at e_1681_2
  have e_1681_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_3
  simp only [← c102861, ← c102862, k2, ← c102863] at e_1681_3
  have hr := e_1681_3
  simp only [e_1681_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1681 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16697)) := by
  have c102864 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102865 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102866 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102867 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102868 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102869 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1681_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_4
  simp only [← c102864, ← c102865, ← c102866] at e_1681_4
  have e_1681_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_5
  simp only [← c102867, ← c102868, k2, ← c102869] at e_1681_5
  have hr := e_1681_5
  simp only [e_1681_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1681 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16698)) := by
  have c102870 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102871 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102872 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102873 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102874 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102875 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1681_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_6
  simp only [← c102870, ← c102871, ← c102872] at e_1681_6
  have e_1681_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_7
  simp only [← c102873, ← c102874, k2, ← c102875] at e_1681_7
  have hr := e_1681_7
  simp only [e_1681_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1681 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16699)) := by
  have c102876 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102877 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102878 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102879 := (publicBatchWrapper89_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102880 := (publicBatchWrapper89_copies3215 a h).1
  have c102881 := (publicBatchWrapper89_copies3215 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1681_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_8
  simp only [← c102876, ← c102877, ← c102878] at e_1681_8
  have e_1681_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_9
  simp only [← c102879, ← c102880, k2, ← c102881] at e_1681_9
  have hr := e_1681_9
  simp only [e_1681_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1681 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16700)) := by
  have c102882 := (publicBatchWrapper89_copies3215 a h).2.2.1
  have c102883 := (publicBatchWrapper89_copies3215 a h).2.2.2.1
  have c102884 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.1
  have c102885 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.1
  have c102886 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.1
  have c102887 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1681_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_10
  simp only [← c102882, ← c102883, ← c102884] at e_1681_10
  have e_1681_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_11
  simp only [← c102885, ← c102886, k2, ← c102887] at e_1681_11
  have hr := e_1681_11
  simp only [e_1681_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1681 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16701)) := by
  have c102888 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.1
  have c102889 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.1
  have c102890 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102891 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102892 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102893 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1681_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_12
  simp only [← c102888, ← c102889, ← c102890] at e_1681_12
  have e_1681_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_13
  simp only [← c102891, ← c102892, k2, ← c102893] at e_1681_13
  have hr := e_1681_13
  simp only [e_1681_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1681 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16702)) := by
  have c102894 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102895 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102896 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102897 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102898 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102899 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1681_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_14
  simp only [← c102894, ← c102895, ← c102896] at e_1681_14
  have e_1681_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_15
  simp only [← c102897, ← c102898, k2, ← c102899] at e_1681_15
  have hr := e_1681_15
  simp only [e_1681_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1681 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16703)) := by
  have c102900 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102901 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102902 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102903 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102904 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102905 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1681_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_16
  simp only [← c102900, ← c102901, ← c102902] at e_1681_16
  have e_1681_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_17
  simp only [← c102903, ← c102904, k2, ← c102905] at e_1681_17
  have hr := e_1681_17
  simp only [e_1681_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1681 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16704)) := by
  have c102906 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102907 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102908 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102909 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102910 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102911 := (publicBatchWrapper89_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1681_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_18
  simp only [← c102906, ← c102907, ← c102908] at e_1681_18
  have e_1681_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1681 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_19
  simp only [← c102909, ← c102910, k2, ← c102911] at e_1681_19
  have hr := e_1681_19
  simp only [e_1681_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1682 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16705)) := by
  have c102912 := (publicBatchWrapper89_copies3216 a h).1
  have c102913 := (publicBatchWrapper89_copies3216 a h).2.1
  have c102914 := (publicBatchWrapper89_copies3216 a h).2.2.1
  have c102915 := (publicBatchWrapper89_copies3216 a h).2.2.2.1
  have c102916 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.1
  have c102917 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1682_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_0
  simp only [← c102912, ← c102913, ← c102914] at e_1682_0
  have e_1682_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_1
  simp only [← c102915, ← c102916, k2, ← c102917] at e_1682_1
  have hr := e_1682_1
  simp only [e_1682_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1682 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16706)) := by
  have c102918 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.1
  have c102919 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.1
  have c102920 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.1
  have c102921 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.1
  have c102922 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102923 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1682_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_2
  simp only [← c102918, ← c102919, ← c102920] at e_1682_2
  have e_1682_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_3
  simp only [← c102921, ← c102922, k2, ← c102923] at e_1682_3
  have hr := e_1682_3
  simp only [e_1682_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1682 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16707)) := by
  have c102924 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102925 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102926 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102927 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102928 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102929 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1682_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_4
  simp only [← c102924, ← c102925, ← c102926] at e_1682_4
  have e_1682_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_5
  simp only [← c102927, ← c102928, k2, ← c102929] at e_1682_5
  have hr := e_1682_5
  simp only [e_1682_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1682 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16708)) := by
  have c102930 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102931 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102932 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102933 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102934 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102935 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1682_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_6
  simp only [← c102930, ← c102931, ← c102932] at e_1682_6
  have e_1682_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_7
  simp only [← c102933, ← c102934, k2, ← c102935] at e_1682_7
  have hr := e_1682_7
  simp only [e_1682_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1682 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16709)) := by
  have c102936 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102937 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102938 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102939 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102940 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102941 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1682_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_8
  simp only [← c102936, ← c102937, ← c102938] at e_1682_8
  have e_1682_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_9
  simp only [← c102939, ← c102940, k2, ← c102941] at e_1682_9
  have hr := e_1682_9
  simp only [e_1682_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1682 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16710)) := by
  have c102942 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102943 := (publicBatchWrapper89_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102944 := (publicBatchWrapper89_copies3217 a h).1
  have c102945 := (publicBatchWrapper89_copies3217 a h).2.1
  have c102946 := (publicBatchWrapper89_copies3217 a h).2.2.1
  have c102947 := (publicBatchWrapper89_copies3217 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1682_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_10
  simp only [← c102942, ← c102943, ← c102944] at e_1682_10
  have e_1682_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_11
  simp only [← c102945, ← c102946, k2, ← c102947] at e_1682_11
  have hr := e_1682_11
  simp only [e_1682_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1682 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16711)) := by
  have c102948 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.1
  have c102949 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.1
  have c102950 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.1
  have c102951 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.1
  have c102952 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.1
  have c102953 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1682_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_12
  simp only [← c102948, ← c102949, ← c102950] at e_1682_12
  have e_1682_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_13
  simp only [← c102951, ← c102952, k2, ← c102953] at e_1682_13
  have hr := e_1682_13
  simp only [e_1682_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1682 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16712)) := by
  have c102954 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102955 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102956 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102957 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102958 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102959 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1682_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_14
  simp only [← c102954, ← c102955, ← c102956] at e_1682_14
  have e_1682_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_15
  simp only [← c102957, ← c102958, k2, ← c102959] at e_1682_15
  have hr := e_1682_15
  simp only [e_1682_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1682 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16713)) := by
  have c102960 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102961 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102962 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102963 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102964 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102965 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1682_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_16
  simp only [← c102960, ← c102961, ← c102962] at e_1682_16
  have e_1682_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_17
  simp only [← c102963, ← c102964, k2, ← c102965] at e_1682_17
  have hr := e_1682_17
  simp only [e_1682_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1682 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16714)) := by
  have c102966 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102967 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102968 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102969 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102970 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102971 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1682_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_18
  simp only [← c102966, ← c102967, ← c102968] at e_1682_18
  have e_1682_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1682 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_19
  simp only [← c102969, ← c102970, k2, ← c102971] at e_1682_19
  have hr := e_1682_19
  simp only [e_1682_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1683 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16715)) := by
  have c102972 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102973 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102974 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102975 := (publicBatchWrapper89_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102976 := (publicBatchWrapper89_copies3218 a h).1
  have c102977 := (publicBatchWrapper89_copies3218 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1683_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_0
  simp only [← c102972, ← c102973, ← c102974] at e_1683_0
  have e_1683_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_1
  simp only [← c102975, ← c102976, k2, ← c102977] at e_1683_1
  have hr := e_1683_1
  simp only [e_1683_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1683 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16716)) := by
  have c102978 := (publicBatchWrapper89_copies3218 a h).2.2.1
  have c102979 := (publicBatchWrapper89_copies3218 a h).2.2.2.1
  have c102980 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.1
  have c102981 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.1
  have c102982 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.1
  have c102983 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1683_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_2
  simp only [← c102978, ← c102979, ← c102980] at e_1683_2
  have e_1683_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_3
  simp only [← c102981, ← c102982, k2, ← c102983] at e_1683_3
  have hr := e_1683_3
  simp only [e_1683_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1683 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16717)) := by
  have c102984 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.1
  have c102985 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.1
  have c102986 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102987 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102988 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102989 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1683_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_4
  simp only [← c102984, ← c102985, ← c102986] at e_1683_4
  have e_1683_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_5
  simp only [← c102987, ← c102988, k2, ← c102989] at e_1683_5
  have hr := e_1683_5
  simp only [e_1683_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1683 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16718)) := by
  have c102990 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102991 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102992 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102993 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102994 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102995 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1683_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_6
  simp only [← c102990, ← c102991, ← c102992] at e_1683_6
  have e_1683_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_7
  simp only [← c102993, ← c102994, k2, ← c102995] at e_1683_7
  have hr := e_1683_7
  simp only [e_1683_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1683 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16719)) := by
  have c102996 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102997 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102998 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102999 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103000 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103001 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1683_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_8
  simp only [← c102996, ← c102997, ← c102998] at e_1683_8
  have e_1683_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_9
  simp only [← c102999, ← c103000, k2, ← c103001] at e_1683_9
  have hr := e_1683_9
  simp only [e_1683_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1683 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16720)) := by
  have c103002 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103003 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103004 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103005 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103006 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103007 := (publicBatchWrapper89_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1683_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_10
  simp only [← c103002, ← c103003, ← c103004] at e_1683_10
  have e_1683_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_11
  simp only [← c103005, ← c103006, k2, ← c103007] at e_1683_11
  have hr := e_1683_11
  simp only [e_1683_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1683 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16721)) := by
  have c103008 := (publicBatchWrapper89_copies3219 a h).1
  have c103009 := (publicBatchWrapper89_copies3219 a h).2.1
  have c103010 := (publicBatchWrapper89_copies3219 a h).2.2.1
  have c103011 := (publicBatchWrapper89_copies3219 a h).2.2.2.1
  have c103012 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.1
  have c103013 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1683_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_12
  simp only [← c103008, ← c103009, ← c103010] at e_1683_12
  have e_1683_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_13
  simp only [← c103011, ← c103012, k2, ← c103013] at e_1683_13
  have hr := e_1683_13
  simp only [e_1683_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1683 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16722)) := by
  have c103014 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.1
  have c103015 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.1
  have c103016 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.1
  have c103017 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.1
  have c103018 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103019 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1683_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_14
  simp only [← c103014, ← c103015, ← c103016] at e_1683_14
  have e_1683_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_15
  simp only [← c103017, ← c103018, k2, ← c103019] at e_1683_15
  have hr := e_1683_15
  simp only [e_1683_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1683 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16723)) := by
  have c103020 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103021 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103022 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103023 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103024 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103025 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1683_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_16
  simp only [← c103020, ← c103021, ← c103022] at e_1683_16
  have e_1683_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_17
  simp only [← c103023, ← c103024, k2, ← c103025] at e_1683_17
  have hr := e_1683_17
  simp only [e_1683_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1683 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16724)) := by
  have c103026 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103027 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103028 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103029 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103030 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103031 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1683_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_18
  simp only [← c103026, ← c103027, ← c103028] at e_1683_18
  have e_1683_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1683 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_19
  simp only [← c103029, ← c103030, k2, ← c103031] at e_1683_19
  have hr := e_1683_19
  simp only [e_1683_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1684 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16725)) := by
  have c103032 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103033 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103034 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103035 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103036 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103037 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1684_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_0
  simp only [← c103032, ← c103033, ← c103034] at e_1684_0
  have e_1684_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_1
  simp only [← c103035, ← c103036, k2, ← c103037] at e_1684_1
  have hr := e_1684_1
  simp only [e_1684_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1684 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16726)) := by
  have c103038 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103039 := (publicBatchWrapper89_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103040 := (publicBatchWrapper89_copies3220 a h).1
  have c103041 := (publicBatchWrapper89_copies3220 a h).2.1
  have c103042 := (publicBatchWrapper89_copies3220 a h).2.2.1
  have c103043 := (publicBatchWrapper89_copies3220 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1684_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_2
  simp only [← c103038, ← c103039, ← c103040] at e_1684_2
  have e_1684_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_3
  simp only [← c103041, ← c103042, k2, ← c103043] at e_1684_3
  have hr := e_1684_3
  simp only [e_1684_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1684 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16727)) := by
  have c103044 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.1
  have c103045 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.1
  have c103046 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.1
  have c103047 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.1
  have c103048 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.1
  have c103049 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1684_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_4
  simp only [← c103044, ← c103045, ← c103046] at e_1684_4
  have e_1684_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_5
  simp only [← c103047, ← c103048, k2, ← c103049] at e_1684_5
  have hr := e_1684_5
  simp only [e_1684_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1684 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16728)) := by
  have c103050 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103051 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103052 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103053 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103054 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103055 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1684_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_6
  simp only [← c103050, ← c103051, ← c103052] at e_1684_6
  have e_1684_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_7
  simp only [← c103053, ← c103054, k2, ← c103055] at e_1684_7
  have hr := e_1684_7
  simp only [e_1684_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1684 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16729)) := by
  have c103056 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103057 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103058 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103059 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103060 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103061 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1684_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_8
  simp only [← c103056, ← c103057, ← c103058] at e_1684_8
  have e_1684_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_9
  simp only [← c103059, ← c103060, k2, ← c103061] at e_1684_9
  have hr := e_1684_9
  simp only [e_1684_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1684 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16730)) := by
  have c103062 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103063 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103064 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103065 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103066 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103067 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1684_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_10
  simp only [← c103062, ← c103063, ← c103064] at e_1684_10
  have e_1684_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_11
  simp only [← c103065, ← c103066, k2, ← c103067] at e_1684_11
  have hr := e_1684_11
  simp only [e_1684_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1684 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16731)) := by
  have c103068 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103069 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103070 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103071 := (publicBatchWrapper89_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103072 := (publicBatchWrapper89_copies3221 a h).1
  have c103073 := (publicBatchWrapper89_copies3221 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1684_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_12
  simp only [← c103068, ← c103069, ← c103070] at e_1684_12
  have e_1684_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_13
  simp only [← c103071, ← c103072, k2, ← c103073] at e_1684_13
  have hr := e_1684_13
  simp only [e_1684_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1684 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16732)) := by
  have c103074 := (publicBatchWrapper89_copies3221 a h).2.2.1
  have c103075 := (publicBatchWrapper89_copies3221 a h).2.2.2.1
  have c103076 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.1
  have c103077 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.1
  have c103078 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.1
  have c103079 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1684_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_14
  simp only [← c103074, ← c103075, ← c103076] at e_1684_14
  have e_1684_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_15
  simp only [← c103077, ← c103078, k2, ← c103079] at e_1684_15
  have hr := e_1684_15
  simp only [e_1684_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1684 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16733)) := by
  have c103080 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.1
  have c103081 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.1
  have c103082 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103083 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103084 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103085 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1684_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_16
  simp only [← c103080, ← c103081, ← c103082] at e_1684_16
  have e_1684_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_17
  simp only [← c103083, ← c103084, k2, ← c103085] at e_1684_17
  have hr := e_1684_17
  simp only [e_1684_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1684 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16734)) := by
  have c103086 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103087 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103088 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103089 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103090 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103091 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1684_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_18
  simp only [← c103086, ← c103087, ← c103088] at e_1684_18
  have e_1684_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1684 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_19
  simp only [← c103089, ← c103090, k2, ← c103091] at e_1684_19
  have hr := e_1684_19
  simp only [e_1684_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1685 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16735)) := by
  have c103092 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103093 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103094 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103095 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103096 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103097 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1685_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_0
  simp only [← c103092, ← c103093, ← c103094] at e_1685_0
  have e_1685_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_1
  simp only [← c103095, ← c103096, k2, ← c103097] at e_1685_1
  have hr := e_1685_1
  simp only [e_1685_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1685 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16736)) := by
  have c103098 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103099 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103100 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103101 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103102 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103103 := (publicBatchWrapper89_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1685_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_2
  simp only [← c103098, ← c103099, ← c103100] at e_1685_2
  have e_1685_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_3
  simp only [← c103101, ← c103102, k2, ← c103103] at e_1685_3
  have hr := e_1685_3
  simp only [e_1685_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1685 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16737)) := by
  have c103104 := (publicBatchWrapper89_copies3222 a h).1
  have c103105 := (publicBatchWrapper89_copies3222 a h).2.1
  have c103106 := (publicBatchWrapper89_copies3222 a h).2.2.1
  have c103107 := (publicBatchWrapper89_copies3222 a h).2.2.2.1
  have c103108 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.1
  have c103109 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1685_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_4
  simp only [← c103104, ← c103105, ← c103106] at e_1685_4
  have e_1685_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_5
  simp only [← c103107, ← c103108, k2, ← c103109] at e_1685_5
  have hr := e_1685_5
  simp only [e_1685_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1685 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16738)) := by
  have c103110 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.1
  have c103111 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.1
  have c103112 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.1
  have c103113 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.1
  have c103114 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103115 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1685_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_6
  simp only [← c103110, ← c103111, ← c103112] at e_1685_6
  have e_1685_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_7
  simp only [← c103113, ← c103114, k2, ← c103115] at e_1685_7
  have hr := e_1685_7
  simp only [e_1685_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1685 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16739)) := by
  have c103116 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103117 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103118 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103119 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103120 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103121 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1685_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_8
  simp only [← c103116, ← c103117, ← c103118] at e_1685_8
  have e_1685_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_9
  simp only [← c103119, ← c103120, k2, ← c103121] at e_1685_9
  have hr := e_1685_9
  simp only [e_1685_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1685 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16740)) := by
  have c103122 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103123 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103124 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103125 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103126 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103127 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1685_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_10
  simp only [← c103122, ← c103123, ← c103124] at e_1685_10
  have e_1685_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_11
  simp only [← c103125, ← c103126, k2, ← c103127] at e_1685_11
  have hr := e_1685_11
  simp only [e_1685_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1685 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16741)) := by
  have c103128 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103129 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103130 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103131 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103132 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103133 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1685_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_12
  simp only [← c103128, ← c103129, ← c103130] at e_1685_12
  have e_1685_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_13
  simp only [← c103131, ← c103132, k2, ← c103133] at e_1685_13
  have hr := e_1685_13
  simp only [e_1685_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1685 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16742)) := by
  have c103134 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103135 := (publicBatchWrapper89_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103136 := (publicBatchWrapper89_copies3223 a h).1
  have c103137 := (publicBatchWrapper89_copies3223 a h).2.1
  have c103138 := (publicBatchWrapper89_copies3223 a h).2.2.1
  have c103139 := (publicBatchWrapper89_copies3223 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1685_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_14
  simp only [← c103134, ← c103135, ← c103136] at e_1685_14
  have e_1685_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_15
  simp only [← c103137, ← c103138, k2, ← c103139] at e_1685_15
  have hr := e_1685_15
  simp only [e_1685_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1685 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16743)) := by
  have c103140 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.1
  have c103141 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.1
  have c103142 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.1
  have c103143 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.1
  have c103144 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.1
  have c103145 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1685_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_16
  simp only [← c103140, ← c103141, ← c103142] at e_1685_16
  have e_1685_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_17
  simp only [← c103143, ← c103144, k2, ← c103145] at e_1685_17
  have hr := e_1685_17
  simp only [e_1685_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1685 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16744)) := by
  have c103146 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103147 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103148 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103149 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103150 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103151 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1685_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_18
  simp only [← c103146, ← c103147, ← c103148] at e_1685_18
  have e_1685_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1685 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_19
  simp only [← c103149, ← c103150, k2, ← c103151] at e_1685_19
  have hr := e_1685_19
  simp only [e_1685_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1686 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16745)) := by
  have c103152 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103153 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103154 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103155 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103156 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103157 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1686_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_0
  simp only [← c103152, ← c103153, ← c103154] at e_1686_0
  have e_1686_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_1
  simp only [← c103155, ← c103156, k2, ← c103157] at e_1686_1
  have hr := e_1686_1
  simp only [e_1686_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1686 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16746)) := by
  have c103158 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103159 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103160 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103161 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103162 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103163 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1686_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_2
  simp only [← c103158, ← c103159, ← c103160] at e_1686_2
  have e_1686_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_3
  simp only [← c103161, ← c103162, k2, ← c103163] at e_1686_3
  have hr := e_1686_3
  simp only [e_1686_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1686 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16747)) := by
  have c103164 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103165 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103166 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103167 := (publicBatchWrapper89_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103168 := (publicBatchWrapper89_copies3224 a h).1
  have c103169 := (publicBatchWrapper89_copies3224 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1686_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_4
  simp only [← c103164, ← c103165, ← c103166] at e_1686_4
  have e_1686_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_5
  simp only [← c103167, ← c103168, k2, ← c103169] at e_1686_5
  have hr := e_1686_5
  simp only [e_1686_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1686 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16748)) := by
  have c103170 := (publicBatchWrapper89_copies3224 a h).2.2.1
  have c103171 := (publicBatchWrapper89_copies3224 a h).2.2.2.1
  have c103172 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.1
  have c103173 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.1
  have c103174 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.1
  have c103175 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1686_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_6
  simp only [← c103170, ← c103171, ← c103172] at e_1686_6
  have e_1686_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_7
  simp only [← c103173, ← c103174, k2, ← c103175] at e_1686_7
  have hr := e_1686_7
  simp only [e_1686_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1686 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16749)) := by
  have c103176 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.1
  have c103177 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.1
  have c103178 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103179 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103180 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103181 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1686_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_8
  simp only [← c103176, ← c103177, ← c103178] at e_1686_8
  have e_1686_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_9
  simp only [← c103179, ← c103180, k2, ← c103181] at e_1686_9
  have hr := e_1686_9
  simp only [e_1686_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1686 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16750)) := by
  have c103182 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103183 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103184 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103185 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103186 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103187 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1686_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_10
  simp only [← c103182, ← c103183, ← c103184] at e_1686_10
  have e_1686_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_11
  simp only [← c103185, ← c103186, k2, ← c103187] at e_1686_11
  have hr := e_1686_11
  simp only [e_1686_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1686 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16751)) := by
  have c103188 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103189 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103190 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103191 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103192 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103193 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1686_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_12
  simp only [← c103188, ← c103189, ← c103190] at e_1686_12
  have e_1686_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_13
  simp only [← c103191, ← c103192, k2, ← c103193] at e_1686_13
  have hr := e_1686_13
  simp only [e_1686_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1686 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16752)) := by
  have c103194 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103195 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103196 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103197 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103198 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103199 := (publicBatchWrapper89_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1686_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_14
  simp only [← c103194, ← c103195, ← c103196] at e_1686_14
  have e_1686_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_15
  simp only [← c103197, ← c103198, k2, ← c103199] at e_1686_15
  have hr := e_1686_15
  simp only [e_1686_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1686 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16753)) := by
  have c103200 := (publicBatchWrapper89_copies3225 a h).1
  have c103201 := (publicBatchWrapper89_copies3225 a h).2.1
  have c103202 := (publicBatchWrapper89_copies3225 a h).2.2.1
  have c103203 := (publicBatchWrapper89_copies3225 a h).2.2.2.1
  have c103204 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.1
  have c103205 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1686_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_16
  simp only [← c103200, ← c103201, ← c103202] at e_1686_16
  have e_1686_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_17
  simp only [← c103203, ← c103204, k2, ← c103205] at e_1686_17
  have hr := e_1686_17
  simp only [e_1686_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1686 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16754)) := by
  have c103206 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.1
  have c103207 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.1
  have c103208 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.1
  have c103209 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.1
  have c103210 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103211 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1686_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_18
  simp only [← c103206, ← c103207, ← c103208] at e_1686_18
  have e_1686_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1686 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_19
  simp only [← c103209, ← c103210, k2, ← c103211] at e_1686_19
  have hr := e_1686_19
  simp only [e_1686_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1687 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16755)) := by
  have c103212 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103213 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103214 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103215 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103216 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103217 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1687_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_0
  simp only [← c103212, ← c103213, ← c103214] at e_1687_0
  have e_1687_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_1
  simp only [← c103215, ← c103216, k2, ← c103217] at e_1687_1
  have hr := e_1687_1
  simp only [e_1687_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1687 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16756)) := by
  have c103218 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103219 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103220 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103221 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103222 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103223 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1687_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_2
  simp only [← c103218, ← c103219, ← c103220] at e_1687_2
  have e_1687_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_3
  simp only [← c103221, ← c103222, k2, ← c103223] at e_1687_3
  have hr := e_1687_3
  simp only [e_1687_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1687 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16757)) := by
  have c103224 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103225 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103226 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103227 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103228 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103229 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1687_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_4
  simp only [← c103224, ← c103225, ← c103226] at e_1687_4
  have e_1687_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_5
  simp only [← c103227, ← c103228, k2, ← c103229] at e_1687_5
  have hr := e_1687_5
  simp only [e_1687_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1687 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16758)) := by
  have c103230 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103231 := (publicBatchWrapper89_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103232 := (publicBatchWrapper89_copies3226 a h).1
  have c103233 := (publicBatchWrapper89_copies3226 a h).2.1
  have c103234 := (publicBatchWrapper89_copies3226 a h).2.2.1
  have c103235 := (publicBatchWrapper89_copies3226 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1687_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_6
  simp only [← c103230, ← c103231, ← c103232] at e_1687_6
  have e_1687_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_7
  simp only [← c103233, ← c103234, k2, ← c103235] at e_1687_7
  have hr := e_1687_7
  simp only [e_1687_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1687 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16759)) := by
  have c103236 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.1
  have c103237 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.1
  have c103238 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.1
  have c103239 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.1
  have c103240 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.1
  have c103241 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1687_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_8
  simp only [← c103236, ← c103237, ← c103238] at e_1687_8
  have e_1687_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_9
  simp only [← c103239, ← c103240, k2, ← c103241] at e_1687_9
  have hr := e_1687_9
  simp only [e_1687_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1687 47) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16760)) := by
  have c103242 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103243 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103244 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103245 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103246 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103247 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1687_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_10
  simp only [← c103242, ← c103243, ← c103244] at e_1687_10
  have e_1687_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_11
  simp only [← c103245, ← c103246, k2, ← c103247] at e_1687_11
  have hr := e_1687_11
  simp only [e_1687_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1687 55) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16761)) := by
  have c103248 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103249 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103250 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103251 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103252 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103253 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1687_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_12
  simp only [← c103248, ← c103249, ← c103250] at e_1687_12
  have e_1687_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_13
  simp only [← c103251, ← c103252, k2, ← c103253] at e_1687_13
  have hr := e_1687_13
  simp only [e_1687_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1687 63) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16762)) := by
  have c103254 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103255 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103256 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103257 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103258 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103259 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1687_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_14
  simp only [← c103254, ← c103255, ← c103256] at e_1687_14
  have e_1687_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_15
  simp only [← c103257, ← c103258, k2, ← c103259] at e_1687_15
  have hr := e_1687_15
  simp only [e_1687_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1687 71) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16763)) := by
  have c103260 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103261 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103262 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103263 := (publicBatchWrapper89_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103264 := (publicBatchWrapper89_copies3227 a h).1
  have c103265 := (publicBatchWrapper89_copies3227 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1687_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_16
  simp only [← c103260, ← c103261, ← c103262] at e_1687_16
  have e_1687_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_17
  simp only [← c103263, ← c103264, k2, ← c103265] at e_1687_17
  have hr := e_1687_17
  simp only [e_1687_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1687 79) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16764)) := by
  have c103266 := (publicBatchWrapper89_copies3227 a h).2.2.1
  have c103267 := (publicBatchWrapper89_copies3227 a h).2.2.2.1
  have c103268 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.1
  have c103269 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.1
  have c103270 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.1
  have c103271 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1687_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_18
  simp only [← c103266, ← c103267, ← c103268] at e_1687_18
  have e_1687_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1687 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_19
  simp only [← c103269, ← c103270, k2, ← c103271] at e_1687_19
  have hr := e_1687_19
  simp only [e_1687_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1688 7) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16765)) := by
  have c103272 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.1
  have c103273 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.1
  have c103274 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103275 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103276 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103277 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1688_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_0
  simp only [← c103272, ← c103273, ← c103274] at e_1688_0
  have e_1688_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_1
  simp only [← c103275, ← c103276, k2, ← c103277] at e_1688_1
  have hr := e_1688_1
  simp only [e_1688_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1688 15) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16766)) := by
  have c103278 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103279 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103280 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103281 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103282 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103283 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1688_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_2
  simp only [← c103278, ← c103279, ← c103280] at e_1688_2
  have e_1688_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_3
  simp only [← c103281, ← c103282, k2, ← c103283] at e_1688_3
  have hr := e_1688_3
  simp only [e_1688_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1688 23) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16767)) := by
  have c103284 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103285 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103286 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103287 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103288 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103289 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1688_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_4
  simp only [← c103284, ← c103285, ← c103286] at e_1688_4
  have e_1688_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_5
  simp only [← c103287, ← c103288, k2, ← c103289] at e_1688_5
  have hr := e_1688_5
  simp only [e_1688_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1688 31) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16768)) := by
  have c103290 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103291 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103292 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103293 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103294 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103295 := (publicBatchWrapper89_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1688_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_6
  simp only [← c103290, ← c103291, ← c103292] at e_1688_6
  have e_1688_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_7
  simp only [← c103293, ← c103294, k2, ← c103295] at e_1688_7
  have hr := e_1688_7
  simp only [e_1688_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1688 39) = bselect (a (.wire 81 31)) (a (.virt 200)) (a (.virt 16769)) := by
  have c103296 := (publicBatchWrapper89_copies3228 a h).1
  have c103297 := (publicBatchWrapper89_copies3228 a h).2.1
  have c103298 := (publicBatchWrapper89_copies3228 a h).2.2.1
  have c103299 := (publicBatchWrapper89_copies3228 a h).2.2.2.1
  have c103300 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.1
  have c103301 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1688_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_8
  simp only [← c103296, ← c103297, ← c103298] at e_1688_8
  have e_1688_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_9
  simp only [← c103299, ← c103300, k2, ← c103301] at e_1688_9
  have hr := e_1688_9
  simp only [e_1688_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1688 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16810)) := by
  have c103302 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.1
  have c103303 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.1
  have c103304 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.1
  have c103305 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.1
  have c103306 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103307 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1688_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_10
  simp only [← c103302, ← c103303, ← c103304] at e_1688_10
  have e_1688_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_11
  simp only [← c103305, ← c103306, k2, ← c103307] at e_1688_11
  have hr := e_1688_11
  simp only [e_1688_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1688 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16811)) := by
  have c103308 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103309 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103310 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103311 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103312 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103313 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1688_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_12
  simp only [← c103308, ← c103309, ← c103310] at e_1688_12
  have e_1688_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_13
  simp only [← c103311, ← c103312, k2, ← c103313] at e_1688_13
  have hr := e_1688_13
  simp only [e_1688_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1688 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16812)) := by
  have c103314 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103315 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103316 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103317 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103318 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103319 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1688_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_14
  simp only [← c103314, ← c103315, ← c103316] at e_1688_14
  have e_1688_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_15
  simp only [← c103317, ← c103318, k2, ← c103319] at e_1688_15
  have hr := e_1688_15
  simp only [e_1688_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1688 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16813)) := by
  have c103320 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103321 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103322 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103323 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103324 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103325 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1688_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_16
  simp only [← c103320, ← c103321, ← c103322] at e_1688_16
  have e_1688_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_17
  simp only [← c103323, ← c103324, k2, ← c103325] at e_1688_17
  have hr := e_1688_17
  simp only [e_1688_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1688 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16814)) := by
  have c103326 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103327 := (publicBatchWrapper89_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103328 := (publicBatchWrapper89_copies3229 a h).1
  have c103329 := (publicBatchWrapper89_copies3229 a h).2.1
  have c103330 := (publicBatchWrapper89_copies3229 a h).2.2.1
  have c103331 := (publicBatchWrapper89_copies3229 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1688_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_18
  simp only [← c103326, ← c103327, ← c103328] at e_1688_18
  have e_1688_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1688 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_19
  simp only [← c103329, ← c103330, k2, ← c103331] at e_1688_19
  have hr := e_1688_19
  simp only [e_1688_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1689 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16815)) := by
  have c103332 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.1
  have c103333 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.1
  have c103334 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.1
  have c103335 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.1
  have c103336 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.1
  have c103337 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1689_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_0
  simp only [← c103332, ← c103333, ← c103334] at e_1689_0
  have e_1689_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_1
  simp only [← c103335, ← c103336, k2, ← c103337] at e_1689_1
  have hr := e_1689_1
  simp only [e_1689_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1689 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16816)) := by
  have c103338 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103339 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103340 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103341 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103342 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103343 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1689_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_2
  simp only [← c103338, ← c103339, ← c103340] at e_1689_2
  have e_1689_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_3
  simp only [← c103341, ← c103342, k2, ← c103343] at e_1689_3
  have hr := e_1689_3
  simp only [e_1689_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1689 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16817)) := by
  have c103344 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103345 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103346 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103347 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103348 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103349 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1689_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_4
  simp only [← c103344, ← c103345, ← c103346] at e_1689_4
  have e_1689_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_5
  simp only [← c103347, ← c103348, k2, ← c103349] at e_1689_5
  have hr := e_1689_5
  simp only [e_1689_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1689 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16818)) := by
  have c103350 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103351 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103352 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103353 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103354 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103355 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1689_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_6
  simp only [← c103350, ← c103351, ← c103352] at e_1689_6
  have e_1689_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_7
  simp only [← c103353, ← c103354, k2, ← c103355] at e_1689_7
  have hr := e_1689_7
  simp only [e_1689_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1689 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16819)) := by
  have c103356 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103357 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103358 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103359 := (publicBatchWrapper89_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103360 := (publicBatchWrapper89_copies3230 a h).1
  have c103361 := (publicBatchWrapper89_copies3230 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1689_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_8
  simp only [← c103356, ← c103357, ← c103358] at e_1689_8
  have e_1689_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_9
  simp only [← c103359, ← c103360, k2, ← c103361] at e_1689_9
  have hr := e_1689_9
  simp only [e_1689_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1689 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16820)) := by
  have c103362 := (publicBatchWrapper89_copies3230 a h).2.2.1
  have c103363 := (publicBatchWrapper89_copies3230 a h).2.2.2.1
  have c103364 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.1
  have c103365 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.1
  have c103366 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.1
  have c103367 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1689_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_10
  simp only [← c103362, ← c103363, ← c103364] at e_1689_10
  have e_1689_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_11
  simp only [← c103365, ← c103366, k2, ← c103367] at e_1689_11
  have hr := e_1689_11
  simp only [e_1689_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1689 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16821)) := by
  have c103368 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.1
  have c103369 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.1
  have c103370 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103371 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103372 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103373 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1689_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_12
  simp only [← c103368, ← c103369, ← c103370] at e_1689_12
  have e_1689_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_13
  simp only [← c103371, ← c103372, k2, ← c103373] at e_1689_13
  have hr := e_1689_13
  simp only [e_1689_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1689 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16822)) := by
  have c103374 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103375 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103376 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103377 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103378 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103379 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1689_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_14
  simp only [← c103374, ← c103375, ← c103376] at e_1689_14
  have e_1689_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_15
  simp only [← c103377, ← c103378, k2, ← c103379] at e_1689_15
  have hr := e_1689_15
  simp only [e_1689_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1689 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16823)) := by
  have c103380 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103381 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103382 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103383 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103384 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103385 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1689_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_16
  simp only [← c103380, ← c103381, ← c103382] at e_1689_16
  have e_1689_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_17
  simp only [← c103383, ← c103384, k2, ← c103385] at e_1689_17
  have hr := e_1689_17
  simp only [e_1689_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1689 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16824)) := by
  have c103386 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103387 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103388 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103389 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103390 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103391 := (publicBatchWrapper89_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1689_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_18
  simp only [← c103386, ← c103387, ← c103388] at e_1689_18
  have e_1689_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1689 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_19
  simp only [← c103389, ← c103390, k2, ← c103391] at e_1689_19
  have hr := e_1689_19
  simp only [e_1689_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1690 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16825)) := by
  have c103392 := (publicBatchWrapper89_copies3231 a h).1
  have c103393 := (publicBatchWrapper89_copies3231 a h).2.1
  have c103394 := (publicBatchWrapper89_copies3231 a h).2.2.1
  have c103395 := (publicBatchWrapper89_copies3231 a h).2.2.2.1
  have c103396 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.1
  have c103397 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1690_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_0
  simp only [← c103392, ← c103393, ← c103394] at e_1690_0
  have e_1690_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_1
  simp only [← c103395, ← c103396, k2, ← c103397] at e_1690_1
  have hr := e_1690_1
  simp only [e_1690_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1690 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16826)) := by
  have c103398 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.1
  have c103399 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.1
  have c103400 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.1
  have c103401 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.1
  have c103402 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103403 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1690_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_2
  simp only [← c103398, ← c103399, ← c103400] at e_1690_2
  have e_1690_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_3
  simp only [← c103401, ← c103402, k2, ← c103403] at e_1690_3
  have hr := e_1690_3
  simp only [e_1690_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1690 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16827)) := by
  have c103404 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103405 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103406 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103407 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103408 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103409 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1690_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_4
  simp only [← c103404, ← c103405, ← c103406] at e_1690_4
  have e_1690_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_5
  simp only [← c103407, ← c103408, k2, ← c103409] at e_1690_5
  have hr := e_1690_5
  simp only [e_1690_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1690 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16828)) := by
  have c103410 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103411 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103412 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103413 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103414 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103415 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1690_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_6
  simp only [← c103410, ← c103411, ← c103412] at e_1690_6
  have e_1690_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_7
  simp only [← c103413, ← c103414, k2, ← c103415] at e_1690_7
  have hr := e_1690_7
  simp only [e_1690_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1690 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16829)) := by
  have c103416 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103417 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103418 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103419 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103420 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103421 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1690_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_8
  simp only [← c103416, ← c103417, ← c103418] at e_1690_8
  have e_1690_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_9
  simp only [← c103419, ← c103420, k2, ← c103421] at e_1690_9
  have hr := e_1690_9
  simp only [e_1690_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1690 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16830)) := by
  have c103422 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103423 := (publicBatchWrapper89_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103424 := (publicBatchWrapper89_copies3232 a h).1
  have c103425 := (publicBatchWrapper89_copies3232 a h).2.1
  have c103426 := (publicBatchWrapper89_copies3232 a h).2.2.1
  have c103427 := (publicBatchWrapper89_copies3232 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1690_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_10
  simp only [← c103422, ← c103423, ← c103424] at e_1690_10
  have e_1690_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_11
  simp only [← c103425, ← c103426, k2, ← c103427] at e_1690_11
  have hr := e_1690_11
  simp only [e_1690_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1690 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16831)) := by
  have c103428 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.1
  have c103429 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.1
  have c103430 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.1
  have c103431 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.1
  have c103432 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.1
  have c103433 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1690_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_12
  simp only [← c103428, ← c103429, ← c103430] at e_1690_12
  have e_1690_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_13
  simp only [← c103431, ← c103432, k2, ← c103433] at e_1690_13
  have hr := e_1690_13
  simp only [e_1690_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1690 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16832)) := by
  have c103434 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103435 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103436 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103437 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103438 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103439 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1690_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_14
  simp only [← c103434, ← c103435, ← c103436] at e_1690_14
  have e_1690_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_15
  simp only [← c103437, ← c103438, k2, ← c103439] at e_1690_15
  have hr := e_1690_15
  simp only [e_1690_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1690 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16833)) := by
  have c103440 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103441 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103442 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103443 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103444 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103445 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1690_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_16
  simp only [← c103440, ← c103441, ← c103442] at e_1690_16
  have e_1690_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_17
  simp only [← c103443, ← c103444, k2, ← c103445] at e_1690_17
  have hr := e_1690_17
  simp only [e_1690_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1690 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16834)) := by
  have c103446 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103447 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103448 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103449 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103450 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103451 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1690_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_18
  simp only [← c103446, ← c103447, ← c103448] at e_1690_18
  have e_1690_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1690 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_19
  simp only [← c103449, ← c103450, k2, ← c103451] at e_1690_19
  have hr := e_1690_19
  simp only [e_1690_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1691 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16835)) := by
  have c103452 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103453 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103454 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103455 := (publicBatchWrapper89_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103456 := (publicBatchWrapper89_copies3233 a h).1
  have c103457 := (publicBatchWrapper89_copies3233 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1691_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_0
  simp only [← c103452, ← c103453, ← c103454] at e_1691_0
  have e_1691_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_1
  simp only [← c103455, ← c103456, k2, ← c103457] at e_1691_1
  have hr := e_1691_1
  simp only [e_1691_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1691 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16836)) := by
  have c103458 := (publicBatchWrapper89_copies3233 a h).2.2.1
  have c103459 := (publicBatchWrapper89_copies3233 a h).2.2.2.1
  have c103460 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.1
  have c103461 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.1
  have c103462 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.1
  have c103463 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1691_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_2
  simp only [← c103458, ← c103459, ← c103460] at e_1691_2
  have e_1691_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_3
  simp only [← c103461, ← c103462, k2, ← c103463] at e_1691_3
  have hr := e_1691_3
  simp only [e_1691_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1691 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16837)) := by
  have c103464 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.1
  have c103465 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.1
  have c103466 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103467 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103468 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103469 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1691_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_4
  simp only [← c103464, ← c103465, ← c103466] at e_1691_4
  have e_1691_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_5
  simp only [← c103467, ← c103468, k2, ← c103469] at e_1691_5
  have hr := e_1691_5
  simp only [e_1691_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1691 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16838)) := by
  have c103470 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103471 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103472 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103473 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103474 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103475 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1691_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_6
  simp only [← c103470, ← c103471, ← c103472] at e_1691_6
  have e_1691_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_7
  simp only [← c103473, ← c103474, k2, ← c103475] at e_1691_7
  have hr := e_1691_7
  simp only [e_1691_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1691 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16839)) := by
  have c103476 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103477 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103478 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103479 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103480 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103481 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1691_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_8
  simp only [← c103476, ← c103477, ← c103478] at e_1691_8
  have e_1691_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_9
  simp only [← c103479, ← c103480, k2, ← c103481] at e_1691_9
  have hr := e_1691_9
  simp only [e_1691_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1691 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16840)) := by
  have c103482 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103483 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103484 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103485 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103486 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103487 := (publicBatchWrapper89_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1691_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_10
  simp only [← c103482, ← c103483, ← c103484] at e_1691_10
  have e_1691_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_11
  simp only [← c103485, ← c103486, k2, ← c103487] at e_1691_11
  have hr := e_1691_11
  simp only [e_1691_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1691 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16841)) := by
  have c103488 := (publicBatchWrapper89_copies3234 a h).1
  have c103489 := (publicBatchWrapper89_copies3234 a h).2.1
  have c103490 := (publicBatchWrapper89_copies3234 a h).2.2.1
  have c103491 := (publicBatchWrapper89_copies3234 a h).2.2.2.1
  have c103492 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.1
  have c103493 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1691_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_12
  simp only [← c103488, ← c103489, ← c103490] at e_1691_12
  have e_1691_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_13
  simp only [← c103491, ← c103492, k2, ← c103493] at e_1691_13
  have hr := e_1691_13
  simp only [e_1691_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1691 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16842)) := by
  have c103494 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.1
  have c103495 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.1
  have c103496 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.1
  have c103497 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.1
  have c103498 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103499 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1691_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_14
  simp only [← c103494, ← c103495, ← c103496] at e_1691_14
  have e_1691_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_15
  simp only [← c103497, ← c103498, k2, ← c103499] at e_1691_15
  have hr := e_1691_15
  simp only [e_1691_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1691 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16843)) := by
  have c103500 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103501 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103502 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103503 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103504 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103505 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1691_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_16
  simp only [← c103500, ← c103501, ← c103502] at e_1691_16
  have e_1691_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_17
  simp only [← c103503, ← c103504, k2, ← c103505] at e_1691_17
  have hr := e_1691_17
  simp only [e_1691_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1691 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16844)) := by
  have c103506 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103507 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103508 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103509 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103510 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103511 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1691_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_18
  simp only [← c103506, ← c103507, ← c103508] at e_1691_18
  have e_1691_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1691 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_19
  simp only [← c103509, ← c103510, k2, ← c103511] at e_1691_19
  have hr := e_1691_19
  simp only [e_1691_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1692 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16845)) := by
  have c103512 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103513 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103514 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103515 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103516 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103517 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1692_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_0
  simp only [← c103512, ← c103513, ← c103514] at e_1692_0
  have e_1692_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_1
  simp only [← c103515, ← c103516, k2, ← c103517] at e_1692_1
  have hr := e_1692_1
  simp only [e_1692_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1692 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16846)) := by
  have c103518 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103519 := (publicBatchWrapper89_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103520 := (publicBatchWrapper89_copies3235 a h).1
  have c103521 := (publicBatchWrapper89_copies3235 a h).2.1
  have c103522 := (publicBatchWrapper89_copies3235 a h).2.2.1
  have c103523 := (publicBatchWrapper89_copies3235 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1692_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_2
  simp only [← c103518, ← c103519, ← c103520] at e_1692_2
  have e_1692_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_3
  simp only [← c103521, ← c103522, k2, ← c103523] at e_1692_3
  have hr := e_1692_3
  simp only [e_1692_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1692 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16847)) := by
  have c103524 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.1
  have c103525 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.1
  have c103526 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.1
  have c103527 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.1
  have c103528 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.1
  have c103529 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1692_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_4
  simp only [← c103524, ← c103525, ← c103526] at e_1692_4
  have e_1692_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_5
  simp only [← c103527, ← c103528, k2, ← c103529] at e_1692_5
  have hr := e_1692_5
  simp only [e_1692_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1692 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16848)) := by
  have c103530 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103531 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103532 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103533 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103534 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103535 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1692_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_6
  simp only [← c103530, ← c103531, ← c103532] at e_1692_6
  have e_1692_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_7
  simp only [← c103533, ← c103534, k2, ← c103535] at e_1692_7
  have hr := e_1692_7
  simp only [e_1692_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1692 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16849)) := by
  have c103536 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103537 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103538 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103539 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103540 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103541 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1692_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_8
  simp only [← c103536, ← c103537, ← c103538] at e_1692_8
  have e_1692_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_9
  simp only [← c103539, ← c103540, k2, ← c103541] at e_1692_9
  have hr := e_1692_9
  simp only [e_1692_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1692 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16850)) := by
  have c103542 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103543 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103544 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103545 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103546 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103547 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1692_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_10
  simp only [← c103542, ← c103543, ← c103544] at e_1692_10
  have e_1692_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_11
  simp only [← c103545, ← c103546, k2, ← c103547] at e_1692_11
  have hr := e_1692_11
  simp only [e_1692_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1692 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16851)) := by
  have c103548 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103549 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103550 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103551 := (publicBatchWrapper89_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103552 := (publicBatchWrapper89_copies3236 a h).1
  have c103553 := (publicBatchWrapper89_copies3236 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1692_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_12
  simp only [← c103548, ← c103549, ← c103550] at e_1692_12
  have e_1692_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_13
  simp only [← c103551, ← c103552, k2, ← c103553] at e_1692_13
  have hr := e_1692_13
  simp only [e_1692_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1692 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16852)) := by
  have c103554 := (publicBatchWrapper89_copies3236 a h).2.2.1
  have c103555 := (publicBatchWrapper89_copies3236 a h).2.2.2.1
  have c103556 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.1
  have c103557 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.1
  have c103558 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.1
  have c103559 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1692_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_14
  simp only [← c103554, ← c103555, ← c103556] at e_1692_14
  have e_1692_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_15
  simp only [← c103557, ← c103558, k2, ← c103559] at e_1692_15
  have hr := e_1692_15
  simp only [e_1692_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1692 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16853)) := by
  have c103560 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.1
  have c103561 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.1
  have c103562 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103563 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103564 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103565 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1692_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_16
  simp only [← c103560, ← c103561, ← c103562] at e_1692_16
  have e_1692_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_17
  simp only [← c103563, ← c103564, k2, ← c103565] at e_1692_17
  have hr := e_1692_17
  simp only [e_1692_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1692 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16854)) := by
  have c103566 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103567 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103568 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103569 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103570 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103571 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1692_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_18
  simp only [← c103566, ← c103567, ← c103568] at e_1692_18
  have e_1692_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1692 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_19
  simp only [← c103569, ← c103570, k2, ← c103571] at e_1692_19
  have hr := e_1692_19
  simp only [e_1692_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1693 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16855)) := by
  have c103572 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103573 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103574 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103575 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103576 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103577 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1693_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_0
  simp only [← c103572, ← c103573, ← c103574] at e_1693_0
  have e_1693_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_1
  simp only [← c103575, ← c103576, k2, ← c103577] at e_1693_1
  have hr := e_1693_1
  simp only [e_1693_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1693 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16856)) := by
  have c103578 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103579 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103580 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103581 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103582 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103583 := (publicBatchWrapper89_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1693_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_2
  simp only [← c103578, ← c103579, ← c103580] at e_1693_2
  have e_1693_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_3
  simp only [← c103581, ← c103582, k2, ← c103583] at e_1693_3
  have hr := e_1693_3
  simp only [e_1693_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1693 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16857)) := by
  have c103584 := (publicBatchWrapper89_copies3237 a h).1
  have c103585 := (publicBatchWrapper89_copies3237 a h).2.1
  have c103586 := (publicBatchWrapper89_copies3237 a h).2.2.1
  have c103587 := (publicBatchWrapper89_copies3237 a h).2.2.2.1
  have c103588 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.1
  have c103589 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1693_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_4
  simp only [← c103584, ← c103585, ← c103586] at e_1693_4
  have e_1693_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_5
  simp only [← c103587, ← c103588, k2, ← c103589] at e_1693_5
  have hr := e_1693_5
  simp only [e_1693_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1693 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16858)) := by
  have c103590 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.1
  have c103591 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.1
  have c103592 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.1
  have c103593 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.1
  have c103594 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103595 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1693_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_6
  simp only [← c103590, ← c103591, ← c103592] at e_1693_6
  have e_1693_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_7
  simp only [← c103593, ← c103594, k2, ← c103595] at e_1693_7
  have hr := e_1693_7
  simp only [e_1693_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1693 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16859)) := by
  have c103596 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103597 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103598 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103599 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103600 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103601 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1693_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_8
  simp only [← c103596, ← c103597, ← c103598] at e_1693_8
  have e_1693_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_9
  simp only [← c103599, ← c103600, k2, ← c103601] at e_1693_9
  have hr := e_1693_9
  simp only [e_1693_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1693 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16860)) := by
  have c103602 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103603 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103604 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103605 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103606 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103607 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1693_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_10
  simp only [← c103602, ← c103603, ← c103604] at e_1693_10
  have e_1693_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_11
  simp only [← c103605, ← c103606, k2, ← c103607] at e_1693_11
  have hr := e_1693_11
  simp only [e_1693_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1693 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16861)) := by
  have c103608 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103609 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103610 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103611 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103612 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103613 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1693_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_12
  simp only [← c103608, ← c103609, ← c103610] at e_1693_12
  have e_1693_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_13
  simp only [← c103611, ← c103612, k2, ← c103613] at e_1693_13
  have hr := e_1693_13
  simp only [e_1693_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1693 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16862)) := by
  have c103614 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103615 := (publicBatchWrapper89_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103616 := (publicBatchWrapper89_copies3238 a h).1
  have c103617 := (publicBatchWrapper89_copies3238 a h).2.1
  have c103618 := (publicBatchWrapper89_copies3238 a h).2.2.1
  have c103619 := (publicBatchWrapper89_copies3238 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1693_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_14
  simp only [← c103614, ← c103615, ← c103616] at e_1693_14
  have e_1693_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_15
  simp only [← c103617, ← c103618, k2, ← c103619] at e_1693_15
  have hr := e_1693_15
  simp only [e_1693_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1693 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16863)) := by
  have c103620 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.1
  have c103621 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.1
  have c103622 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.1
  have c103623 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.1
  have c103624 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.1
  have c103625 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1693_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_16
  simp only [← c103620, ← c103621, ← c103622] at e_1693_16
  have e_1693_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_17
  simp only [← c103623, ← c103624, k2, ← c103625] at e_1693_17
  have hr := e_1693_17
  simp only [e_1693_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1693 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16864)) := by
  have c103626 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103627 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103628 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103629 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103630 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103631 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1693_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_18
  simp only [← c103626, ← c103627, ← c103628] at e_1693_18
  have e_1693_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1693 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_19
  simp only [← c103629, ← c103630, k2, ← c103631] at e_1693_19
  have hr := e_1693_19
  simp only [e_1693_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1694 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16865)) := by
  have c103632 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103633 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103634 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103635 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103636 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103637 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1694_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_0
  simp only [← c103632, ← c103633, ← c103634] at e_1694_0
  have e_1694_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_1
  simp only [← c103635, ← c103636, k2, ← c103637] at e_1694_1
  have hr := e_1694_1
  simp only [e_1694_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1694 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16866)) := by
  have c103638 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103639 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103640 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103641 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103642 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103643 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1694_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_2
  simp only [← c103638, ← c103639, ← c103640] at e_1694_2
  have e_1694_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_3
  simp only [← c103641, ← c103642, k2, ← c103643] at e_1694_3
  have hr := e_1694_3
  simp only [e_1694_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1694 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16867)) := by
  have c103644 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103645 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103646 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103647 := (publicBatchWrapper89_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103648 := (publicBatchWrapper89_copies3239 a h).1
  have c103649 := (publicBatchWrapper89_copies3239 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1694_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_4
  simp only [← c103644, ← c103645, ← c103646] at e_1694_4
  have e_1694_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_5
  simp only [← c103647, ← c103648, k2, ← c103649] at e_1694_5
  have hr := e_1694_5
  simp only [e_1694_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1694 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16868)) := by
  have c103650 := (publicBatchWrapper89_copies3239 a h).2.2.1
  have c103651 := (publicBatchWrapper89_copies3239 a h).2.2.2.1
  have c103652 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.1
  have c103653 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.1
  have c103654 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.1
  have c103655 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1694_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_6
  simp only [← c103650, ← c103651, ← c103652] at e_1694_6
  have e_1694_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_7
  simp only [← c103653, ← c103654, k2, ← c103655] at e_1694_7
  have hr := e_1694_7
  simp only [e_1694_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1694 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16869)) := by
  have c103656 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.1
  have c103657 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.1
  have c103658 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103659 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103660 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103661 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1694_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_8
  simp only [← c103656, ← c103657, ← c103658] at e_1694_8
  have e_1694_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_9
  simp only [← c103659, ← c103660, k2, ← c103661] at e_1694_9
  have hr := e_1694_9
  simp only [e_1694_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1694 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16870)) := by
  have c103662 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103663 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103664 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103665 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103666 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103667 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1694_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_10
  simp only [← c103662, ← c103663, ← c103664] at e_1694_10
  have e_1694_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_11
  simp only [← c103665, ← c103666, k2, ← c103667] at e_1694_11
  have hr := e_1694_11
  simp only [e_1694_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1694 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16871)) := by
  have c103668 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103669 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103670 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103671 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103672 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103673 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1694_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_12
  simp only [← c103668, ← c103669, ← c103670] at e_1694_12
  have e_1694_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_13
  simp only [← c103671, ← c103672, k2, ← c103673] at e_1694_13
  have hr := e_1694_13
  simp only [e_1694_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1694 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16872)) := by
  have c103674 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103675 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103676 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103677 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103678 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103679 := (publicBatchWrapper89_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1694_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_14
  simp only [← c103674, ← c103675, ← c103676] at e_1694_14
  have e_1694_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_15
  simp only [← c103677, ← c103678, k2, ← c103679] at e_1694_15
  have hr := e_1694_15
  simp only [e_1694_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1694 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16873)) := by
  have c103680 := (publicBatchWrapper89_copies3240 a h).1
  have c103681 := (publicBatchWrapper89_copies3240 a h).2.1
  have c103682 := (publicBatchWrapper89_copies3240 a h).2.2.1
  have c103683 := (publicBatchWrapper89_copies3240 a h).2.2.2.1
  have c103684 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.1
  have c103685 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1694_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_16
  simp only [← c103680, ← c103681, ← c103682] at e_1694_16
  have e_1694_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_17
  simp only [← c103683, ← c103684, k2, ← c103685] at e_1694_17
  have hr := e_1694_17
  simp only [e_1694_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1694 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16874)) := by
  have c103686 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.1
  have c103687 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.1
  have c103688 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.1
  have c103689 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.1
  have c103690 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103691 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1694_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_18
  simp only [← c103686, ← c103687, ← c103688] at e_1694_18
  have e_1694_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1694 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_19
  simp only [← c103689, ← c103690, k2, ← c103691] at e_1694_19
  have hr := e_1694_19
  simp only [e_1694_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1695 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16875)) := by
  have c103692 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103693 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103694 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103695 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103696 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103697 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1695_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_0
  simp only [← c103692, ← c103693, ← c103694] at e_1695_0
  have e_1695_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_1
  simp only [← c103695, ← c103696, k2, ← c103697] at e_1695_1
  have hr := e_1695_1
  simp only [e_1695_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1695 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16876)) := by
  have c103698 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103699 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103700 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103701 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103702 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103703 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1695_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_2
  simp only [← c103698, ← c103699, ← c103700] at e_1695_2
  have e_1695_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_3
  simp only [← c103701, ← c103702, k2, ← c103703] at e_1695_3
  have hr := e_1695_3
  simp only [e_1695_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1695 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16877)) := by
  have c103704 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103705 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103706 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103707 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103708 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103709 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1695_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_4
  simp only [← c103704, ← c103705, ← c103706] at e_1695_4
  have e_1695_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_5
  simp only [← c103707, ← c103708, k2, ← c103709] at e_1695_5
  have hr := e_1695_5
  simp only [e_1695_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1695 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16878)) := by
  have c103710 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103711 := (publicBatchWrapper89_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103712 := (publicBatchWrapper89_copies3241 a h).1
  have c103713 := (publicBatchWrapper89_copies3241 a h).2.1
  have c103714 := (publicBatchWrapper89_copies3241 a h).2.2.1
  have c103715 := (publicBatchWrapper89_copies3241 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1695_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_6
  simp only [← c103710, ← c103711, ← c103712] at e_1695_6
  have e_1695_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_7
  simp only [← c103713, ← c103714, k2, ← c103715] at e_1695_7
  have hr := e_1695_7
  simp only [e_1695_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1695 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16879)) := by
  have c103716 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.1
  have c103717 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.1
  have c103718 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.1
  have c103719 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.1
  have c103720 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.1
  have c103721 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1695_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_8
  simp only [← c103716, ← c103717, ← c103718] at e_1695_8
  have e_1695_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_9
  simp only [← c103719, ← c103720, k2, ← c103721] at e_1695_9
  have hr := e_1695_9
  simp only [e_1695_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1695 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16880)) := by
  have c103722 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103723 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103724 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103725 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103726 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103727 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1695_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_10
  simp only [← c103722, ← c103723, ← c103724] at e_1695_10
  have e_1695_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_11
  simp only [← c103725, ← c103726, k2, ← c103727] at e_1695_11
  have hr := e_1695_11
  simp only [e_1695_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1695 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16881)) := by
  have c103728 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103729 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103730 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103731 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103732 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103733 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1695_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_12
  simp only [← c103728, ← c103729, ← c103730] at e_1695_12
  have e_1695_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_13
  simp only [← c103731, ← c103732, k2, ← c103733] at e_1695_13
  have hr := e_1695_13
  simp only [e_1695_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1695 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16882)) := by
  have c103734 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103735 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103736 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103737 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103738 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103739 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1695_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_14
  simp only [← c103734, ← c103735, ← c103736] at e_1695_14
  have e_1695_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_15
  simp only [← c103737, ← c103738, k2, ← c103739] at e_1695_15
  have hr := e_1695_15
  simp only [e_1695_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1695 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16883)) := by
  have c103740 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103741 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103742 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103743 := (publicBatchWrapper89_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103744 := (publicBatchWrapper89_copies3242 a h).1
  have c103745 := (publicBatchWrapper89_copies3242 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1695_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_16
  simp only [← c103740, ← c103741, ← c103742] at e_1695_16
  have e_1695_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_17
  simp only [← c103743, ← c103744, k2, ← c103745] at e_1695_17
  have hr := e_1695_17
  simp only [e_1695_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1695 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16884)) := by
  have c103746 := (publicBatchWrapper89_copies3242 a h).2.2.1
  have c103747 := (publicBatchWrapper89_copies3242 a h).2.2.2.1
  have c103748 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.1
  have c103749 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.1
  have c103750 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.1
  have c103751 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1695_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_18
  simp only [← c103746, ← c103747, ← c103748] at e_1695_18
  have e_1695_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1695 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_19
  simp only [← c103749, ← c103750, k2, ← c103751] at e_1695_19
  have hr := e_1695_19
  simp only [e_1695_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1696 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16885)) := by
  have c103752 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.1
  have c103753 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.1
  have c103754 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103755 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103756 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103757 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1696_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_0
  simp only [← c103752, ← c103753, ← c103754] at e_1696_0
  have e_1696_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_1
  simp only [← c103755, ← c103756, k2, ← c103757] at e_1696_1
  have hr := e_1696_1
  simp only [e_1696_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1696 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16886)) := by
  have c103758 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103759 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103760 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103761 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103762 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103763 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1696_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_2
  simp only [← c103758, ← c103759, ← c103760] at e_1696_2
  have e_1696_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_3
  simp only [← c103761, ← c103762, k2, ← c103763] at e_1696_3
  have hr := e_1696_3
  simp only [e_1696_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1696 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16887)) := by
  have c103764 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103765 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103766 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103767 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103768 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103769 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1696_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_4
  simp only [← c103764, ← c103765, ← c103766] at e_1696_4
  have e_1696_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_5
  simp only [← c103767, ← c103768, k2, ← c103769] at e_1696_5
  have hr := e_1696_5
  simp only [e_1696_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1696 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16888)) := by
  have c103770 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103771 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103772 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103773 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103774 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103775 := (publicBatchWrapper89_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1696_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_6
  simp only [← c103770, ← c103771, ← c103772] at e_1696_6
  have e_1696_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_7
  simp only [← c103773, ← c103774, k2, ← c103775] at e_1696_7
  have hr := e_1696_7
  simp only [e_1696_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1696 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16889)) := by
  have c103776 := (publicBatchWrapper89_copies3243 a h).1
  have c103777 := (publicBatchWrapper89_copies3243 a h).2.1
  have c103778 := (publicBatchWrapper89_copies3243 a h).2.2.1
  have c103779 := (publicBatchWrapper89_copies3243 a h).2.2.2.1
  have c103780 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.1
  have c103781 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1696_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_8
  simp only [← c103776, ← c103777, ← c103778] at e_1696_8
  have e_1696_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_9
  simp only [← c103779, ← c103780, k2, ← c103781] at e_1696_9
  have hr := e_1696_9
  simp only [e_1696_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1696 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16890)) := by
  have c103782 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.1
  have c103783 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.1
  have c103784 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.1
  have c103785 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.1
  have c103786 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103787 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1696_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_10
  simp only [← c103782, ← c103783, ← c103784] at e_1696_10
  have e_1696_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_11
  simp only [← c103785, ← c103786, k2, ← c103787] at e_1696_11
  have hr := e_1696_11
  simp only [e_1696_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1696 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16891)) := by
  have c103788 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103789 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103790 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103791 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103792 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103793 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1696_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_12
  simp only [← c103788, ← c103789, ← c103790] at e_1696_12
  have e_1696_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_13
  simp only [← c103791, ← c103792, k2, ← c103793] at e_1696_13
  have hr := e_1696_13
  simp only [e_1696_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1696 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16892)) := by
  have c103794 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103795 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103796 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103797 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103798 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103799 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1696_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_14
  simp only [← c103794, ← c103795, ← c103796] at e_1696_14
  have e_1696_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_15
  simp only [← c103797, ← c103798, k2, ← c103799] at e_1696_15
  have hr := e_1696_15
  simp only [e_1696_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1696 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16893)) := by
  have c103800 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103801 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103802 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103803 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103804 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103805 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1696_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_16
  simp only [← c103800, ← c103801, ← c103802] at e_1696_16
  have e_1696_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_17
  simp only [← c103803, ← c103804, k2, ← c103805] at e_1696_17
  have hr := e_1696_17
  simp only [e_1696_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1696 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16894)) := by
  have c103806 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103807 := (publicBatchWrapper89_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103808 := (publicBatchWrapper89_copies3244 a h).1
  have c103809 := (publicBatchWrapper89_copies3244 a h).2.1
  have c103810 := (publicBatchWrapper89_copies3244 a h).2.2.1
  have c103811 := (publicBatchWrapper89_copies3244 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1696_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_18
  simp only [← c103806, ← c103807, ← c103808] at e_1696_18
  have e_1696_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1696 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_19
  simp only [← c103809, ← c103810, k2, ← c103811] at e_1696_19
  have hr := e_1696_19
  simp only [e_1696_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1697 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16895)) := by
  have c103812 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.1
  have c103813 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.1
  have c103814 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.1
  have c103815 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.1
  have c103816 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.1
  have c103817 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1697_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_0
  simp only [← c103812, ← c103813, ← c103814] at e_1697_0
  have e_1697_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_1
  simp only [← c103815, ← c103816, k2, ← c103817] at e_1697_1
  have hr := e_1697_1
  simp only [e_1697_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1697 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16896)) := by
  have c103818 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103819 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103820 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103821 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103822 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103823 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1697_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_2
  simp only [← c103818, ← c103819, ← c103820] at e_1697_2
  have e_1697_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_3
  simp only [← c103821, ← c103822, k2, ← c103823] at e_1697_3
  have hr := e_1697_3
  simp only [e_1697_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1697 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16897)) := by
  have c103824 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103825 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103826 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103827 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103828 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103829 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1697_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_4
  simp only [← c103824, ← c103825, ← c103826] at e_1697_4
  have e_1697_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_5
  simp only [← c103827, ← c103828, k2, ← c103829] at e_1697_5
  have hr := e_1697_5
  simp only [e_1697_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1697 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16898)) := by
  have c103830 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103831 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103832 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103833 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103834 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103835 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1697_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_6
  simp only [← c103830, ← c103831, ← c103832] at e_1697_6
  have e_1697_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_7
  simp only [← c103833, ← c103834, k2, ← c103835] at e_1697_7
  have hr := e_1697_7
  simp only [e_1697_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1697 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16899)) := by
  have c103836 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103837 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103838 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103839 := (publicBatchWrapper89_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103840 := (publicBatchWrapper89_copies3245 a h).1
  have c103841 := (publicBatchWrapper89_copies3245 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1697_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_8
  simp only [← c103836, ← c103837, ← c103838] at e_1697_8
  have e_1697_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_9
  simp only [← c103839, ← c103840, k2, ← c103841] at e_1697_9
  have hr := e_1697_9
  simp only [e_1697_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1697 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16900)) := by
  have c103842 := (publicBatchWrapper89_copies3245 a h).2.2.1
  have c103843 := (publicBatchWrapper89_copies3245 a h).2.2.2.1
  have c103844 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.1
  have c103845 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.1
  have c103846 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.1
  have c103847 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1697_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_10
  simp only [← c103842, ← c103843, ← c103844] at e_1697_10
  have e_1697_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_11
  simp only [← c103845, ← c103846, k2, ← c103847] at e_1697_11
  have hr := e_1697_11
  simp only [e_1697_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1697 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16901)) := by
  have c103848 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.1
  have c103849 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.1
  have c103850 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103851 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103852 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103853 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1697_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_12
  simp only [← c103848, ← c103849, ← c103850] at e_1697_12
  have e_1697_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_13
  simp only [← c103851, ← c103852, k2, ← c103853] at e_1697_13
  have hr := e_1697_13
  simp only [e_1697_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1697 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16902)) := by
  have c103854 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103855 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103856 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103857 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103858 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103859 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1697_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_14
  simp only [← c103854, ← c103855, ← c103856] at e_1697_14
  have e_1697_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_15
  simp only [← c103857, ← c103858, k2, ← c103859] at e_1697_15
  have hr := e_1697_15
  simp only [e_1697_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1697 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16903)) := by
  have c103860 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103861 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103862 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103863 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103864 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103865 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1697_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_16
  simp only [← c103860, ← c103861, ← c103862] at e_1697_16
  have e_1697_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_17
  simp only [← c103863, ← c103864, k2, ← c103865] at e_1697_17
  have hr := e_1697_17
  simp only [e_1697_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1697 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16904)) := by
  have c103866 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103867 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103868 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103869 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103870 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103871 := (publicBatchWrapper89_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1697_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_18
  simp only [← c103866, ← c103867, ← c103868] at e_1697_18
  have e_1697_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1697 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_19
  simp only [← c103869, ← c103870, k2, ← c103871] at e_1697_19
  have hr := e_1697_19
  simp only [e_1697_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1698 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16905)) := by
  have c103872 := (publicBatchWrapper89_copies3246 a h).1
  have c103873 := (publicBatchWrapper89_copies3246 a h).2.1
  have c103874 := (publicBatchWrapper89_copies3246 a h).2.2.1
  have c103875 := (publicBatchWrapper89_copies3246 a h).2.2.2.1
  have c103876 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.1
  have c103877 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1698_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_0
  simp only [← c103872, ← c103873, ← c103874] at e_1698_0
  have e_1698_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_1
  simp only [← c103875, ← c103876, k2, ← c103877] at e_1698_1
  have hr := e_1698_1
  simp only [e_1698_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1698 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16906)) := by
  have c103878 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.1
  have c103879 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.1
  have c103880 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.1
  have c103881 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.1
  have c103882 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103883 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1698_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_2
  simp only [← c103878, ← c103879, ← c103880] at e_1698_2
  have e_1698_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_3
  simp only [← c103881, ← c103882, k2, ← c103883] at e_1698_3
  have hr := e_1698_3
  simp only [e_1698_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1698 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16907)) := by
  have c103884 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103885 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103886 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103887 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103888 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103889 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1698_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_4
  simp only [← c103884, ← c103885, ← c103886] at e_1698_4
  have e_1698_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_5
  simp only [← c103887, ← c103888, k2, ← c103889] at e_1698_5
  have hr := e_1698_5
  simp only [e_1698_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1698 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16908)) := by
  have c103890 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103891 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103892 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103893 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103894 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103895 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1698_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_6
  simp only [← c103890, ← c103891, ← c103892] at e_1698_6
  have e_1698_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_7
  simp only [← c103893, ← c103894, k2, ← c103895] at e_1698_7
  have hr := e_1698_7
  simp only [e_1698_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1698 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16909)) := by
  have c103896 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103897 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103898 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103899 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103900 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103901 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1698_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_8
  simp only [← c103896, ← c103897, ← c103898] at e_1698_8
  have e_1698_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_9
  simp only [← c103899, ← c103900, k2, ← c103901] at e_1698_9
  have hr := e_1698_9
  simp only [e_1698_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1698 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16910)) := by
  have c103902 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103903 := (publicBatchWrapper89_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103904 := (publicBatchWrapper89_copies3247 a h).1
  have c103905 := (publicBatchWrapper89_copies3247 a h).2.1
  have c103906 := (publicBatchWrapper89_copies3247 a h).2.2.1
  have c103907 := (publicBatchWrapper89_copies3247 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1698_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_10
  simp only [← c103902, ← c103903, ← c103904] at e_1698_10
  have e_1698_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_11
  simp only [← c103905, ← c103906, k2, ← c103907] at e_1698_11
  have hr := e_1698_11
  simp only [e_1698_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1698 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16911)) := by
  have c103908 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.1
  have c103909 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.1
  have c103910 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.1
  have c103911 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.1
  have c103912 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.1
  have c103913 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1698_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_12
  simp only [← c103908, ← c103909, ← c103910] at e_1698_12
  have e_1698_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_13
  simp only [← c103911, ← c103912, k2, ← c103913] at e_1698_13
  have hr := e_1698_13
  simp only [e_1698_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1698 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16912)) := by
  have c103914 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103915 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103916 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103917 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103918 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103919 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1698_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_14
  simp only [← c103914, ← c103915, ← c103916] at e_1698_14
  have e_1698_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_15
  simp only [← c103917, ← c103918, k2, ← c103919] at e_1698_15
  have hr := e_1698_15
  simp only [e_1698_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1698 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16913)) := by
  have c103920 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103921 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103922 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103923 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103924 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103925 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1698_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_16
  simp only [← c103920, ← c103921, ← c103922] at e_1698_16
  have e_1698_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_17
  simp only [← c103923, ← c103924, k2, ← c103925] at e_1698_17
  have hr := e_1698_17
  simp only [e_1698_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1698 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16914)) := by
  have c103926 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103927 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103928 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103929 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103930 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103931 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1698_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_18
  simp only [← c103926, ← c103927, ← c103928] at e_1698_18
  have e_1698_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1698 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_19
  simp only [← c103929, ← c103930, k2, ← c103931] at e_1698_19
  have hr := e_1698_19
  simp only [e_1698_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1699 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16915)) := by
  have c103932 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103933 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103934 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103935 := (publicBatchWrapper89_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103936 := (publicBatchWrapper89_copies3248 a h).1
  have c103937 := (publicBatchWrapper89_copies3248 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1699_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_0
  simp only [← c103932, ← c103933, ← c103934] at e_1699_0
  have e_1699_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_1
  simp only [← c103935, ← c103936, k2, ← c103937] at e_1699_1
  have hr := e_1699_1
  simp only [e_1699_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1699 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16916)) := by
  have c103938 := (publicBatchWrapper89_copies3248 a h).2.2.1
  have c103939 := (publicBatchWrapper89_copies3248 a h).2.2.2.1
  have c103940 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.1
  have c103941 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.1
  have c103942 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.1
  have c103943 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1699_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_2
  simp only [← c103938, ← c103939, ← c103940] at e_1699_2
  have e_1699_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_3
  simp only [← c103941, ← c103942, k2, ← c103943] at e_1699_3
  have hr := e_1699_3
  simp only [e_1699_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1699 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16917)) := by
  have c103944 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.1
  have c103945 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.1
  have c103946 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103947 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103948 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103949 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1699_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_4
  simp only [← c103944, ← c103945, ← c103946] at e_1699_4
  have e_1699_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_5
  simp only [← c103947, ← c103948, k2, ← c103949] at e_1699_5
  have hr := e_1699_5
  simp only [e_1699_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1699 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16918)) := by
  have c103950 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103951 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103952 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103953 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103954 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103955 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1699_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_6
  simp only [← c103950, ← c103951, ← c103952] at e_1699_6
  have e_1699_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_7
  simp only [← c103953, ← c103954, k2, ← c103955] at e_1699_7
  have hr := e_1699_7
  simp only [e_1699_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1699 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16919)) := by
  have c103956 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103957 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103958 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103959 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103960 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103961 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1699_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_8
  simp only [← c103956, ← c103957, ← c103958] at e_1699_8
  have e_1699_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_9
  simp only [← c103959, ← c103960, k2, ← c103961] at e_1699_9
  have hr := e_1699_9
  simp only [e_1699_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1699 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16920)) := by
  have c103962 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103963 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103964 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103965 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103966 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103967 := (publicBatchWrapper89_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1699_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_10
  simp only [← c103962, ← c103963, ← c103964] at e_1699_10
  have e_1699_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_11
  simp only [← c103965, ← c103966, k2, ← c103967] at e_1699_11
  have hr := e_1699_11
  simp only [e_1699_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1699 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16921)) := by
  have c103968 := (publicBatchWrapper89_copies3249 a h).1
  have c103969 := (publicBatchWrapper89_copies3249 a h).2.1
  have c103970 := (publicBatchWrapper89_copies3249 a h).2.2.1
  have c103971 := (publicBatchWrapper89_copies3249 a h).2.2.2.1
  have c103972 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.1
  have c103973 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1699_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_12
  simp only [← c103968, ← c103969, ← c103970] at e_1699_12
  have e_1699_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_13
  simp only [← c103971, ← c103972, k2, ← c103973] at e_1699_13
  have hr := e_1699_13
  simp only [e_1699_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1699 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16922)) := by
  have c103974 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.1
  have c103975 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.1
  have c103976 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.1
  have c103977 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.1
  have c103978 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103979 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1699_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_14
  simp only [← c103974, ← c103975, ← c103976] at e_1699_14
  have e_1699_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_15
  simp only [← c103977, ← c103978, k2, ← c103979] at e_1699_15
  have hr := e_1699_15
  simp only [e_1699_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1699 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16923)) := by
  have c103980 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103981 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103982 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103983 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103984 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103985 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1699_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_16
  simp only [← c103980, ← c103981, ← c103982] at e_1699_16
  have e_1699_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_17
  simp only [← c103983, ← c103984, k2, ← c103985] at e_1699_17
  have hr := e_1699_17
  simp only [e_1699_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1699 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16924)) := by
  have c103986 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103987 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103988 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103989 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103990 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103991 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1699_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_18
  simp only [← c103986, ← c103987, ← c103988] at e_1699_18
  have e_1699_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1699 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_19
  simp only [← c103989, ← c103990, k2, ← c103991] at e_1699_19
  have hr := e_1699_19
  simp only [e_1699_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1700 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16925)) := by
  have c103992 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103993 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103994 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103995 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103996 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103997 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1700_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_0
  simp only [← c103992, ← c103993, ← c103994] at e_1700_0
  have e_1700_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_1
  simp only [← c103995, ← c103996, k2, ← c103997] at e_1700_1
  have hr := e_1700_1
  simp only [e_1700_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1700 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16926)) := by
  have c103998 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103999 := (publicBatchWrapper89_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104000 := (publicBatchWrapper89_copies3250 a h).1
  have c104001 := (publicBatchWrapper89_copies3250 a h).2.1
  have c104002 := (publicBatchWrapper89_copies3250 a h).2.2.1
  have c104003 := (publicBatchWrapper89_copies3250 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1700_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_2
  simp only [← c103998, ← c103999, ← c104000] at e_1700_2
  have e_1700_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_3
  simp only [← c104001, ← c104002, k2, ← c104003] at e_1700_3
  have hr := e_1700_3
  simp only [e_1700_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1700 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16927)) := by
  have c104004 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.1
  have c104005 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.1
  have c104006 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.1
  have c104007 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.1
  have c104008 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.1
  have c104009 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1700_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_4
  simp only [← c104004, ← c104005, ← c104006] at e_1700_4
  have e_1700_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_5
  simp only [← c104007, ← c104008, k2, ← c104009] at e_1700_5
  have hr := e_1700_5
  simp only [e_1700_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1700 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16928)) := by
  have c104010 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104011 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104012 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104013 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104014 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104015 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1700_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_6
  simp only [← c104010, ← c104011, ← c104012] at e_1700_6
  have e_1700_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_7
  simp only [← c104013, ← c104014, k2, ← c104015] at e_1700_7
  have hr := e_1700_7
  simp only [e_1700_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1700 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16929)) := by
  have c104016 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104017 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104018 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104019 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104020 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104021 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1700_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_8
  simp only [← c104016, ← c104017, ← c104018] at e_1700_8
  have e_1700_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_9
  simp only [← c104019, ← c104020, k2, ← c104021] at e_1700_9
  have hr := e_1700_9
  simp only [e_1700_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1700 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16930)) := by
  have c104022 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104023 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104024 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104025 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104026 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104027 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1700_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_10
  simp only [← c104022, ← c104023, ← c104024] at e_1700_10
  have e_1700_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_11
  simp only [← c104025, ← c104026, k2, ← c104027] at e_1700_11
  have hr := e_1700_11
  simp only [e_1700_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1700 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16931)) := by
  have c104028 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104029 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104030 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104031 := (publicBatchWrapper89_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104032 := (publicBatchWrapper89_copies3251 a h).1
  have c104033 := (publicBatchWrapper89_copies3251 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1700_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_12
  simp only [← c104028, ← c104029, ← c104030] at e_1700_12
  have e_1700_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_13
  simp only [← c104031, ← c104032, k2, ← c104033] at e_1700_13
  have hr := e_1700_13
  simp only [e_1700_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1700 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16932)) := by
  have c104034 := (publicBatchWrapper89_copies3251 a h).2.2.1
  have c104035 := (publicBatchWrapper89_copies3251 a h).2.2.2.1
  have c104036 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.1
  have c104037 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.1
  have c104038 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.1
  have c104039 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1700_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_14
  simp only [← c104034, ← c104035, ← c104036] at e_1700_14
  have e_1700_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_15
  simp only [← c104037, ← c104038, k2, ← c104039] at e_1700_15
  have hr := e_1700_15
  simp only [e_1700_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1700 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16933)) := by
  have c104040 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.1
  have c104041 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.1
  have c104042 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104043 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104044 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104045 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1700_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_16
  simp only [← c104040, ← c104041, ← c104042] at e_1700_16
  have e_1700_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_17
  simp only [← c104043, ← c104044, k2, ← c104045] at e_1700_17
  have hr := e_1700_17
  simp only [e_1700_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1700 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16934)) := by
  have c104046 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104047 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104048 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104049 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104050 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104051 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1700_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_18
  simp only [← c104046, ← c104047, ← c104048] at e_1700_18
  have e_1700_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1700 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_19
  simp only [← c104049, ← c104050, k2, ← c104051] at e_1700_19
  have hr := e_1700_19
  simp only [e_1700_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1701 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16935)) := by
  have c104052 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104053 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104054 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104055 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104056 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104057 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1701_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_0
  simp only [← c104052, ← c104053, ← c104054] at e_1701_0
  have e_1701_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_1
  simp only [← c104055, ← c104056, k2, ← c104057] at e_1701_1
  have hr := e_1701_1
  simp only [e_1701_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1701 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16936)) := by
  have c104058 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104059 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104060 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104061 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104062 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104063 := (publicBatchWrapper89_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1701_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_2
  simp only [← c104058, ← c104059, ← c104060] at e_1701_2
  have e_1701_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_3
  simp only [← c104061, ← c104062, k2, ← c104063] at e_1701_3
  have hr := e_1701_3
  simp only [e_1701_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1701 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16937)) := by
  have c104064 := (publicBatchWrapper89_copies3252 a h).1
  have c104065 := (publicBatchWrapper89_copies3252 a h).2.1
  have c104066 := (publicBatchWrapper89_copies3252 a h).2.2.1
  have c104067 := (publicBatchWrapper89_copies3252 a h).2.2.2.1
  have c104068 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.1
  have c104069 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1701_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_4
  simp only [← c104064, ← c104065, ← c104066] at e_1701_4
  have e_1701_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_5
  simp only [← c104067, ← c104068, k2, ← c104069] at e_1701_5
  have hr := e_1701_5
  simp only [e_1701_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1701 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16938)) := by
  have c104070 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.1
  have c104071 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.1
  have c104072 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.1
  have c104073 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.1
  have c104074 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104075 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1701_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_6
  simp only [← c104070, ← c104071, ← c104072] at e_1701_6
  have e_1701_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_7
  simp only [← c104073, ← c104074, k2, ← c104075] at e_1701_7
  have hr := e_1701_7
  simp only [e_1701_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1701 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16939)) := by
  have c104076 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104077 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104078 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104079 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104080 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104081 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1701_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_8
  simp only [← c104076, ← c104077, ← c104078] at e_1701_8
  have e_1701_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_9
  simp only [← c104079, ← c104080, k2, ← c104081] at e_1701_9
  have hr := e_1701_9
  simp only [e_1701_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1701 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16940)) := by
  have c104082 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104083 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104084 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104085 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104086 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104087 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1701_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_10
  simp only [← c104082, ← c104083, ← c104084] at e_1701_10
  have e_1701_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_11
  simp only [← c104085, ← c104086, k2, ← c104087] at e_1701_11
  have hr := e_1701_11
  simp only [e_1701_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1701 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16941)) := by
  have c104088 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104089 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104090 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104091 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104092 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104093 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1701_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_12
  simp only [← c104088, ← c104089, ← c104090] at e_1701_12
  have e_1701_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_13
  simp only [← c104091, ← c104092, k2, ← c104093] at e_1701_13
  have hr := e_1701_13
  simp only [e_1701_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1701 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16942)) := by
  have c104094 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104095 := (publicBatchWrapper89_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104096 := (publicBatchWrapper89_copies3253 a h).1
  have c104097 := (publicBatchWrapper89_copies3253 a h).2.1
  have c104098 := (publicBatchWrapper89_copies3253 a h).2.2.1
  have c104099 := (publicBatchWrapper89_copies3253 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1701_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_14
  simp only [← c104094, ← c104095, ← c104096] at e_1701_14
  have e_1701_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_15
  simp only [← c104097, ← c104098, k2, ← c104099] at e_1701_15
  have hr := e_1701_15
  simp only [e_1701_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1701 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16943)) := by
  have c104100 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.1
  have c104101 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.1
  have c104102 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.1
  have c104103 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.1
  have c104104 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.1
  have c104105 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1701_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_16
  simp only [← c104100, ← c104101, ← c104102] at e_1701_16
  have e_1701_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_17
  simp only [← c104103, ← c104104, k2, ← c104105] at e_1701_17
  have hr := e_1701_17
  simp only [e_1701_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1701 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16944)) := by
  have c104106 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104107 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104108 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104109 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104110 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104111 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1701_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_18
  simp only [← c104106, ← c104107, ← c104108] at e_1701_18
  have e_1701_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1701 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_19
  simp only [← c104109, ← c104110, k2, ← c104111] at e_1701_19
  have hr := e_1701_19
  simp only [e_1701_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1702 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16945)) := by
  have c104112 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104113 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104114 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104115 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104116 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104117 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1702_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_0
  simp only [← c104112, ← c104113, ← c104114] at e_1702_0
  have e_1702_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_1
  simp only [← c104115, ← c104116, k2, ← c104117] at e_1702_1
  have hr := e_1702_1
  simp only [e_1702_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1702 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16946)) := by
  have c104118 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104119 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104120 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104121 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104122 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104123 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1702_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_2
  simp only [← c104118, ← c104119, ← c104120] at e_1702_2
  have e_1702_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_3
  simp only [← c104121, ← c104122, k2, ← c104123] at e_1702_3
  have hr := e_1702_3
  simp only [e_1702_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1702 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16947)) := by
  have c104124 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104125 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104126 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104127 := (publicBatchWrapper89_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104128 := (publicBatchWrapper89_copies3254 a h).1
  have c104129 := (publicBatchWrapper89_copies3254 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1702_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_4
  simp only [← c104124, ← c104125, ← c104126] at e_1702_4
  have e_1702_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_5
  simp only [← c104127, ← c104128, k2, ← c104129] at e_1702_5
  have hr := e_1702_5
  simp only [e_1702_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1702 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16948)) := by
  have c104130 := (publicBatchWrapper89_copies3254 a h).2.2.1
  have c104131 := (publicBatchWrapper89_copies3254 a h).2.2.2.1
  have c104132 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.1
  have c104133 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.1
  have c104134 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.1
  have c104135 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1702_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_6
  simp only [← c104130, ← c104131, ← c104132] at e_1702_6
  have e_1702_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_7
  simp only [← c104133, ← c104134, k2, ← c104135] at e_1702_7
  have hr := e_1702_7
  simp only [e_1702_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1702 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16949)) := by
  have c104136 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.1
  have c104137 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.1
  have c104138 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104139 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104140 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104141 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1702_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_8
  simp only [← c104136, ← c104137, ← c104138] at e_1702_8
  have e_1702_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_9
  simp only [← c104139, ← c104140, k2, ← c104141] at e_1702_9
  have hr := e_1702_9
  simp only [e_1702_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1702 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16950)) := by
  have c104142 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104143 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104144 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104145 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104146 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104147 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1702_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_10
  simp only [← c104142, ← c104143, ← c104144] at e_1702_10
  have e_1702_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_11
  simp only [← c104145, ← c104146, k2, ← c104147] at e_1702_11
  have hr := e_1702_11
  simp only [e_1702_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1702 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16951)) := by
  have c104148 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104149 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104150 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104151 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104152 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104153 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1702_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_12
  simp only [← c104148, ← c104149, ← c104150] at e_1702_12
  have e_1702_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_13
  simp only [← c104151, ← c104152, k2, ← c104153] at e_1702_13
  have hr := e_1702_13
  simp only [e_1702_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1702 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16952)) := by
  have c104154 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104155 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104156 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104157 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104158 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104159 := (publicBatchWrapper89_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1702_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_14
  simp only [← c104154, ← c104155, ← c104156] at e_1702_14
  have e_1702_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_15
  simp only [← c104157, ← c104158, k2, ← c104159] at e_1702_15
  have hr := e_1702_15
  simp only [e_1702_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1702 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16953)) := by
  have c104160 := (publicBatchWrapper89_copies3255 a h).1
  have c104161 := (publicBatchWrapper89_copies3255 a h).2.1
  have c104162 := (publicBatchWrapper89_copies3255 a h).2.2.1
  have c104163 := (publicBatchWrapper89_copies3255 a h).2.2.2.1
  have c104164 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.1
  have c104165 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1702_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_16
  simp only [← c104160, ← c104161, ← c104162] at e_1702_16
  have e_1702_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_17
  simp only [← c104163, ← c104164, k2, ← c104165] at e_1702_17
  have hr := e_1702_17
  simp only [e_1702_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1702 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16954)) := by
  have c104166 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.1
  have c104167 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.1
  have c104168 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.1
  have c104169 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.1
  have c104170 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104171 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1702_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_18
  simp only [← c104166, ← c104167, ← c104168] at e_1702_18
  have e_1702_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1702 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_19
  simp only [← c104169, ← c104170, k2, ← c104171] at e_1702_19
  have hr := e_1702_19
  simp only [e_1702_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1703 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16955)) := by
  have c104172 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104173 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104174 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104175 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104176 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104177 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1703_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_0
  simp only [← c104172, ← c104173, ← c104174] at e_1703_0
  have e_1703_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_1
  simp only [← c104175, ← c104176, k2, ← c104177] at e_1703_1
  have hr := e_1703_1
  simp only [e_1703_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1703 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16956)) := by
  have c104178 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104179 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104180 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104181 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104182 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104183 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1703_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_2
  simp only [← c104178, ← c104179, ← c104180] at e_1703_2
  have e_1703_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_3
  simp only [← c104181, ← c104182, k2, ← c104183] at e_1703_3
  have hr := e_1703_3
  simp only [e_1703_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1703 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16957)) := by
  have c104184 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104185 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104186 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104187 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104188 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104189 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1703_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_4
  simp only [← c104184, ← c104185, ← c104186] at e_1703_4
  have e_1703_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_5
  simp only [← c104187, ← c104188, k2, ← c104189] at e_1703_5
  have hr := e_1703_5
  simp only [e_1703_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1703 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16958)) := by
  have c104190 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104191 := (publicBatchWrapper89_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104192 := (publicBatchWrapper89_copies3256 a h).1
  have c104193 := (publicBatchWrapper89_copies3256 a h).2.1
  have c104194 := (publicBatchWrapper89_copies3256 a h).2.2.1
  have c104195 := (publicBatchWrapper89_copies3256 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1703_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_6
  simp only [← c104190, ← c104191, ← c104192] at e_1703_6
  have e_1703_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_7
  simp only [← c104193, ← c104194, k2, ← c104195] at e_1703_7
  have hr := e_1703_7
  simp only [e_1703_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1703 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16959)) := by
  have c104196 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.1
  have c104197 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.1
  have c104198 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.1
  have c104199 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.1
  have c104200 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.1
  have c104201 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1703_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_8
  simp only [← c104196, ← c104197, ← c104198] at e_1703_8
  have e_1703_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_9
  simp only [← c104199, ← c104200, k2, ← c104201] at e_1703_9
  have hr := e_1703_9
  simp only [e_1703_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1703 47) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16960)) := by
  have c104202 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104203 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104204 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104205 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104206 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104207 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1703_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_10
  simp only [← c104202, ← c104203, ← c104204] at e_1703_10
  have e_1703_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_11
  simp only [← c104205, ← c104206, k2, ← c104207] at e_1703_11
  have hr := e_1703_11
  simp only [e_1703_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1703 55) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16961)) := by
  have c104208 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104209 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104210 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104211 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104212 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104213 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1703_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_12
  simp only [← c104208, ← c104209, ← c104210] at e_1703_12
  have e_1703_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_13
  simp only [← c104211, ← c104212, k2, ← c104213] at e_1703_13
  have hr := e_1703_13
  simp only [e_1703_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1703 63) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16962)) := by
  have c104214 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104215 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104216 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104217 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104218 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104219 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1703_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_14
  simp only [← c104214, ← c104215, ← c104216] at e_1703_14
  have e_1703_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_15
  simp only [← c104217, ← c104218, k2, ← c104219] at e_1703_15
  have hr := e_1703_15
  simp only [e_1703_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1703 71) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16963)) := by
  have c104220 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104221 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104222 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104223 := (publicBatchWrapper89_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104224 := (publicBatchWrapper89_copies3257 a h).1
  have c104225 := (publicBatchWrapper89_copies3257 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1703_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_16
  simp only [← c104220, ← c104221, ← c104222] at e_1703_16
  have e_1703_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_17
  simp only [← c104223, ← c104224, k2, ← c104225] at e_1703_17
  have hr := e_1703_17
  simp only [e_1703_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1703 79) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16964)) := by
  have c104226 := (publicBatchWrapper89_copies3257 a h).2.2.1
  have c104227 := (publicBatchWrapper89_copies3257 a h).2.2.2.1
  have c104228 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.1
  have c104229 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.1
  have c104230 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.1
  have c104231 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1703_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_18
  simp only [← c104226, ← c104227, ← c104228] at e_1703_18
  have e_1703_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1703 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_19
  simp only [← c104229, ← c104230, k2, ← c104231] at e_1703_19
  have hr := e_1703_19
  simp only [e_1703_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1704 7) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16965)) := by
  have c104232 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.1
  have c104233 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.1
  have c104234 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104235 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104236 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104237 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1704_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_0
  simp only [← c104232, ← c104233, ← c104234] at e_1704_0
  have e_1704_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_1
  simp only [← c104235, ← c104236, k2, ← c104237] at e_1704_1
  have hr := e_1704_1
  simp only [e_1704_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1704 15) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16966)) := by
  have c104238 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104239 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104240 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104241 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104242 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104243 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1704_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_2
  simp only [← c104238, ← c104239, ← c104240] at e_1704_2
  have e_1704_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_3
  simp only [← c104241, ← c104242, k2, ← c104243] at e_1704_3
  have hr := e_1704_3
  simp only [e_1704_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1704 23) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16967)) := by
  have c104244 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104245 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104246 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104247 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104248 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104249 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1704_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_4
  simp only [← c104244, ← c104245, ← c104246] at e_1704_4
  have e_1704_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_5
  simp only [← c104247, ← c104248, k2, ← c104249] at e_1704_5
  have hr := e_1704_5
  simp only [e_1704_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1704 31) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16968)) := by
  have c104250 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104251 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104252 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104253 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104254 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104255 := (publicBatchWrapper89_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1704_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_6
  simp only [← c104250, ← c104251, ← c104252] at e_1704_6
  have e_1704_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_7
  simp only [← c104253, ← c104254, k2, ← c104255] at e_1704_7
  have hr := e_1704_7
  simp only [e_1704_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1704 39) = bselect (a (.wire 81 75)) (a (.virt 200)) (a (.virt 16969)) := by
  have c104256 := (publicBatchWrapper89_copies3258 a h).1
  have c104257 := (publicBatchWrapper89_copies3258 a h).2.1
  have c104258 := (publicBatchWrapper89_copies3258 a h).2.2.1
  have c104259 := (publicBatchWrapper89_copies3258 a h).2.2.2.1
  have c104260 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.1
  have c104261 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1704_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_8
  simp only [← c104256, ← c104257, ← c104258] at e_1704_8
  have e_1704_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_9
  simp only [← c104259, ← c104260, k2, ← c104261] at e_1704_9
  have hr := e_1704_9
  simp only [e_1704_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1704 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17010)) := by
  have c104262 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.1
  have c104263 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.1
  have c104264 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.1
  have c104265 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.1
  have c104266 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104267 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1704_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_10
  simp only [← c104262, ← c104263, ← c104264] at e_1704_10
  have e_1704_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_11
  simp only [← c104265, ← c104266, k2, ← c104267] at e_1704_11
  have hr := e_1704_11
  simp only [e_1704_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1704 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17011)) := by
  have c104268 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104269 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104270 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104271 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104272 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104273 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1704_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_12
  simp only [← c104268, ← c104269, ← c104270] at e_1704_12
  have e_1704_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_13
  simp only [← c104271, ← c104272, k2, ← c104273] at e_1704_13
  have hr := e_1704_13
  simp only [e_1704_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1704 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17012)) := by
  have c104274 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104275 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104276 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104277 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104278 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104279 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1704_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_14
  simp only [← c104274, ← c104275, ← c104276] at e_1704_14
  have e_1704_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_15
  simp only [← c104277, ← c104278, k2, ← c104279] at e_1704_15
  have hr := e_1704_15
  simp only [e_1704_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1704 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17013)) := by
  have c104280 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104281 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104282 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104283 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104284 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104285 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1704_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_16
  simp only [← c104280, ← c104281, ← c104282] at e_1704_16
  have e_1704_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_17
  simp only [← c104283, ← c104284, k2, ← c104285] at e_1704_17
  have hr := e_1704_17
  simp only [e_1704_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1704 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17014)) := by
  have c104286 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104287 := (publicBatchWrapper89_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104288 := (publicBatchWrapper89_copies3259 a h).1
  have c104289 := (publicBatchWrapper89_copies3259 a h).2.1
  have c104290 := (publicBatchWrapper89_copies3259 a h).2.2.1
  have c104291 := (publicBatchWrapper89_copies3259 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1704_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_18
  simp only [← c104286, ← c104287, ← c104288] at e_1704_18
  have e_1704_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1704 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_19
  simp only [← c104289, ← c104290, k2, ← c104291] at e_1704_19
  have hr := e_1704_19
  simp only [e_1704_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1705 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17015)) := by
  have c104292 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.1
  have c104293 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.1
  have c104294 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.1
  have c104295 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.1
  have c104296 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.1
  have c104297 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1705_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_0
  simp only [← c104292, ← c104293, ← c104294] at e_1705_0
  have e_1705_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_1
  simp only [← c104295, ← c104296, k2, ← c104297] at e_1705_1
  have hr := e_1705_1
  simp only [e_1705_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1705 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17016)) := by
  have c104298 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104299 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104300 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104301 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104302 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104303 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1705_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_2
  simp only [← c104298, ← c104299, ← c104300] at e_1705_2
  have e_1705_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_3
  simp only [← c104301, ← c104302, k2, ← c104303] at e_1705_3
  have hr := e_1705_3
  simp only [e_1705_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1705 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17017)) := by
  have c104304 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104305 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104306 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104307 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104308 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104309 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1705_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_4
  simp only [← c104304, ← c104305, ← c104306] at e_1705_4
  have e_1705_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_5
  simp only [← c104307, ← c104308, k2, ← c104309] at e_1705_5
  have hr := e_1705_5
  simp only [e_1705_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1705 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17018)) := by
  have c104310 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104311 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104312 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104313 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104314 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104315 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1705_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_6
  simp only [← c104310, ← c104311, ← c104312] at e_1705_6
  have e_1705_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_7
  simp only [← c104313, ← c104314, k2, ← c104315] at e_1705_7
  have hr := e_1705_7
  simp only [e_1705_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1705 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17019)) := by
  have c104316 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104317 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104318 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104319 := (publicBatchWrapper89_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104320 := (publicBatchWrapper89_copies3260 a h).1
  have c104321 := (publicBatchWrapper89_copies3260 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1705_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_8
  simp only [← c104316, ← c104317, ← c104318] at e_1705_8
  have e_1705_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_9
  simp only [← c104319, ← c104320, k2, ← c104321] at e_1705_9
  have hr := e_1705_9
  simp only [e_1705_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1705 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17020)) := by
  have c104322 := (publicBatchWrapper89_copies3260 a h).2.2.1
  have c104323 := (publicBatchWrapper89_copies3260 a h).2.2.2.1
  have c104324 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.1
  have c104325 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.1
  have c104326 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.1
  have c104327 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1705_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_10
  simp only [← c104322, ← c104323, ← c104324] at e_1705_10
  have e_1705_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_11
  simp only [← c104325, ← c104326, k2, ← c104327] at e_1705_11
  have hr := e_1705_11
  simp only [e_1705_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1705 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17021)) := by
  have c104328 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.1
  have c104329 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.1
  have c104330 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104331 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104332 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104333 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1705_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_12
  simp only [← c104328, ← c104329, ← c104330] at e_1705_12
  have e_1705_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_13
  simp only [← c104331, ← c104332, k2, ← c104333] at e_1705_13
  have hr := e_1705_13
  simp only [e_1705_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1705 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17022)) := by
  have c104334 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104335 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104336 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104337 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104338 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104339 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1705_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_14
  simp only [← c104334, ← c104335, ← c104336] at e_1705_14
  have e_1705_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_15
  simp only [← c104337, ← c104338, k2, ← c104339] at e_1705_15
  have hr := e_1705_15
  simp only [e_1705_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1705 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17023)) := by
  have c104340 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104341 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104342 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104343 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104344 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104345 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1705_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_16
  simp only [← c104340, ← c104341, ← c104342] at e_1705_16
  have e_1705_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_17
  simp only [← c104343, ← c104344, k2, ← c104345] at e_1705_17
  have hr := e_1705_17
  simp only [e_1705_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1705 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17024)) := by
  have c104346 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104347 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104348 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104349 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104350 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104351 := (publicBatchWrapper89_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1705_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_18
  simp only [← c104346, ← c104347, ← c104348] at e_1705_18
  have e_1705_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1705 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_19
  simp only [← c104349, ← c104350, k2, ← c104351] at e_1705_19
  have hr := e_1705_19
  simp only [e_1705_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1706 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17025)) := by
  have c104352 := (publicBatchWrapper89_copies3261 a h).1
  have c104353 := (publicBatchWrapper89_copies3261 a h).2.1
  have c104354 := (publicBatchWrapper89_copies3261 a h).2.2.1
  have c104355 := (publicBatchWrapper89_copies3261 a h).2.2.2.1
  have c104356 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.1
  have c104357 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1706_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_0
  simp only [← c104352, ← c104353, ← c104354] at e_1706_0
  have e_1706_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_1
  simp only [← c104355, ← c104356, k2, ← c104357] at e_1706_1
  have hr := e_1706_1
  simp only [e_1706_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1706 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17026)) := by
  have c104358 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.1
  have c104359 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.1
  have c104360 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.1
  have c104361 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.1
  have c104362 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104363 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1706_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_2
  simp only [← c104358, ← c104359, ← c104360] at e_1706_2
  have e_1706_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_3
  simp only [← c104361, ← c104362, k2, ← c104363] at e_1706_3
  have hr := e_1706_3
  simp only [e_1706_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1706 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17027)) := by
  have c104364 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104365 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104366 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104367 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104368 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104369 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1706_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_4
  simp only [← c104364, ← c104365, ← c104366] at e_1706_4
  have e_1706_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_5
  simp only [← c104367, ← c104368, k2, ← c104369] at e_1706_5
  have hr := e_1706_5
  simp only [e_1706_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1706 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17028)) := by
  have c104370 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104371 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104372 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104373 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104374 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104375 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1706_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_6
  simp only [← c104370, ← c104371, ← c104372] at e_1706_6
  have e_1706_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_7
  simp only [← c104373, ← c104374, k2, ← c104375] at e_1706_7
  have hr := e_1706_7
  simp only [e_1706_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1706 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17029)) := by
  have c104376 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104377 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104378 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104379 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104380 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104381 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1706_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_8
  simp only [← c104376, ← c104377, ← c104378] at e_1706_8
  have e_1706_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_9
  simp only [← c104379, ← c104380, k2, ← c104381] at e_1706_9
  have hr := e_1706_9
  simp only [e_1706_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1706 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17030)) := by
  have c104382 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104383 := (publicBatchWrapper89_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104384 := (publicBatchWrapper89_copies3262 a h).1
  have c104385 := (publicBatchWrapper89_copies3262 a h).2.1
  have c104386 := (publicBatchWrapper89_copies3262 a h).2.2.1
  have c104387 := (publicBatchWrapper89_copies3262 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1706_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_10
  simp only [← c104382, ← c104383, ← c104384] at e_1706_10
  have e_1706_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_11
  simp only [← c104385, ← c104386, k2, ← c104387] at e_1706_11
  have hr := e_1706_11
  simp only [e_1706_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1706 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17031)) := by
  have c104388 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.1
  have c104389 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.1
  have c104390 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.1
  have c104391 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.1
  have c104392 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.1
  have c104393 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1706_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_12
  simp only [← c104388, ← c104389, ← c104390] at e_1706_12
  have e_1706_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_13
  simp only [← c104391, ← c104392, k2, ← c104393] at e_1706_13
  have hr := e_1706_13
  simp only [e_1706_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1706 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17032)) := by
  have c104394 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104395 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104396 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104397 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104398 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104399 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1706_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_14
  simp only [← c104394, ← c104395, ← c104396] at e_1706_14
  have e_1706_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_15
  simp only [← c104397, ← c104398, k2, ← c104399] at e_1706_15
  have hr := e_1706_15
  simp only [e_1706_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1706 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17033)) := by
  have c104400 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104401 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104402 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104403 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104404 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104405 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1706_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_16
  simp only [← c104400, ← c104401, ← c104402] at e_1706_16
  have e_1706_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_17
  simp only [← c104403, ← c104404, k2, ← c104405] at e_1706_17
  have hr := e_1706_17
  simp only [e_1706_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1706 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17034)) := by
  have c104406 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104407 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104408 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104409 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104410 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104411 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1706_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_18
  simp only [← c104406, ← c104407, ← c104408] at e_1706_18
  have e_1706_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1706 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_19
  simp only [← c104409, ← c104410, k2, ← c104411] at e_1706_19
  have hr := e_1706_19
  simp only [e_1706_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1707 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17035)) := by
  have c104412 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104413 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104414 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104415 := (publicBatchWrapper89_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104416 := (publicBatchWrapper89_copies3263 a h).1
  have c104417 := (publicBatchWrapper89_copies3263 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1707_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_0
  simp only [← c104412, ← c104413, ← c104414] at e_1707_0
  have e_1707_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_1
  simp only [← c104415, ← c104416, k2, ← c104417] at e_1707_1
  have hr := e_1707_1
  simp only [e_1707_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1707 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17036)) := by
  have c104418 := (publicBatchWrapper89_copies3263 a h).2.2.1
  have c104419 := (publicBatchWrapper89_copies3263 a h).2.2.2.1
  have c104420 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.1
  have c104421 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.1
  have c104422 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.1
  have c104423 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1707_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_2
  simp only [← c104418, ← c104419, ← c104420] at e_1707_2
  have e_1707_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_3
  simp only [← c104421, ← c104422, k2, ← c104423] at e_1707_3
  have hr := e_1707_3
  simp only [e_1707_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1707 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17037)) := by
  have c104424 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.1
  have c104425 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.1
  have c104426 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104427 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104428 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104429 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1707_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_4
  simp only [← c104424, ← c104425, ← c104426] at e_1707_4
  have e_1707_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_5
  simp only [← c104427, ← c104428, k2, ← c104429] at e_1707_5
  have hr := e_1707_5
  simp only [e_1707_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1707 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17038)) := by
  have c104430 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104431 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104432 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104433 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104434 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104435 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1707_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_6
  simp only [← c104430, ← c104431, ← c104432] at e_1707_6
  have e_1707_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_7
  simp only [← c104433, ← c104434, k2, ← c104435] at e_1707_7
  have hr := e_1707_7
  simp only [e_1707_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1707 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17039)) := by
  have c104436 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104437 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104438 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104439 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104440 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104441 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1707_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_8
  simp only [← c104436, ← c104437, ← c104438] at e_1707_8
  have e_1707_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_9
  simp only [← c104439, ← c104440, k2, ← c104441] at e_1707_9
  have hr := e_1707_9
  simp only [e_1707_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1707 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17040)) := by
  have c104442 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104443 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104444 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104445 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104446 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104447 := (publicBatchWrapper89_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1707_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_10
  simp only [← c104442, ← c104443, ← c104444] at e_1707_10
  have e_1707_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_11
  simp only [← c104445, ← c104446, k2, ← c104447] at e_1707_11
  have hr := e_1707_11
  simp only [e_1707_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1707 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17041)) := by
  have c104448 := (publicBatchWrapper89_copies3264 a h).1
  have c104449 := (publicBatchWrapper89_copies3264 a h).2.1
  have c104450 := (publicBatchWrapper89_copies3264 a h).2.2.1
  have c104451 := (publicBatchWrapper89_copies3264 a h).2.2.2.1
  have c104452 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.1
  have c104453 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1707_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_12
  simp only [← c104448, ← c104449, ← c104450] at e_1707_12
  have e_1707_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_13
  simp only [← c104451, ← c104452, k2, ← c104453] at e_1707_13
  have hr := e_1707_13
  simp only [e_1707_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1707 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17042)) := by
  have c104454 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.1
  have c104455 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.1
  have c104456 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.1
  have c104457 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.1
  have c104458 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104459 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1707_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_14
  simp only [← c104454, ← c104455, ← c104456] at e_1707_14
  have e_1707_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_15
  simp only [← c104457, ← c104458, k2, ← c104459] at e_1707_15
  have hr := e_1707_15
  simp only [e_1707_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1707 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17043)) := by
  have c104460 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104461 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104462 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104463 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104464 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104465 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1707_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_16
  simp only [← c104460, ← c104461, ← c104462] at e_1707_16
  have e_1707_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_17
  simp only [← c104463, ← c104464, k2, ← c104465] at e_1707_17
  have hr := e_1707_17
  simp only [e_1707_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1707 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17044)) := by
  have c104466 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104467 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104468 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104469 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104470 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104471 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1707_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_18
  simp only [← c104466, ← c104467, ← c104468] at e_1707_18
  have e_1707_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1707 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_19
  simp only [← c104469, ← c104470, k2, ← c104471] at e_1707_19
  have hr := e_1707_19
  simp only [e_1707_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1708 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17045)) := by
  have c104472 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104473 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104474 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104475 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104476 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104477 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1708_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_0
  simp only [← c104472, ← c104473, ← c104474] at e_1708_0
  have e_1708_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_1
  simp only [← c104475, ← c104476, k2, ← c104477] at e_1708_1
  have hr := e_1708_1
  simp only [e_1708_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1708 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17046)) := by
  have c104478 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104479 := (publicBatchWrapper89_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104480 := (publicBatchWrapper89_copies3265 a h).1
  have c104481 := (publicBatchWrapper89_copies3265 a h).2.1
  have c104482 := (publicBatchWrapper89_copies3265 a h).2.2.1
  have c104483 := (publicBatchWrapper89_copies3265 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1708_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_2
  simp only [← c104478, ← c104479, ← c104480] at e_1708_2
  have e_1708_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_3
  simp only [← c104481, ← c104482, k2, ← c104483] at e_1708_3
  have hr := e_1708_3
  simp only [e_1708_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1708 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17047)) := by
  have c104484 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.1
  have c104485 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.1
  have c104486 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.1
  have c104487 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.1
  have c104488 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.1
  have c104489 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1708_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_4
  simp only [← c104484, ← c104485, ← c104486] at e_1708_4
  have e_1708_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_5
  simp only [← c104487, ← c104488, k2, ← c104489] at e_1708_5
  have hr := e_1708_5
  simp only [e_1708_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1708 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17048)) := by
  have c104490 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104491 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104492 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104493 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104494 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104495 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1708_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_6
  simp only [← c104490, ← c104491, ← c104492] at e_1708_6
  have e_1708_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_7
  simp only [← c104493, ← c104494, k2, ← c104495] at e_1708_7
  have hr := e_1708_7
  simp only [e_1708_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1708 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17049)) := by
  have c104496 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104497 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104498 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104499 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104500 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104501 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1708_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_8
  simp only [← c104496, ← c104497, ← c104498] at e_1708_8
  have e_1708_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_9
  simp only [← c104499, ← c104500, k2, ← c104501] at e_1708_9
  have hr := e_1708_9
  simp only [e_1708_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1708 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17050)) := by
  have c104502 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104503 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104504 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104505 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104506 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104507 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1708_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_10
  simp only [← c104502, ← c104503, ← c104504] at e_1708_10
  have e_1708_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_11
  simp only [← c104505, ← c104506, k2, ← c104507] at e_1708_11
  have hr := e_1708_11
  simp only [e_1708_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1708 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17051)) := by
  have c104508 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104509 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104510 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104511 := (publicBatchWrapper89_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104512 := (publicBatchWrapper89_copies3266 a h).1
  have c104513 := (publicBatchWrapper89_copies3266 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1708_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_12
  simp only [← c104508, ← c104509, ← c104510] at e_1708_12
  have e_1708_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_13
  simp only [← c104511, ← c104512, k2, ← c104513] at e_1708_13
  have hr := e_1708_13
  simp only [e_1708_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1708 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17052)) := by
  have c104514 := (publicBatchWrapper89_copies3266 a h).2.2.1
  have c104515 := (publicBatchWrapper89_copies3266 a h).2.2.2.1
  have c104516 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.1
  have c104517 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.1
  have c104518 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.1
  have c104519 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1708_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_14
  simp only [← c104514, ← c104515, ← c104516] at e_1708_14
  have e_1708_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_15
  simp only [← c104517, ← c104518, k2, ← c104519] at e_1708_15
  have hr := e_1708_15
  simp only [e_1708_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1708 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17053)) := by
  have c104520 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.1
  have c104521 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.1
  have c104522 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104523 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104524 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104525 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1708_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_16
  simp only [← c104520, ← c104521, ← c104522] at e_1708_16
  have e_1708_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_17
  simp only [← c104523, ← c104524, k2, ← c104525] at e_1708_17
  have hr := e_1708_17
  simp only [e_1708_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1708 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17054)) := by
  have c104526 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104527 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104528 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104529 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104530 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104531 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1708_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_18
  simp only [← c104526, ← c104527, ← c104528] at e_1708_18
  have e_1708_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1708 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_19
  simp only [← c104529, ← c104530, k2, ← c104531] at e_1708_19
  have hr := e_1708_19
  simp only [e_1708_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1709 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17055)) := by
  have c104532 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104533 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104534 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104535 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104536 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104537 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1709_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_0
  simp only [← c104532, ← c104533, ← c104534] at e_1709_0
  have e_1709_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_1
  simp only [← c104535, ← c104536, k2, ← c104537] at e_1709_1
  have hr := e_1709_1
  simp only [e_1709_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1709 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17056)) := by
  have c104538 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104539 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104540 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104541 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104542 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104543 := (publicBatchWrapper89_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1709_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_2
  simp only [← c104538, ← c104539, ← c104540] at e_1709_2
  have e_1709_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_3
  simp only [← c104541, ← c104542, k2, ← c104543] at e_1709_3
  have hr := e_1709_3
  simp only [e_1709_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1709 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17057)) := by
  have c104544 := (publicBatchWrapper89_copies3267 a h).1
  have c104545 := (publicBatchWrapper89_copies3267 a h).2.1
  have c104546 := (publicBatchWrapper89_copies3267 a h).2.2.1
  have c104547 := (publicBatchWrapper89_copies3267 a h).2.2.2.1
  have c104548 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.1
  have c104549 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1709_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_4
  simp only [← c104544, ← c104545, ← c104546] at e_1709_4
  have e_1709_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_5
  simp only [← c104547, ← c104548, k2, ← c104549] at e_1709_5
  have hr := e_1709_5
  simp only [e_1709_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1709 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17058)) := by
  have c104550 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.1
  have c104551 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.1
  have c104552 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.1
  have c104553 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.1
  have c104554 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104555 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1709_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_6
  simp only [← c104550, ← c104551, ← c104552] at e_1709_6
  have e_1709_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_7
  simp only [← c104553, ← c104554, k2, ← c104555] at e_1709_7
  have hr := e_1709_7
  simp only [e_1709_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1709 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17059)) := by
  have c104556 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104557 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104558 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104559 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104560 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104561 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1709_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_8
  simp only [← c104556, ← c104557, ← c104558] at e_1709_8
  have e_1709_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_9
  simp only [← c104559, ← c104560, k2, ← c104561] at e_1709_9
  have hr := e_1709_9
  simp only [e_1709_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1709 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17060)) := by
  have c104562 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104563 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104564 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104565 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104566 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104567 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1709_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_10
  simp only [← c104562, ← c104563, ← c104564] at e_1709_10
  have e_1709_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_11
  simp only [← c104565, ← c104566, k2, ← c104567] at e_1709_11
  have hr := e_1709_11
  simp only [e_1709_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1709 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17061)) := by
  have c104568 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104569 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104570 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104571 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104572 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104573 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1709_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_12
  simp only [← c104568, ← c104569, ← c104570] at e_1709_12
  have e_1709_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_13
  simp only [← c104571, ← c104572, k2, ← c104573] at e_1709_13
  have hr := e_1709_13
  simp only [e_1709_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1709 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17062)) := by
  have c104574 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104575 := (publicBatchWrapper89_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104576 := (publicBatchWrapper89_copies3268 a h).1
  have c104577 := (publicBatchWrapper89_copies3268 a h).2.1
  have c104578 := (publicBatchWrapper89_copies3268 a h).2.2.1
  have c104579 := (publicBatchWrapper89_copies3268 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1709_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_14
  simp only [← c104574, ← c104575, ← c104576] at e_1709_14
  have e_1709_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_15
  simp only [← c104577, ← c104578, k2, ← c104579] at e_1709_15
  have hr := e_1709_15
  simp only [e_1709_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1709 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17063)) := by
  have c104580 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.1
  have c104581 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.1
  have c104582 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.1
  have c104583 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.1
  have c104584 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.1
  have c104585 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1709_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_16
  simp only [← c104580, ← c104581, ← c104582] at e_1709_16
  have e_1709_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_17
  simp only [← c104583, ← c104584, k2, ← c104585] at e_1709_17
  have hr := e_1709_17
  simp only [e_1709_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1709 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17064)) := by
  have c104586 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104587 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104588 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104589 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104590 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104591 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1709_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_18
  simp only [← c104586, ← c104587, ← c104588] at e_1709_18
  have e_1709_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1709 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_19
  simp only [← c104589, ← c104590, k2, ← c104591] at e_1709_19
  have hr := e_1709_19
  simp only [e_1709_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1710 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17065)) := by
  have c104592 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104593 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104594 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104595 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104596 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104597 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1710_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_0
  simp only [← c104592, ← c104593, ← c104594] at e_1710_0
  have e_1710_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_1
  simp only [← c104595, ← c104596, k2, ← c104597] at e_1710_1
  have hr := e_1710_1
  simp only [e_1710_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1710 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17066)) := by
  have c104598 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104599 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104600 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104601 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104602 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104603 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1710_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_2
  simp only [← c104598, ← c104599, ← c104600] at e_1710_2
  have e_1710_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_3
  simp only [← c104601, ← c104602, k2, ← c104603] at e_1710_3
  have hr := e_1710_3
  simp only [e_1710_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1710 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17067)) := by
  have c104604 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104605 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104606 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104607 := (publicBatchWrapper89_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104608 := (publicBatchWrapper89_copies3269 a h).1
  have c104609 := (publicBatchWrapper89_copies3269 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1710_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_4
  simp only [← c104604, ← c104605, ← c104606] at e_1710_4
  have e_1710_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_5
  simp only [← c104607, ← c104608, k2, ← c104609] at e_1710_5
  have hr := e_1710_5
  simp only [e_1710_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1710 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17068)) := by
  have c104610 := (publicBatchWrapper89_copies3269 a h).2.2.1
  have c104611 := (publicBatchWrapper89_copies3269 a h).2.2.2.1
  have c104612 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.1
  have c104613 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.1
  have c104614 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.1
  have c104615 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1710_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_6
  simp only [← c104610, ← c104611, ← c104612] at e_1710_6
  have e_1710_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_7
  simp only [← c104613, ← c104614, k2, ← c104615] at e_1710_7
  have hr := e_1710_7
  simp only [e_1710_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1710 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17069)) := by
  have c104616 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.1
  have c104617 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.1
  have c104618 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104619 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104620 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104621 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1710_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_8
  simp only [← c104616, ← c104617, ← c104618] at e_1710_8
  have e_1710_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_9
  simp only [← c104619, ← c104620, k2, ← c104621] at e_1710_9
  have hr := e_1710_9
  simp only [e_1710_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1710 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17070)) := by
  have c104622 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104623 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104624 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104625 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104626 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104627 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1710_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_10
  simp only [← c104622, ← c104623, ← c104624] at e_1710_10
  have e_1710_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_11
  simp only [← c104625, ← c104626, k2, ← c104627] at e_1710_11
  have hr := e_1710_11
  simp only [e_1710_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1710 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17071)) := by
  have c104628 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104629 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104630 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104631 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104632 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104633 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1710_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_12
  simp only [← c104628, ← c104629, ← c104630] at e_1710_12
  have e_1710_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_13
  simp only [← c104631, ← c104632, k2, ← c104633] at e_1710_13
  have hr := e_1710_13
  simp only [e_1710_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1710 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17072)) := by
  have c104634 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104635 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104636 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104637 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104638 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104639 := (publicBatchWrapper89_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1710_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_14
  simp only [← c104634, ← c104635, ← c104636] at e_1710_14
  have e_1710_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_15
  simp only [← c104637, ← c104638, k2, ← c104639] at e_1710_15
  have hr := e_1710_15
  simp only [e_1710_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1710 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17073)) := by
  have c104640 := (publicBatchWrapper89_copies3270 a h).1
  have c104641 := (publicBatchWrapper89_copies3270 a h).2.1
  have c104642 := (publicBatchWrapper89_copies3270 a h).2.2.1
  have c104643 := (publicBatchWrapper89_copies3270 a h).2.2.2.1
  have c104644 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.1
  have c104645 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1710_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_16
  simp only [← c104640, ← c104641, ← c104642] at e_1710_16
  have e_1710_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_17
  simp only [← c104643, ← c104644, k2, ← c104645] at e_1710_17
  have hr := e_1710_17
  simp only [e_1710_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1710 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17074)) := by
  have c104646 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.1
  have c104647 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.1
  have c104648 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.1
  have c104649 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.1
  have c104650 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104651 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1710_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_18
  simp only [← c104646, ← c104647, ← c104648] at e_1710_18
  have e_1710_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1710 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_19
  simp only [← c104649, ← c104650, k2, ← c104651] at e_1710_19
  have hr := e_1710_19
  simp only [e_1710_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1711 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17075)) := by
  have c104652 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104653 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104654 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104655 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104656 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104657 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1711_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_0
  simp only [← c104652, ← c104653, ← c104654] at e_1711_0
  have e_1711_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_1
  simp only [← c104655, ← c104656, k2, ← c104657] at e_1711_1
  have hr := e_1711_1
  simp only [e_1711_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1711 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17076)) := by
  have c104658 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104659 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104660 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104661 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104662 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104663 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1711_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_2
  simp only [← c104658, ← c104659, ← c104660] at e_1711_2
  have e_1711_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_3
  simp only [← c104661, ← c104662, k2, ← c104663] at e_1711_3
  have hr := e_1711_3
  simp only [e_1711_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1711 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17077)) := by
  have c104664 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104665 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104666 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104667 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104668 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104669 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1711_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_4
  simp only [← c104664, ← c104665, ← c104666] at e_1711_4
  have e_1711_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_5
  simp only [← c104667, ← c104668, k2, ← c104669] at e_1711_5
  have hr := e_1711_5
  simp only [e_1711_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1711 31) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17078)) := by
  have c104670 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104671 := (publicBatchWrapper89_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104672 := (publicBatchWrapper89_copies3271 a h).1
  have c104673 := (publicBatchWrapper89_copies3271 a h).2.1
  have c104674 := (publicBatchWrapper89_copies3271 a h).2.2.1
  have c104675 := (publicBatchWrapper89_copies3271 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1711_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_6
  simp only [← c104670, ← c104671, ← c104672] at e_1711_6
  have e_1711_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_7
  simp only [← c104673, ← c104674, k2, ← c104675] at e_1711_7
  have hr := e_1711_7
  simp only [e_1711_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1711 39) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17079)) := by
  have c104676 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.1
  have c104677 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.1
  have c104678 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.1
  have c104679 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.1
  have c104680 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.1
  have c104681 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1711_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_8
  simp only [← c104676, ← c104677, ← c104678] at e_1711_8
  have e_1711_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_9
  simp only [← c104679, ← c104680, k2, ← c104681] at e_1711_9
  have hr := e_1711_9
  simp only [e_1711_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1711 47) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17080)) := by
  have c104682 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104683 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104684 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104685 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104686 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104687 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1711_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_10
  simp only [← c104682, ← c104683, ← c104684] at e_1711_10
  have e_1711_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_11
  simp only [← c104685, ← c104686, k2, ← c104687] at e_1711_11
  have hr := e_1711_11
  simp only [e_1711_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1711 55) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17081)) := by
  have c104688 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104689 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104690 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104691 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104692 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104693 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1711_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_12
  simp only [← c104688, ← c104689, ← c104690] at e_1711_12
  have e_1711_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_13
  simp only [← c104691, ← c104692, k2, ← c104693] at e_1711_13
  have hr := e_1711_13
  simp only [e_1711_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1711 63) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17082)) := by
  have c104694 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104695 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104696 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104697 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104698 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104699 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1711_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_14
  simp only [← c104694, ← c104695, ← c104696] at e_1711_14
  have e_1711_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_15
  simp only [← c104697, ← c104698, k2, ← c104699] at e_1711_15
  have hr := e_1711_15
  simp only [e_1711_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1711 71) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17083)) := by
  have c104700 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104701 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104702 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104703 := (publicBatchWrapper89_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104704 := (publicBatchWrapper89_copies3272 a h).1
  have c104705 := (publicBatchWrapper89_copies3272 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1711_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_16
  simp only [← c104700, ← c104701, ← c104702] at e_1711_16
  have e_1711_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_17
  simp only [← c104703, ← c104704, k2, ← c104705] at e_1711_17
  have hr := e_1711_17
  simp only [e_1711_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1711 79) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17084)) := by
  have c104706 := (publicBatchWrapper89_copies3272 a h).2.2.1
  have c104707 := (publicBatchWrapper89_copies3272 a h).2.2.2.1
  have c104708 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.1
  have c104709 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.1
  have c104710 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.1
  have c104711 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1711_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_18
  simp only [← c104706, ← c104707, ← c104708] at e_1711_18
  have e_1711_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1711 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_19
  simp only [← c104709, ← c104710, k2, ← c104711] at e_1711_19
  have hr := e_1711_19
  simp only [e_1711_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1712 7) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17085)) := by
  have c104712 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.1
  have c104713 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.1
  have c104714 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104715 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104716 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104717 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1712_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_0
  simp only [← c104712, ← c104713, ← c104714] at e_1712_0
  have e_1712_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_1
  simp only [← c104715, ← c104716, k2, ← c104717] at e_1712_1
  have hr := e_1712_1
  simp only [e_1712_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1712 15) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17086)) := by
  have c104718 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104719 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104720 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104721 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104722 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104723 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1712_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_2
  simp only [← c104718, ← c104719, ← c104720] at e_1712_2
  have e_1712_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_3
  simp only [← c104721, ← c104722, k2, ← c104723] at e_1712_3
  have hr := e_1712_3
  simp only [e_1712_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1712 23) = bselect (a (.wire 83 39)) (a (.virt 200)) (a (.virt 17087)) := by
  have c104724 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104725 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104726 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104727 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104728 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104729 := (publicBatchWrapper89_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1712_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_4
  simp only [← c104724, ← c104725, ← c104726] at e_1712_4
  have e_1712_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1712 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_5
  simp only [← c104727, ← c104728, k2, ← c104729] at e_1712_5
  have hr := e_1712_5
  simp only [e_1712_4] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
