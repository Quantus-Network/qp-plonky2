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

theorem publicBatchWrapper86_f5140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 554 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2746)) := by
  have c35193 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35194 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35195 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35196 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35197 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35198 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_554_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_16
  simp only [← c35193, ← c35194, ← c35195] at e_554_16
  have e_554_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_17
  simp only [← c35196, ← c35197, k2, ← c35198] at e_554_17
  have hr := e_554_17
  simp only [e_554_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 554 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2747)) := by
  have c35199 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35200 := (publicBatchWrapper86_copies1100 a h).1
  have c35201 := (publicBatchWrapper86_copies1100 a h).2.1
  have c35202 := (publicBatchWrapper86_copies1100 a h).2.2.1
  have c35203 := (publicBatchWrapper86_copies1100 a h).2.2.2.1
  have c35204 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_554_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_18
  simp only [← c35199, ← c35200, ← c35201] at e_554_18
  have e_554_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_19
  simp only [← c35202, ← c35203, k2, ← c35204] at e_554_19
  have hr := e_554_19
  simp only [e_554_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 555 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2748)) := by
  have c35205 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.1
  have c35206 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.1
  have c35207 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.1
  have c35208 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.1
  have c35209 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.1
  have c35210 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_555_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_0
  simp only [← c35205, ← c35206, ← c35207] at e_555_0
  have e_555_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_1
  simp only [← c35208, ← c35209, k2, ← c35210] at e_555_1
  have hr := e_555_1
  simp only [e_555_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 555 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2749)) := by
  have c35211 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35212 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35213 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35214 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35215 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35216 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_555_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_2
  simp only [← c35211, ← c35212, ← c35213] at e_555_2
  have e_555_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_3
  simp only [← c35214, ← c35215, k2, ← c35216] at e_555_3
  have hr := e_555_3
  simp only [e_555_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 555 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2750)) := by
  have c35217 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35218 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35219 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35220 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35221 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35222 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_555_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_4
  simp only [← c35217, ← c35218, ← c35219] at e_555_4
  have e_555_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_5
  simp only [← c35220, ← c35221, k2, ← c35222] at e_555_5
  have hr := e_555_5
  simp only [e_555_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 555 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2751)) := by
  have c35223 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35224 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35225 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35226 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35227 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35228 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_555_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_6
  simp only [← c35223, ← c35224, ← c35225] at e_555_6
  have e_555_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_7
  simp only [← c35226, ← c35227, k2, ← c35228] at e_555_7
  have hr := e_555_7
  simp only [e_555_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 555 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2752)) := by
  have c35229 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35230 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35231 := (publicBatchWrapper86_copies1100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35232 := (publicBatchWrapper86_copies1101 a h).1
  have c35233 := (publicBatchWrapper86_copies1101 a h).2.1
  have c35234 := (publicBatchWrapper86_copies1101 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_555_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_8
  simp only [← c35229, ← c35230, ← c35231] at e_555_8
  have e_555_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_9
  simp only [← c35232, ← c35233, k2, ← c35234] at e_555_9
  have hr := e_555_9
  simp only [e_555_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 555 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2753)) := by
  have c35235 := (publicBatchWrapper86_copies1101 a h).2.2.2.1
  have c35236 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.1
  have c35237 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.1
  have c35238 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.1
  have c35239 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.1
  have c35240 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_555_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_10
  simp only [← c35235, ← c35236, ← c35237] at e_555_10
  have e_555_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_11
  simp only [← c35238, ← c35239, k2, ← c35240] at e_555_11
  have hr := e_555_11
  simp only [e_555_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 555 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2754)) := by
  have c35241 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.1
  have c35242 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35243 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35244 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35245 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35246 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_555_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_12
  simp only [← c35241, ← c35242, ← c35243] at e_555_12
  have e_555_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_13
  simp only [← c35244, ← c35245, k2, ← c35246] at e_555_13
  have hr := e_555_13
  simp only [e_555_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 555 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2755)) := by
  have c35247 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35248 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35249 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35250 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35251 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35252 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_555_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_14
  simp only [← c35247, ← c35248, ← c35249] at e_555_14
  have e_555_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_15
  simp only [← c35250, ← c35251, k2, ← c35252] at e_555_15
  have hr := e_555_15
  simp only [e_555_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 555 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2756)) := by
  have c35253 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35254 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35255 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35256 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35257 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35258 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_555_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_16
  simp only [← c35253, ← c35254, ← c35255] at e_555_16
  have e_555_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_17
  simp only [← c35256, ← c35257, k2, ← c35258] at e_555_17
  have hr := e_555_17
  simp only [e_555_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 555 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2757)) := by
  have c35259 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35260 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35261 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35262 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35263 := (publicBatchWrapper86_copies1101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35264 := (publicBatchWrapper86_copies1102 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_555_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_18
  simp only [← c35259, ← c35260, ← c35261] at e_555_18
  have e_555_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row555 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_555_19
  simp only [← c35262, ← c35263, k2, ← c35264] at e_555_19
  have hr := e_555_19
  simp only [e_555_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 556 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2758)) := by
  have c35265 := (publicBatchWrapper86_copies1102 a h).2.1
  have c35266 := (publicBatchWrapper86_copies1102 a h).2.2.1
  have c35267 := (publicBatchWrapper86_copies1102 a h).2.2.2.1
  have c35268 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.1
  have c35269 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.1
  have c35270 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_556_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_0
  simp only [← c35265, ← c35266, ← c35267] at e_556_0
  have e_556_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_1
  simp only [← c35268, ← c35269, k2, ← c35270] at e_556_1
  have hr := e_556_1
  simp only [e_556_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 556 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2759)) := by
  have c35271 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.1
  have c35272 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.1
  have c35273 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.1
  have c35274 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35275 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35276 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_556_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_2
  simp only [← c35271, ← c35272, ← c35273] at e_556_2
  have e_556_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_3
  simp only [← c35274, ← c35275, k2, ← c35276] at e_556_3
  have hr := e_556_3
  simp only [e_556_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 556 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2760)) := by
  have c35277 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35278 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35279 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35280 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35281 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35282 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_556_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_4
  simp only [← c35277, ← c35278, ← c35279] at e_556_4
  have e_556_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_5
  simp only [← c35280, ← c35281, k2, ← c35282] at e_556_5
  have hr := e_556_5
  simp only [e_556_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 556 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2761)) := by
  have c35283 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35284 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35285 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35286 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35287 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35288 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_556_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_6
  simp only [← c35283, ← c35284, ← c35285] at e_556_6
  have e_556_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_7
  simp only [← c35286, ← c35287, k2, ← c35288] at e_556_7
  have hr := e_556_7
  simp only [e_556_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 556 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2762)) := by
  have c35289 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35290 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35291 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35292 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35293 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35294 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_556_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_8
  simp only [← c35289, ← c35290, ← c35291] at e_556_8
  have e_556_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_9
  simp only [← c35292, ← c35293, k2, ← c35294] at e_556_9
  have hr := e_556_9
  simp only [e_556_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 556 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2763)) := by
  have c35295 := (publicBatchWrapper86_copies1102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35296 := (publicBatchWrapper86_copies1103 a h).1
  have c35297 := (publicBatchWrapper86_copies1103 a h).2.1
  have c35298 := (publicBatchWrapper86_copies1103 a h).2.2.1
  have c35299 := (publicBatchWrapper86_copies1103 a h).2.2.2.1
  have c35300 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_556_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_10
  simp only [← c35295, ← c35296, ← c35297] at e_556_10
  have e_556_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_11
  simp only [← c35298, ← c35299, k2, ← c35300] at e_556_11
  have hr := e_556_11
  simp only [e_556_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 556 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2764)) := by
  have c35301 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.1
  have c35302 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.1
  have c35303 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.1
  have c35304 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.1
  have c35305 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.1
  have c35306 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_556_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_12
  simp only [← c35301, ← c35302, ← c35303] at e_556_12
  have e_556_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_13
  simp only [← c35304, ← c35305, k2, ← c35306] at e_556_13
  have hr := e_556_13
  simp only [e_556_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 556 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2765)) := by
  have c35307 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35308 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35309 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35310 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35311 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35312 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_556_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_14
  simp only [← c35307, ← c35308, ← c35309] at e_556_14
  have e_556_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_15
  simp only [← c35310, ← c35311, k2, ← c35312] at e_556_15
  have hr := e_556_15
  simp only [e_556_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 556 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2766)) := by
  have c35313 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35314 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35315 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35316 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35317 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35318 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_556_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_16
  simp only [← c35313, ← c35314, ← c35315] at e_556_16
  have e_556_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_17
  simp only [← c35316, ← c35317, k2, ← c35318] at e_556_17
  have hr := e_556_17
  simp only [e_556_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 556 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2767)) := by
  have c35319 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35320 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35321 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35322 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35323 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35324 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_556_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_18
  simp only [← c35319, ← c35320, ← c35321] at e_556_18
  have e_556_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row556 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_556_19
  simp only [← c35322, ← c35323, k2, ← c35324] at e_556_19
  have hr := e_556_19
  simp only [e_556_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 557 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2768)) := by
  have c35325 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35326 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35327 := (publicBatchWrapper86_copies1103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35328 := (publicBatchWrapper86_copies1104 a h).1
  have c35329 := (publicBatchWrapper86_copies1104 a h).2.1
  have c35330 := (publicBatchWrapper86_copies1104 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_557_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_0
  simp only [← c35325, ← c35326, ← c35327] at e_557_0
  have e_557_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_1
  simp only [← c35328, ← c35329, k2, ← c35330] at e_557_1
  have hr := e_557_1
  simp only [e_557_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 557 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2769)) := by
  have c35331 := (publicBatchWrapper86_copies1104 a h).2.2.2.1
  have c35332 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.1
  have c35333 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.1
  have c35334 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.1
  have c35335 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.1
  have c35336 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_557_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_2
  simp only [← c35331, ← c35332, ← c35333] at e_557_2
  have e_557_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_3
  simp only [← c35334, ← c35335, k2, ← c35336] at e_557_3
  have hr := e_557_3
  simp only [e_557_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 557 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2810)) := by
  have c35337 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.1
  have c35338 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35339 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35340 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35341 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35342 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_557_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_4
  simp only [← c35337, ← c35338, ← c35339] at e_557_4
  have e_557_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_5
  simp only [← c35340, ← c35341, k2, ← c35342] at e_557_5
  have hr := e_557_5
  simp only [e_557_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 557 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2811)) := by
  have c35343 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35344 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35345 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35346 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35347 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35348 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_557_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_6
  simp only [← c35343, ← c35344, ← c35345] at e_557_6
  have e_557_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_7
  simp only [← c35346, ← c35347, k2, ← c35348] at e_557_7
  have hr := e_557_7
  simp only [e_557_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 557 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2812)) := by
  have c35349 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35350 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35351 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35352 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35353 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35354 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_557_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_8
  simp only [← c35349, ← c35350, ← c35351] at e_557_8
  have e_557_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_9
  simp only [← c35352, ← c35353, k2, ← c35354] at e_557_9
  have hr := e_557_9
  simp only [e_557_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 557 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2813)) := by
  have c35355 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35356 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35357 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35358 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35359 := (publicBatchWrapper86_copies1104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35360 := (publicBatchWrapper86_copies1105 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_557_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_10
  simp only [← c35355, ← c35356, ← c35357] at e_557_10
  have e_557_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_11
  simp only [← c35358, ← c35359, k2, ← c35360] at e_557_11
  have hr := e_557_11
  simp only [e_557_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 557 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2814)) := by
  have c35361 := (publicBatchWrapper86_copies1105 a h).2.1
  have c35362 := (publicBatchWrapper86_copies1105 a h).2.2.1
  have c35363 := (publicBatchWrapper86_copies1105 a h).2.2.2.1
  have c35364 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.1
  have c35365 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.1
  have c35366 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_557_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_12
  simp only [← c35361, ← c35362, ← c35363] at e_557_12
  have e_557_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_13
  simp only [← c35364, ← c35365, k2, ← c35366] at e_557_13
  have hr := e_557_13
  simp only [e_557_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 557 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2815)) := by
  have c35367 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.1
  have c35368 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.1
  have c35369 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.1
  have c35370 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35371 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35372 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_557_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_14
  simp only [← c35367, ← c35368, ← c35369] at e_557_14
  have e_557_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_15
  simp only [← c35370, ← c35371, k2, ← c35372] at e_557_15
  have hr := e_557_15
  simp only [e_557_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 557 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2816)) := by
  have c35373 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35374 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35375 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35376 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35377 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35378 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_557_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_16
  simp only [← c35373, ← c35374, ← c35375] at e_557_16
  have e_557_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_17
  simp only [← c35376, ← c35377, k2, ← c35378] at e_557_17
  have hr := e_557_17
  simp only [e_557_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 557 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2817)) := by
  have c35379 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35380 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35381 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35382 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35383 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35384 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_557_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_18
  simp only [← c35379, ← c35380, ← c35381] at e_557_18
  have e_557_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row557 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_557_19
  simp only [← c35382, ← c35383, k2, ← c35384] at e_557_19
  have hr := e_557_19
  simp only [e_557_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 558 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2818)) := by
  have c35385 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35386 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35387 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35388 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35389 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35390 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_558_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_0
  simp only [← c35385, ← c35386, ← c35387] at e_558_0
  have e_558_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_1
  simp only [← c35388, ← c35389, k2, ← c35390] at e_558_1
  have hr := e_558_1
  simp only [e_558_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 558 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2819)) := by
  have c35391 := (publicBatchWrapper86_copies1105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35392 := (publicBatchWrapper86_copies1106 a h).1
  have c35393 := (publicBatchWrapper86_copies1106 a h).2.1
  have c35394 := (publicBatchWrapper86_copies1106 a h).2.2.1
  have c35395 := (publicBatchWrapper86_copies1106 a h).2.2.2.1
  have c35396 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_558_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_2
  simp only [← c35391, ← c35392, ← c35393] at e_558_2
  have e_558_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_3
  simp only [← c35394, ← c35395, k2, ← c35396] at e_558_3
  have hr := e_558_3
  simp only [e_558_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 558 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2820)) := by
  have c35397 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.1
  have c35398 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.1
  have c35399 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.1
  have c35400 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.1
  have c35401 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.1
  have c35402 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_558_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_4
  simp only [← c35397, ← c35398, ← c35399] at e_558_4
  have e_558_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_5
  simp only [← c35400, ← c35401, k2, ← c35402] at e_558_5
  have hr := e_558_5
  simp only [e_558_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 558 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2821)) := by
  have c35403 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35404 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35405 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35406 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35407 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35408 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_558_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_6
  simp only [← c35403, ← c35404, ← c35405] at e_558_6
  have e_558_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_7
  simp only [← c35406, ← c35407, k2, ← c35408] at e_558_7
  have hr := e_558_7
  simp only [e_558_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 558 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2822)) := by
  have c35409 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35410 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35411 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35412 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35413 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35414 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_558_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_8
  simp only [← c35409, ← c35410, ← c35411] at e_558_8
  have e_558_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_9
  simp only [← c35412, ← c35413, k2, ← c35414] at e_558_9
  have hr := e_558_9
  simp only [e_558_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 558 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2823)) := by
  have c35415 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35416 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35417 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35418 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35419 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35420 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_558_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_10
  simp only [← c35415, ← c35416, ← c35417] at e_558_10
  have e_558_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_11
  simp only [← c35418, ← c35419, k2, ← c35420] at e_558_11
  have hr := e_558_11
  simp only [e_558_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 558 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2824)) := by
  have c35421 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35422 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35423 := (publicBatchWrapper86_copies1106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35424 := (publicBatchWrapper86_copies1107 a h).1
  have c35425 := (publicBatchWrapper86_copies1107 a h).2.1
  have c35426 := (publicBatchWrapper86_copies1107 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_558_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_12
  simp only [← c35421, ← c35422, ← c35423] at e_558_12
  have e_558_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_13
  simp only [← c35424, ← c35425, k2, ← c35426] at e_558_13
  have hr := e_558_13
  simp only [e_558_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 558 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2825)) := by
  have c35427 := (publicBatchWrapper86_copies1107 a h).2.2.2.1
  have c35428 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.1
  have c35429 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.1
  have c35430 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.1
  have c35431 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.1
  have c35432 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_558_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_14
  simp only [← c35427, ← c35428, ← c35429] at e_558_14
  have e_558_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_15
  simp only [← c35430, ← c35431, k2, ← c35432] at e_558_15
  have hr := e_558_15
  simp only [e_558_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 558 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2826)) := by
  have c35433 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.1
  have c35434 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35435 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35436 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35437 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35438 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_558_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_16
  simp only [← c35433, ← c35434, ← c35435] at e_558_16
  have e_558_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_17
  simp only [← c35436, ← c35437, k2, ← c35438] at e_558_17
  have hr := e_558_17
  simp only [e_558_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 558 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2827)) := by
  have c35439 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35440 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35441 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35442 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35443 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35444 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_558_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_18
  simp only [← c35439, ← c35440, ← c35441] at e_558_18
  have e_558_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row558 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_558_19
  simp only [← c35442, ← c35443, k2, ← c35444] at e_558_19
  have hr := e_558_19
  simp only [e_558_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 559 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2828)) := by
  have c35445 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35446 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35447 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35448 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35449 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35450 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_559_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_0
  simp only [← c35445, ← c35446, ← c35447] at e_559_0
  have e_559_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_1
  simp only [← c35448, ← c35449, k2, ← c35450] at e_559_1
  have hr := e_559_1
  simp only [e_559_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 559 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2829)) := by
  have c35451 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35452 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35453 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35454 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35455 := (publicBatchWrapper86_copies1107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35456 := (publicBatchWrapper86_copies1108 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_559_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_2
  simp only [← c35451, ← c35452, ← c35453] at e_559_2
  have e_559_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_3
  simp only [← c35454, ← c35455, k2, ← c35456] at e_559_3
  have hr := e_559_3
  simp only [e_559_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 559 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2830)) := by
  have c35457 := (publicBatchWrapper86_copies1108 a h).2.1
  have c35458 := (publicBatchWrapper86_copies1108 a h).2.2.1
  have c35459 := (publicBatchWrapper86_copies1108 a h).2.2.2.1
  have c35460 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.1
  have c35461 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.1
  have c35462 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_559_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_4
  simp only [← c35457, ← c35458, ← c35459] at e_559_4
  have e_559_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_5
  simp only [← c35460, ← c35461, k2, ← c35462] at e_559_5
  have hr := e_559_5
  simp only [e_559_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 559 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2831)) := by
  have c35463 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.1
  have c35464 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.1
  have c35465 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.1
  have c35466 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35467 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35468 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_559_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_6
  simp only [← c35463, ← c35464, ← c35465] at e_559_6
  have e_559_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_7
  simp only [← c35466, ← c35467, k2, ← c35468] at e_559_7
  have hr := e_559_7
  simp only [e_559_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 559 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2832)) := by
  have c35469 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35470 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35471 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35472 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35473 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35474 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_559_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_8
  simp only [← c35469, ← c35470, ← c35471] at e_559_8
  have e_559_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_9
  simp only [← c35472, ← c35473, k2, ← c35474] at e_559_9
  have hr := e_559_9
  simp only [e_559_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 559 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2833)) := by
  have c35475 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35476 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35477 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35478 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35479 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35480 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_559_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_10
  simp only [← c35475, ← c35476, ← c35477] at e_559_10
  have e_559_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_11
  simp only [← c35478, ← c35479, k2, ← c35480] at e_559_11
  have hr := e_559_11
  simp only [e_559_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 559 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2834)) := by
  have c35481 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35482 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35483 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35484 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35485 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35486 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_559_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_12
  simp only [← c35481, ← c35482, ← c35483] at e_559_12
  have e_559_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_13
  simp only [← c35484, ← c35485, k2, ← c35486] at e_559_13
  have hr := e_559_13
  simp only [e_559_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 559 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2835)) := by
  have c35487 := (publicBatchWrapper86_copies1108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35488 := (publicBatchWrapper86_copies1109 a h).1
  have c35489 := (publicBatchWrapper86_copies1109 a h).2.1
  have c35490 := (publicBatchWrapper86_copies1109 a h).2.2.1
  have c35491 := (publicBatchWrapper86_copies1109 a h).2.2.2.1
  have c35492 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_559_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_14
  simp only [← c35487, ← c35488, ← c35489] at e_559_14
  have e_559_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_15
  simp only [← c35490, ← c35491, k2, ← c35492] at e_559_15
  have hr := e_559_15
  simp only [e_559_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 559 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2836)) := by
  have c35493 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.1
  have c35494 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.1
  have c35495 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.1
  have c35496 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.1
  have c35497 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.1
  have c35498 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_559_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_16
  simp only [← c35493, ← c35494, ← c35495] at e_559_16
  have e_559_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_17
  simp only [← c35496, ← c35497, k2, ← c35498] at e_559_17
  have hr := e_559_17
  simp only [e_559_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 559 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2837)) := by
  have c35499 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35500 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35501 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35502 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35503 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35504 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_559_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_18
  simp only [← c35499, ← c35500, ← c35501] at e_559_18
  have e_559_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_559_19
  simp only [← c35502, ← c35503, k2, ← c35504] at e_559_19
  have hr := e_559_19
  simp only [e_559_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 560 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2838)) := by
  have c35505 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35506 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35507 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35508 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35509 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35510 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_560_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_0
  simp only [← c35505, ← c35506, ← c35507] at e_560_0
  have e_560_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_1
  simp only [← c35508, ← c35509, k2, ← c35510] at e_560_1
  have hr := e_560_1
  simp only [e_560_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 560 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2839)) := by
  have c35511 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35512 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35513 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35514 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35515 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35516 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_560_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_2
  simp only [← c35511, ← c35512, ← c35513] at e_560_2
  have e_560_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_3
  simp only [← c35514, ← c35515, k2, ← c35516] at e_560_3
  have hr := e_560_3
  simp only [e_560_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 560 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2840)) := by
  have c35517 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35518 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35519 := (publicBatchWrapper86_copies1109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35520 := (publicBatchWrapper86_copies1110 a h).1
  have c35521 := (publicBatchWrapper86_copies1110 a h).2.1
  have c35522 := (publicBatchWrapper86_copies1110 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_560_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_4
  simp only [← c35517, ← c35518, ← c35519] at e_560_4
  have e_560_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_5
  simp only [← c35520, ← c35521, k2, ← c35522] at e_560_5
  have hr := e_560_5
  simp only [e_560_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 560 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2841)) := by
  have c35523 := (publicBatchWrapper86_copies1110 a h).2.2.2.1
  have c35524 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.1
  have c35525 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.1
  have c35526 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.1
  have c35527 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.1
  have c35528 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_560_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_6
  simp only [← c35523, ← c35524, ← c35525] at e_560_6
  have e_560_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_7
  simp only [← c35526, ← c35527, k2, ← c35528] at e_560_7
  have hr := e_560_7
  simp only [e_560_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 560 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2842)) := by
  have c35529 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.1
  have c35530 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35531 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35532 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35533 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35534 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_560_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_8
  simp only [← c35529, ← c35530, ← c35531] at e_560_8
  have e_560_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_9
  simp only [← c35532, ← c35533, k2, ← c35534] at e_560_9
  have hr := e_560_9
  simp only [e_560_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 560 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2843)) := by
  have c35535 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35536 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35537 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35538 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35539 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35540 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_560_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_10
  simp only [← c35535, ← c35536, ← c35537] at e_560_10
  have e_560_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_11
  simp only [← c35538, ← c35539, k2, ← c35540] at e_560_11
  have hr := e_560_11
  simp only [e_560_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 560 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2844)) := by
  have c35541 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35542 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35543 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35544 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35545 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35546 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_560_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_12
  simp only [← c35541, ← c35542, ← c35543] at e_560_12
  have e_560_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_13
  simp only [← c35544, ← c35545, k2, ← c35546] at e_560_13
  have hr := e_560_13
  simp only [e_560_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 560 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2845)) := by
  have c35547 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35548 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35549 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35550 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35551 := (publicBatchWrapper86_copies1110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35552 := (publicBatchWrapper86_copies1111 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_560_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_14
  simp only [← c35547, ← c35548, ← c35549] at e_560_14
  have e_560_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_15
  simp only [← c35550, ← c35551, k2, ← c35552] at e_560_15
  have hr := e_560_15
  simp only [e_560_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 560 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2846)) := by
  have c35553 := (publicBatchWrapper86_copies1111 a h).2.1
  have c35554 := (publicBatchWrapper86_copies1111 a h).2.2.1
  have c35555 := (publicBatchWrapper86_copies1111 a h).2.2.2.1
  have c35556 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.1
  have c35557 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.1
  have c35558 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_560_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_16
  simp only [← c35553, ← c35554, ← c35555] at e_560_16
  have e_560_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_17
  simp only [← c35556, ← c35557, k2, ← c35558] at e_560_17
  have hr := e_560_17
  simp only [e_560_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 560 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2847)) := by
  have c35559 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.1
  have c35560 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.1
  have c35561 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.1
  have c35562 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35563 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35564 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_560_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_18
  simp only [← c35559, ← c35560, ← c35561] at e_560_18
  have e_560_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_560_19
  simp only [← c35562, ← c35563, k2, ← c35564] at e_560_19
  have hr := e_560_19
  simp only [e_560_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 561 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2848)) := by
  have c35565 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35566 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35567 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35568 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35569 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35570 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_561_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_0
  simp only [← c35565, ← c35566, ← c35567] at e_561_0
  have e_561_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_1
  simp only [← c35568, ← c35569, k2, ← c35570] at e_561_1
  have hr := e_561_1
  simp only [e_561_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 561 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2849)) := by
  have c35571 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35572 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35573 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35574 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35575 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35576 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_561_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_2
  simp only [← c35571, ← c35572, ← c35573] at e_561_2
  have e_561_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_3
  simp only [← c35574, ← c35575, k2, ← c35576] at e_561_3
  have hr := e_561_3
  simp only [e_561_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 561 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2850)) := by
  have c35577 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35578 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35579 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35580 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35581 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35582 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_561_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_4
  simp only [← c35577, ← c35578, ← c35579] at e_561_4
  have e_561_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_5
  simp only [← c35580, ← c35581, k2, ← c35582] at e_561_5
  have hr := e_561_5
  simp only [e_561_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 561 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2851)) := by
  have c35583 := (publicBatchWrapper86_copies1111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35584 := (publicBatchWrapper86_copies1112 a h).1
  have c35585 := (publicBatchWrapper86_copies1112 a h).2.1
  have c35586 := (publicBatchWrapper86_copies1112 a h).2.2.1
  have c35587 := (publicBatchWrapper86_copies1112 a h).2.2.2.1
  have c35588 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_561_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_6
  simp only [← c35583, ← c35584, ← c35585] at e_561_6
  have e_561_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_7
  simp only [← c35586, ← c35587, k2, ← c35588] at e_561_7
  have hr := e_561_7
  simp only [e_561_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 561 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2852)) := by
  have c35589 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.1
  have c35590 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.1
  have c35591 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.1
  have c35592 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.1
  have c35593 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.1
  have c35594 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_561_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_8
  simp only [← c35589, ← c35590, ← c35591] at e_561_8
  have e_561_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_9
  simp only [← c35592, ← c35593, k2, ← c35594] at e_561_9
  have hr := e_561_9
  simp only [e_561_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 561 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2853)) := by
  have c35595 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35596 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35597 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35598 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35599 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35600 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_561_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_10
  simp only [← c35595, ← c35596, ← c35597] at e_561_10
  have e_561_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_11
  simp only [← c35598, ← c35599, k2, ← c35600] at e_561_11
  have hr := e_561_11
  simp only [e_561_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 561 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2854)) := by
  have c35601 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35602 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35603 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35604 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35605 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35606 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_561_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_12
  simp only [← c35601, ← c35602, ← c35603] at e_561_12
  have e_561_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_13
  simp only [← c35604, ← c35605, k2, ← c35606] at e_561_13
  have hr := e_561_13
  simp only [e_561_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 561 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2855)) := by
  have c35607 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35608 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35609 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35610 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35611 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35612 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_561_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_14
  simp only [← c35607, ← c35608, ← c35609] at e_561_14
  have e_561_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_15
  simp only [← c35610, ← c35611, k2, ← c35612] at e_561_15
  have hr := e_561_15
  simp only [e_561_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 561 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2856)) := by
  have c35613 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35614 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35615 := (publicBatchWrapper86_copies1112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35616 := (publicBatchWrapper86_copies1113 a h).1
  have c35617 := (publicBatchWrapper86_copies1113 a h).2.1
  have c35618 := (publicBatchWrapper86_copies1113 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_561_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_16
  simp only [← c35613, ← c35614, ← c35615] at e_561_16
  have e_561_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_17
  simp only [← c35616, ← c35617, k2, ← c35618] at e_561_17
  have hr := e_561_17
  simp only [e_561_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 561 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2857)) := by
  have c35619 := (publicBatchWrapper86_copies1113 a h).2.2.2.1
  have c35620 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.1
  have c35621 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.1
  have c35622 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.1
  have c35623 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.1
  have c35624 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_561_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_18
  simp only [← c35619, ← c35620, ← c35621] at e_561_18
  have e_561_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_561_19
  simp only [← c35622, ← c35623, k2, ← c35624] at e_561_19
  have hr := e_561_19
  simp only [e_561_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 562 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2858)) := by
  have c35625 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.1
  have c35626 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35627 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35628 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35629 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35630 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_562_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_0
  simp only [← c35625, ← c35626, ← c35627] at e_562_0
  have e_562_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_1
  simp only [← c35628, ← c35629, k2, ← c35630] at e_562_1
  have hr := e_562_1
  simp only [e_562_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 562 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2859)) := by
  have c35631 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35632 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35633 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35634 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35635 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35636 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_562_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_2
  simp only [← c35631, ← c35632, ← c35633] at e_562_2
  have e_562_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_3
  simp only [← c35634, ← c35635, k2, ← c35636] at e_562_3
  have hr := e_562_3
  simp only [e_562_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 562 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2860)) := by
  have c35637 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35638 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35639 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35640 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35641 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35642 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_562_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_4
  simp only [← c35637, ← c35638, ← c35639] at e_562_4
  have e_562_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_5
  simp only [← c35640, ← c35641, k2, ← c35642] at e_562_5
  have hr := e_562_5
  simp only [e_562_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 562 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2861)) := by
  have c35643 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35644 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35645 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35646 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35647 := (publicBatchWrapper86_copies1113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35648 := (publicBatchWrapper86_copies1114 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_562_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_6
  simp only [← c35643, ← c35644, ← c35645] at e_562_6
  have e_562_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_7
  simp only [← c35646, ← c35647, k2, ← c35648] at e_562_7
  have hr := e_562_7
  simp only [e_562_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 562 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2862)) := by
  have c35649 := (publicBatchWrapper86_copies1114 a h).2.1
  have c35650 := (publicBatchWrapper86_copies1114 a h).2.2.1
  have c35651 := (publicBatchWrapper86_copies1114 a h).2.2.2.1
  have c35652 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.1
  have c35653 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.1
  have c35654 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_562_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_8
  simp only [← c35649, ← c35650, ← c35651] at e_562_8
  have e_562_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_9
  simp only [← c35652, ← c35653, k2, ← c35654] at e_562_9
  have hr := e_562_9
  simp only [e_562_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 562 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2863)) := by
  have c35655 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.1
  have c35656 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.1
  have c35657 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.1
  have c35658 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35659 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35660 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_562_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_10
  simp only [← c35655, ← c35656, ← c35657] at e_562_10
  have e_562_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_11
  simp only [← c35658, ← c35659, k2, ← c35660] at e_562_11
  have hr := e_562_11
  simp only [e_562_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 562 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2864)) := by
  have c35661 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35662 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35663 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35664 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35665 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35666 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_562_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_12
  simp only [← c35661, ← c35662, ← c35663] at e_562_12
  have e_562_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_13
  simp only [← c35664, ← c35665, k2, ← c35666] at e_562_13
  have hr := e_562_13
  simp only [e_562_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 562 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2865)) := by
  have c35667 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35668 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35669 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35670 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35671 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35672 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_562_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_14
  simp only [← c35667, ← c35668, ← c35669] at e_562_14
  have e_562_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_15
  simp only [← c35670, ← c35671, k2, ← c35672] at e_562_15
  have hr := e_562_15
  simp only [e_562_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 562 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2866)) := by
  have c35673 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35674 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35675 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35676 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35677 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35678 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_562_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_16
  simp only [← c35673, ← c35674, ← c35675] at e_562_16
  have e_562_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_17
  simp only [← c35676, ← c35677, k2, ← c35678] at e_562_17
  have hr := e_562_17
  simp only [e_562_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 562 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2867)) := by
  have c35679 := (publicBatchWrapper86_copies1114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35680 := (publicBatchWrapper86_copies1115 a h).1
  have c35681 := (publicBatchWrapper86_copies1115 a h).2.1
  have c35682 := (publicBatchWrapper86_copies1115 a h).2.2.1
  have c35683 := (publicBatchWrapper86_copies1115 a h).2.2.2.1
  have c35684 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_562_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_18
  simp only [← c35679, ← c35680, ← c35681] at e_562_18
  have e_562_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_562_19
  simp only [← c35682, ← c35683, k2, ← c35684] at e_562_19
  have hr := e_562_19
  simp only [e_562_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 563 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2868)) := by
  have c35685 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.1
  have c35686 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.1
  have c35687 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.1
  have c35688 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.1
  have c35689 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.1
  have c35690 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_563_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_0
  simp only [← c35685, ← c35686, ← c35687] at e_563_0
  have e_563_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_1
  simp only [← c35688, ← c35689, k2, ← c35690] at e_563_1
  have hr := e_563_1
  simp only [e_563_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 563 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2869)) := by
  have c35691 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35692 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35693 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35694 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35695 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35696 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_563_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_2
  simp only [← c35691, ← c35692, ← c35693] at e_563_2
  have e_563_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_3
  simp only [← c35694, ← c35695, k2, ← c35696] at e_563_3
  have hr := e_563_3
  simp only [e_563_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 563 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2870)) := by
  have c35697 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35698 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35699 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35700 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35701 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35702 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_563_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_4
  simp only [← c35697, ← c35698, ← c35699] at e_563_4
  have e_563_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_5
  simp only [← c35700, ← c35701, k2, ← c35702] at e_563_5
  have hr := e_563_5
  simp only [e_563_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 563 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2871)) := by
  have c35703 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35704 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35705 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35706 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35707 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35708 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_563_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_6
  simp only [← c35703, ← c35704, ← c35705] at e_563_6
  have e_563_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_7
  simp only [← c35706, ← c35707, k2, ← c35708] at e_563_7
  have hr := e_563_7
  simp only [e_563_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 563 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2872)) := by
  have c35709 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35710 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35711 := (publicBatchWrapper86_copies1115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35712 := (publicBatchWrapper86_copies1116 a h).1
  have c35713 := (publicBatchWrapper86_copies1116 a h).2.1
  have c35714 := (publicBatchWrapper86_copies1116 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_563_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_8
  simp only [← c35709, ← c35710, ← c35711] at e_563_8
  have e_563_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_9
  simp only [← c35712, ← c35713, k2, ← c35714] at e_563_9
  have hr := e_563_9
  simp only [e_563_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 563 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2873)) := by
  have c35715 := (publicBatchWrapper86_copies1116 a h).2.2.2.1
  have c35716 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.1
  have c35717 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.1
  have c35718 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.1
  have c35719 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.1
  have c35720 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_563_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_10
  simp only [← c35715, ← c35716, ← c35717] at e_563_10
  have e_563_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_11
  simp only [← c35718, ← c35719, k2, ← c35720] at e_563_11
  have hr := e_563_11
  simp only [e_563_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 563 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2874)) := by
  have c35721 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.1
  have c35722 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35723 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35724 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35725 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35726 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_563_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_12
  simp only [← c35721, ← c35722, ← c35723] at e_563_12
  have e_563_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_13
  simp only [← c35724, ← c35725, k2, ← c35726] at e_563_13
  have hr := e_563_13
  simp only [e_563_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 563 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2875)) := by
  have c35727 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35728 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35729 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35730 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35731 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35732 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_563_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_14
  simp only [← c35727, ← c35728, ← c35729] at e_563_14
  have e_563_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_15
  simp only [← c35730, ← c35731, k2, ← c35732] at e_563_15
  have hr := e_563_15
  simp only [e_563_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 563 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2876)) := by
  have c35733 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35734 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35735 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35736 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35737 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35738 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_563_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_16
  simp only [← c35733, ← c35734, ← c35735] at e_563_16
  have e_563_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_17
  simp only [← c35736, ← c35737, k2, ← c35738] at e_563_17
  have hr := e_563_17
  simp only [e_563_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 563 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2877)) := by
  have c35739 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35740 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35741 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35742 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35743 := (publicBatchWrapper86_copies1116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35744 := (publicBatchWrapper86_copies1117 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_563_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_18
  simp only [← c35739, ← c35740, ← c35741] at e_563_18
  have e_563_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_563_19
  simp only [← c35742, ← c35743, k2, ← c35744] at e_563_19
  have hr := e_563_19
  simp only [e_563_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 564 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2878)) := by
  have c35745 := (publicBatchWrapper86_copies1117 a h).2.1
  have c35746 := (publicBatchWrapper86_copies1117 a h).2.2.1
  have c35747 := (publicBatchWrapper86_copies1117 a h).2.2.2.1
  have c35748 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.1
  have c35749 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.1
  have c35750 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_564_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_0
  simp only [← c35745, ← c35746, ← c35747] at e_564_0
  have e_564_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_1
  simp only [← c35748, ← c35749, k2, ← c35750] at e_564_1
  have hr := e_564_1
  simp only [e_564_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 564 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2879)) := by
  have c35751 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.1
  have c35752 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.1
  have c35753 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.1
  have c35754 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35755 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35756 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_564_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_2
  simp only [← c35751, ← c35752, ← c35753] at e_564_2
  have e_564_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_3
  simp only [← c35754, ← c35755, k2, ← c35756] at e_564_3
  have hr := e_564_3
  simp only [e_564_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 564 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2880)) := by
  have c35757 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35758 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35759 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35760 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35761 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35762 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_564_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_4
  simp only [← c35757, ← c35758, ← c35759] at e_564_4
  have e_564_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_5
  simp only [← c35760, ← c35761, k2, ← c35762] at e_564_5
  have hr := e_564_5
  simp only [e_564_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 564 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2881)) := by
  have c35763 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35764 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35765 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35766 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35767 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35768 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_564_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_6
  simp only [← c35763, ← c35764, ← c35765] at e_564_6
  have e_564_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_7
  simp only [← c35766, ← c35767, k2, ← c35768] at e_564_7
  have hr := e_564_7
  simp only [e_564_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 564 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2882)) := by
  have c35769 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35770 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35771 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35772 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35773 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35774 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_564_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_8
  simp only [← c35769, ← c35770, ← c35771] at e_564_8
  have e_564_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_9
  simp only [← c35772, ← c35773, k2, ← c35774] at e_564_9
  have hr := e_564_9
  simp only [e_564_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 564 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2883)) := by
  have c35775 := (publicBatchWrapper86_copies1117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35776 := (publicBatchWrapper86_copies1118 a h).1
  have c35777 := (publicBatchWrapper86_copies1118 a h).2.1
  have c35778 := (publicBatchWrapper86_copies1118 a h).2.2.1
  have c35779 := (publicBatchWrapper86_copies1118 a h).2.2.2.1
  have c35780 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_564_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_10
  simp only [← c35775, ← c35776, ← c35777] at e_564_10
  have e_564_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_11
  simp only [← c35778, ← c35779, k2, ← c35780] at e_564_11
  have hr := e_564_11
  simp only [e_564_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 564 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2884)) := by
  have c35781 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.1
  have c35782 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.1
  have c35783 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.1
  have c35784 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.1
  have c35785 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.1
  have c35786 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_564_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_12
  simp only [← c35781, ← c35782, ← c35783] at e_564_12
  have e_564_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_13
  simp only [← c35784, ← c35785, k2, ← c35786] at e_564_13
  have hr := e_564_13
  simp only [e_564_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 564 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2885)) := by
  have c35787 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35788 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35789 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35790 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35791 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35792 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_564_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_14
  simp only [← c35787, ← c35788, ← c35789] at e_564_14
  have e_564_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_15
  simp only [← c35790, ← c35791, k2, ← c35792] at e_564_15
  have hr := e_564_15
  simp only [e_564_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 564 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2886)) := by
  have c35793 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35794 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35795 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35796 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35797 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35798 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_564_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_16
  simp only [← c35793, ← c35794, ← c35795] at e_564_16
  have e_564_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_17
  simp only [← c35796, ← c35797, k2, ← c35798] at e_564_17
  have hr := e_564_17
  simp only [e_564_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 564 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2887)) := by
  have c35799 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35800 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35801 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35802 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35803 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35804 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_564_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_18
  simp only [← c35799, ← c35800, ← c35801] at e_564_18
  have e_564_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_564_19
  simp only [← c35802, ← c35803, k2, ← c35804] at e_564_19
  have hr := e_564_19
  simp only [e_564_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 565 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2888)) := by
  have c35805 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35806 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35807 := (publicBatchWrapper86_copies1118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35808 := (publicBatchWrapper86_copies1119 a h).1
  have c35809 := (publicBatchWrapper86_copies1119 a h).2.1
  have c35810 := (publicBatchWrapper86_copies1119 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_565_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_0
  simp only [← c35805, ← c35806, ← c35807] at e_565_0
  have e_565_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_1
  simp only [← c35808, ← c35809, k2, ← c35810] at e_565_1
  have hr := e_565_1
  simp only [e_565_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 565 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2889)) := by
  have c35811 := (publicBatchWrapper86_copies1119 a h).2.2.2.1
  have c35812 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.1
  have c35813 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.1
  have c35814 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.1
  have c35815 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.1
  have c35816 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_565_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_2
  simp only [← c35811, ← c35812, ← c35813] at e_565_2
  have e_565_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_3
  simp only [← c35814, ← c35815, k2, ← c35816] at e_565_3
  have hr := e_565_3
  simp only [e_565_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 565 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2890)) := by
  have c35817 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.1
  have c35818 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35819 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35820 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35821 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35822 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_565_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_4
  simp only [← c35817, ← c35818, ← c35819] at e_565_4
  have e_565_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_5
  simp only [← c35820, ← c35821, k2, ← c35822] at e_565_5
  have hr := e_565_5
  simp only [e_565_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 565 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2891)) := by
  have c35823 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35824 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35825 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35826 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35827 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35828 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_565_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_6
  simp only [← c35823, ← c35824, ← c35825] at e_565_6
  have e_565_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_7
  simp only [← c35826, ← c35827, k2, ← c35828] at e_565_7
  have hr := e_565_7
  simp only [e_565_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 565 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2892)) := by
  have c35829 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35830 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35831 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35832 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35833 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35834 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_565_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_8
  simp only [← c35829, ← c35830, ← c35831] at e_565_8
  have e_565_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_9
  simp only [← c35832, ← c35833, k2, ← c35834] at e_565_9
  have hr := e_565_9
  simp only [e_565_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 565 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2893)) := by
  have c35835 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35836 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35837 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35838 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35839 := (publicBatchWrapper86_copies1119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35840 := (publicBatchWrapper86_copies1120 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_565_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_10
  simp only [← c35835, ← c35836, ← c35837] at e_565_10
  have e_565_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_11
  simp only [← c35838, ← c35839, k2, ← c35840] at e_565_11
  have hr := e_565_11
  simp only [e_565_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 565 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2894)) := by
  have c35841 := (publicBatchWrapper86_copies1120 a h).2.1
  have c35842 := (publicBatchWrapper86_copies1120 a h).2.2.1
  have c35843 := (publicBatchWrapper86_copies1120 a h).2.2.2.1
  have c35844 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.1
  have c35845 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.1
  have c35846 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_565_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_12
  simp only [← c35841, ← c35842, ← c35843] at e_565_12
  have e_565_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_13
  simp only [← c35844, ← c35845, k2, ← c35846] at e_565_13
  have hr := e_565_13
  simp only [e_565_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 565 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2895)) := by
  have c35847 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.1
  have c35848 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.1
  have c35849 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.1
  have c35850 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35851 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35852 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_565_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_14
  simp only [← c35847, ← c35848, ← c35849] at e_565_14
  have e_565_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_15
  simp only [← c35850, ← c35851, k2, ← c35852] at e_565_15
  have hr := e_565_15
  simp only [e_565_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 565 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2896)) := by
  have c35853 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35854 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35855 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35856 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35857 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35858 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_565_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_16
  simp only [← c35853, ← c35854, ← c35855] at e_565_16
  have e_565_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_17
  simp only [← c35856, ← c35857, k2, ← c35858] at e_565_17
  have hr := e_565_17
  simp only [e_565_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 565 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2897)) := by
  have c35859 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35860 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35861 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35862 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35863 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35864 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_565_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_18
  simp only [← c35859, ← c35860, ← c35861] at e_565_18
  have e_565_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_565_19
  simp only [← c35862, ← c35863, k2, ← c35864] at e_565_19
  have hr := e_565_19
  simp only [e_565_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 566 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2898)) := by
  have c35865 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35866 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35867 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35868 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35869 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35870 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_566_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_0
  simp only [← c35865, ← c35866, ← c35867] at e_566_0
  have e_566_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_1
  simp only [← c35868, ← c35869, k2, ← c35870] at e_566_1
  have hr := e_566_1
  simp only [e_566_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 566 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2899)) := by
  have c35871 := (publicBatchWrapper86_copies1120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35872 := (publicBatchWrapper86_copies1121 a h).1
  have c35873 := (publicBatchWrapper86_copies1121 a h).2.1
  have c35874 := (publicBatchWrapper86_copies1121 a h).2.2.1
  have c35875 := (publicBatchWrapper86_copies1121 a h).2.2.2.1
  have c35876 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_566_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_2
  simp only [← c35871, ← c35872, ← c35873] at e_566_2
  have e_566_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_3
  simp only [← c35874, ← c35875, k2, ← c35876] at e_566_3
  have hr := e_566_3
  simp only [e_566_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 566 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2900)) := by
  have c35877 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.1
  have c35878 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.1
  have c35879 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.1
  have c35880 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.1
  have c35881 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.1
  have c35882 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_566_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_4
  simp only [← c35877, ← c35878, ← c35879] at e_566_4
  have e_566_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_5
  simp only [← c35880, ← c35881, k2, ← c35882] at e_566_5
  have hr := e_566_5
  simp only [e_566_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 566 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2901)) := by
  have c35883 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35884 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35885 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35886 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35887 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35888 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_566_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_6
  simp only [← c35883, ← c35884, ← c35885] at e_566_6
  have e_566_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_7
  simp only [← c35886, ← c35887, k2, ← c35888] at e_566_7
  have hr := e_566_7
  simp only [e_566_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 566 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2902)) := by
  have c35889 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35890 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35891 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35892 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35893 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35894 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_566_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_8
  simp only [← c35889, ← c35890, ← c35891] at e_566_8
  have e_566_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_9
  simp only [← c35892, ← c35893, k2, ← c35894] at e_566_9
  have hr := e_566_9
  simp only [e_566_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 566 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2903)) := by
  have c35895 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35896 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35897 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35898 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35899 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35900 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_566_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_10
  simp only [← c35895, ← c35896, ← c35897] at e_566_10
  have e_566_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_11
  simp only [← c35898, ← c35899, k2, ← c35900] at e_566_11
  have hr := e_566_11
  simp only [e_566_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 566 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2904)) := by
  have c35901 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35902 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35903 := (publicBatchWrapper86_copies1121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35904 := (publicBatchWrapper86_copies1122 a h).1
  have c35905 := (publicBatchWrapper86_copies1122 a h).2.1
  have c35906 := (publicBatchWrapper86_copies1122 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_566_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_12
  simp only [← c35901, ← c35902, ← c35903] at e_566_12
  have e_566_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_13
  simp only [← c35904, ← c35905, k2, ← c35906] at e_566_13
  have hr := e_566_13
  simp only [e_566_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 566 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2905)) := by
  have c35907 := (publicBatchWrapper86_copies1122 a h).2.2.2.1
  have c35908 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.1
  have c35909 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.1
  have c35910 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.1
  have c35911 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.1
  have c35912 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_566_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_14
  simp only [← c35907, ← c35908, ← c35909] at e_566_14
  have e_566_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_15
  simp only [← c35910, ← c35911, k2, ← c35912] at e_566_15
  have hr := e_566_15
  simp only [e_566_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 566 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2906)) := by
  have c35913 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.1
  have c35914 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35915 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35916 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35917 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35918 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_566_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_16
  simp only [← c35913, ← c35914, ← c35915] at e_566_16
  have e_566_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_17
  simp only [← c35916, ← c35917, k2, ← c35918] at e_566_17
  have hr := e_566_17
  simp only [e_566_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 566 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2907)) := by
  have c35919 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35920 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35921 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35922 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35923 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35924 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_566_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_18
  simp only [← c35919, ← c35920, ← c35921] at e_566_18
  have e_566_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_566_19
  simp only [← c35922, ← c35923, k2, ← c35924] at e_566_19
  have hr := e_566_19
  simp only [e_566_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 567 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2908)) := by
  have c35925 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35926 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35927 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35928 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35929 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35930 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_567_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_0
  simp only [← c35925, ← c35926, ← c35927] at e_567_0
  have e_567_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_1
  simp only [← c35928, ← c35929, k2, ← c35930] at e_567_1
  have hr := e_567_1
  simp only [e_567_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 567 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2909)) := by
  have c35931 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35932 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35933 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35934 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35935 := (publicBatchWrapper86_copies1122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35936 := (publicBatchWrapper86_copies1123 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_567_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_2
  simp only [← c35931, ← c35932, ← c35933] at e_567_2
  have e_567_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_3
  simp only [← c35934, ← c35935, k2, ← c35936] at e_567_3
  have hr := e_567_3
  simp only [e_567_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 567 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2910)) := by
  have c35937 := (publicBatchWrapper86_copies1123 a h).2.1
  have c35938 := (publicBatchWrapper86_copies1123 a h).2.2.1
  have c35939 := (publicBatchWrapper86_copies1123 a h).2.2.2.1
  have c35940 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.1
  have c35941 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.1
  have c35942 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_567_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_4
  simp only [← c35937, ← c35938, ← c35939] at e_567_4
  have e_567_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_5
  simp only [← c35940, ← c35941, k2, ← c35942] at e_567_5
  have hr := e_567_5
  simp only [e_567_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 567 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2911)) := by
  have c35943 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.1
  have c35944 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.1
  have c35945 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.1
  have c35946 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35947 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35948 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_567_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_6
  simp only [← c35943, ← c35944, ← c35945] at e_567_6
  have e_567_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_7
  simp only [← c35946, ← c35947, k2, ← c35948] at e_567_7
  have hr := e_567_7
  simp only [e_567_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 567 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2912)) := by
  have c35949 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35950 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35951 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35952 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35953 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35954 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_567_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_8
  simp only [← c35949, ← c35950, ← c35951] at e_567_8
  have e_567_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_9
  simp only [← c35952, ← c35953, k2, ← c35954] at e_567_9
  have hr := e_567_9
  simp only [e_567_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 567 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2913)) := by
  have c35955 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35956 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35957 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35958 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35959 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35960 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_567_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_10
  simp only [← c35955, ← c35956, ← c35957] at e_567_10
  have e_567_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_11
  simp only [← c35958, ← c35959, k2, ← c35960] at e_567_11
  have hr := e_567_11
  simp only [e_567_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 567 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2914)) := by
  have c35961 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35962 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35963 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35964 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35965 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35966 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_567_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_12
  simp only [← c35961, ← c35962, ← c35963] at e_567_12
  have e_567_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_13
  simp only [← c35964, ← c35965, k2, ← c35966] at e_567_13
  have hr := e_567_13
  simp only [e_567_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 567 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2915)) := by
  have c35967 := (publicBatchWrapper86_copies1123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35968 := (publicBatchWrapper86_copies1124 a h).1
  have c35969 := (publicBatchWrapper86_copies1124 a h).2.1
  have c35970 := (publicBatchWrapper86_copies1124 a h).2.2.1
  have c35971 := (publicBatchWrapper86_copies1124 a h).2.2.2.1
  have c35972 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_567_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_14
  simp only [← c35967, ← c35968, ← c35969] at e_567_14
  have e_567_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_15
  simp only [← c35970, ← c35971, k2, ← c35972] at e_567_15
  have hr := e_567_15
  simp only [e_567_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 567 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2916)) := by
  have c35973 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.1
  have c35974 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.1
  have c35975 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.1
  have c35976 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.1
  have c35977 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.1
  have c35978 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_567_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_16
  simp only [← c35973, ← c35974, ← c35975] at e_567_16
  have e_567_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_17
  simp only [← c35976, ← c35977, k2, ← c35978] at e_567_17
  have hr := e_567_17
  simp only [e_567_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 567 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2917)) := by
  have c35979 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35980 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35981 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35982 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35983 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35984 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_567_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_18
  simp only [← c35979, ← c35980, ← c35981] at e_567_18
  have e_567_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_567_19
  simp only [← c35982, ← c35983, k2, ← c35984] at e_567_19
  have hr := e_567_19
  simp only [e_567_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 568 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2918)) := by
  have c35985 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35986 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35987 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35988 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35989 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35990 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_568_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_0
  simp only [← c35985, ← c35986, ← c35987] at e_568_0
  have e_568_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_1
  simp only [← c35988, ← c35989, k2, ← c35990] at e_568_1
  have hr := e_568_1
  simp only [e_568_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 568 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2919)) := by
  have c35991 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35992 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35993 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35994 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35995 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35996 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_568_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_2
  simp only [← c35991, ← c35992, ← c35993] at e_568_2
  have e_568_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_3
  simp only [← c35994, ← c35995, k2, ← c35996] at e_568_3
  have hr := e_568_3
  simp only [e_568_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 568 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2920)) := by
  have c35997 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35998 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35999 := (publicBatchWrapper86_copies1124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36000 := (publicBatchWrapper86_copies1125 a h).1
  have c36001 := (publicBatchWrapper86_copies1125 a h).2.1
  have c36002 := (publicBatchWrapper86_copies1125 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_568_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_4
  simp only [← c35997, ← c35998, ← c35999] at e_568_4
  have e_568_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_5
  simp only [← c36000, ← c36001, k2, ← c36002] at e_568_5
  have hr := e_568_5
  simp only [e_568_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 568 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2921)) := by
  have c36003 := (publicBatchWrapper86_copies1125 a h).2.2.2.1
  have c36004 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.1
  have c36005 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.1
  have c36006 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.1
  have c36007 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.1
  have c36008 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_568_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_6
  simp only [← c36003, ← c36004, ← c36005] at e_568_6
  have e_568_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_7
  simp only [← c36006, ← c36007, k2, ← c36008] at e_568_7
  have hr := e_568_7
  simp only [e_568_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 568 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2922)) := by
  have c36009 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.1
  have c36010 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36011 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36012 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36013 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36014 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_568_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_8
  simp only [← c36009, ← c36010, ← c36011] at e_568_8
  have e_568_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_9
  simp only [← c36012, ← c36013, k2, ← c36014] at e_568_9
  have hr := e_568_9
  simp only [e_568_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 568 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2923)) := by
  have c36015 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36016 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36017 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36018 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36019 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36020 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_568_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_10
  simp only [← c36015, ← c36016, ← c36017] at e_568_10
  have e_568_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_11
  simp only [← c36018, ← c36019, k2, ← c36020] at e_568_11
  have hr := e_568_11
  simp only [e_568_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 568 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2924)) := by
  have c36021 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36022 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36023 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36024 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36025 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36026 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_568_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_12
  simp only [← c36021, ← c36022, ← c36023] at e_568_12
  have e_568_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_13
  simp only [← c36024, ← c36025, k2, ← c36026] at e_568_13
  have hr := e_568_13
  simp only [e_568_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 568 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2925)) := by
  have c36027 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36028 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36029 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36030 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36031 := (publicBatchWrapper86_copies1125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36032 := (publicBatchWrapper86_copies1126 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_568_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_14
  simp only [← c36027, ← c36028, ← c36029] at e_568_14
  have e_568_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_15
  simp only [← c36030, ← c36031, k2, ← c36032] at e_568_15
  have hr := e_568_15
  simp only [e_568_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 568 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2926)) := by
  have c36033 := (publicBatchWrapper86_copies1126 a h).2.1
  have c36034 := (publicBatchWrapper86_copies1126 a h).2.2.1
  have c36035 := (publicBatchWrapper86_copies1126 a h).2.2.2.1
  have c36036 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.1
  have c36037 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.1
  have c36038 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_568_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_16
  simp only [← c36033, ← c36034, ← c36035] at e_568_16
  have e_568_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_17
  simp only [← c36036, ← c36037, k2, ← c36038] at e_568_17
  have hr := e_568_17
  simp only [e_568_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 568 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2927)) := by
  have c36039 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.1
  have c36040 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.1
  have c36041 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.1
  have c36042 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36043 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36044 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_568_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_18
  simp only [← c36039, ← c36040, ← c36041] at e_568_18
  have e_568_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_568_19
  simp only [← c36042, ← c36043, k2, ← c36044] at e_568_19
  have hr := e_568_19
  simp only [e_568_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 569 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2928)) := by
  have c36045 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36046 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36047 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36048 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36049 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36050 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_569_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_0
  simp only [← c36045, ← c36046, ← c36047] at e_569_0
  have e_569_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_1
  simp only [← c36048, ← c36049, k2, ← c36050] at e_569_1
  have hr := e_569_1
  simp only [e_569_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 569 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2929)) := by
  have c36051 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36052 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36053 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36054 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36055 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36056 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_569_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_2
  simp only [← c36051, ← c36052, ← c36053] at e_569_2
  have e_569_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_3
  simp only [← c36054, ← c36055, k2, ← c36056] at e_569_3
  have hr := e_569_3
  simp only [e_569_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 569 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2930)) := by
  have c36057 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36058 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36059 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36060 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36061 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36062 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_569_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_4
  simp only [← c36057, ← c36058, ← c36059] at e_569_4
  have e_569_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_5
  simp only [← c36060, ← c36061, k2, ← c36062] at e_569_5
  have hr := e_569_5
  simp only [e_569_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 569 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2931)) := by
  have c36063 := (publicBatchWrapper86_copies1126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36064 := (publicBatchWrapper86_copies1127 a h).1
  have c36065 := (publicBatchWrapper86_copies1127 a h).2.1
  have c36066 := (publicBatchWrapper86_copies1127 a h).2.2.1
  have c36067 := (publicBatchWrapper86_copies1127 a h).2.2.2.1
  have c36068 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_569_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_6
  simp only [← c36063, ← c36064, ← c36065] at e_569_6
  have e_569_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_7
  simp only [← c36066, ← c36067, k2, ← c36068] at e_569_7
  have hr := e_569_7
  simp only [e_569_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 569 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2932)) := by
  have c36069 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.1
  have c36070 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.1
  have c36071 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.1
  have c36072 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.1
  have c36073 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.1
  have c36074 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_569_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_8
  simp only [← c36069, ← c36070, ← c36071] at e_569_8
  have e_569_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_9
  simp only [← c36072, ← c36073, k2, ← c36074] at e_569_9
  have hr := e_569_9
  simp only [e_569_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 569 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2933)) := by
  have c36075 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36076 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36077 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36078 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36079 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36080 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_569_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_10
  simp only [← c36075, ← c36076, ← c36077] at e_569_10
  have e_569_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_11
  simp only [← c36078, ← c36079, k2, ← c36080] at e_569_11
  have hr := e_569_11
  simp only [e_569_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 569 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2934)) := by
  have c36081 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36082 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36083 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36084 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36085 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36086 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_569_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_12
  simp only [← c36081, ← c36082, ← c36083] at e_569_12
  have e_569_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_13
  simp only [← c36084, ← c36085, k2, ← c36086] at e_569_13
  have hr := e_569_13
  simp only [e_569_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 569 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2935)) := by
  have c36087 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36088 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36089 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36090 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36091 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36092 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_569_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_14
  simp only [← c36087, ← c36088, ← c36089] at e_569_14
  have e_569_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_15
  simp only [← c36090, ← c36091, k2, ← c36092] at e_569_15
  have hr := e_569_15
  simp only [e_569_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 569 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2936)) := by
  have c36093 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36094 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36095 := (publicBatchWrapper86_copies1127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36096 := (publicBatchWrapper86_copies1128 a h).1
  have c36097 := (publicBatchWrapper86_copies1128 a h).2.1
  have c36098 := (publicBatchWrapper86_copies1128 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_569_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_16
  simp only [← c36093, ← c36094, ← c36095] at e_569_16
  have e_569_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_17
  simp only [← c36096, ← c36097, k2, ← c36098] at e_569_17
  have hr := e_569_17
  simp only [e_569_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 569 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2937)) := by
  have c36099 := (publicBatchWrapper86_copies1128 a h).2.2.2.1
  have c36100 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.1
  have c36101 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.1
  have c36102 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.1
  have c36103 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.1
  have c36104 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_569_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_18
  simp only [← c36099, ← c36100, ← c36101] at e_569_18
  have e_569_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_569_19
  simp only [← c36102, ← c36103, k2, ← c36104] at e_569_19
  have hr := e_569_19
  simp only [e_569_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 570 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2938)) := by
  have c36105 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.1
  have c36106 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36107 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36108 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36109 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36110 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_570_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_0
  simp only [← c36105, ← c36106, ← c36107] at e_570_0
  have e_570_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_1
  simp only [← c36108, ← c36109, k2, ← c36110] at e_570_1
  have hr := e_570_1
  simp only [e_570_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 570 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2939)) := by
  have c36111 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36112 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36113 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36114 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36115 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36116 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_570_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_2
  simp only [← c36111, ← c36112, ← c36113] at e_570_2
  have e_570_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_3
  simp only [← c36114, ← c36115, k2, ← c36116] at e_570_3
  have hr := e_570_3
  simp only [e_570_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 570 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2940)) := by
  have c36117 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36118 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36119 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36120 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36121 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36122 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_570_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_4
  simp only [← c36117, ← c36118, ← c36119] at e_570_4
  have e_570_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_5
  simp only [← c36120, ← c36121, k2, ← c36122] at e_570_5
  have hr := e_570_5
  simp only [e_570_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 570 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2941)) := by
  have c36123 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36124 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36125 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36126 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36127 := (publicBatchWrapper86_copies1128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36128 := (publicBatchWrapper86_copies1129 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_570_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_6
  simp only [← c36123, ← c36124, ← c36125] at e_570_6
  have e_570_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_7
  simp only [← c36126, ← c36127, k2, ← c36128] at e_570_7
  have hr := e_570_7
  simp only [e_570_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 570 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2942)) := by
  have c36129 := (publicBatchWrapper86_copies1129 a h).2.1
  have c36130 := (publicBatchWrapper86_copies1129 a h).2.2.1
  have c36131 := (publicBatchWrapper86_copies1129 a h).2.2.2.1
  have c36132 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.1
  have c36133 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.1
  have c36134 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_570_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_8
  simp only [← c36129, ← c36130, ← c36131] at e_570_8
  have e_570_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_9
  simp only [← c36132, ← c36133, k2, ← c36134] at e_570_9
  have hr := e_570_9
  simp only [e_570_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 570 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2943)) := by
  have c36135 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.1
  have c36136 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.1
  have c36137 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.1
  have c36138 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36139 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36140 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_570_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_10
  simp only [← c36135, ← c36136, ← c36137] at e_570_10
  have e_570_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_11
  simp only [← c36138, ← c36139, k2, ← c36140] at e_570_11
  have hr := e_570_11
  simp only [e_570_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 570 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2944)) := by
  have c36141 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36142 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36143 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36144 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36145 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36146 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_570_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_12
  simp only [← c36141, ← c36142, ← c36143] at e_570_12
  have e_570_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_13
  simp only [← c36144, ← c36145, k2, ← c36146] at e_570_13
  have hr := e_570_13
  simp only [e_570_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 570 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2945)) := by
  have c36147 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36148 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36149 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36150 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36151 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36152 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_570_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_14
  simp only [← c36147, ← c36148, ← c36149] at e_570_14
  have e_570_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_15
  simp only [← c36150, ← c36151, k2, ← c36152] at e_570_15
  have hr := e_570_15
  simp only [e_570_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 570 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2946)) := by
  have c36153 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36154 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36155 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36156 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36157 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36158 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_570_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_16
  simp only [← c36153, ← c36154, ← c36155] at e_570_16
  have e_570_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_17
  simp only [← c36156, ← c36157, k2, ← c36158] at e_570_17
  have hr := e_570_17
  simp only [e_570_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 570 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2947)) := by
  have c36159 := (publicBatchWrapper86_copies1129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36160 := (publicBatchWrapper86_copies1130 a h).1
  have c36161 := (publicBatchWrapper86_copies1130 a h).2.1
  have c36162 := (publicBatchWrapper86_copies1130 a h).2.2.1
  have c36163 := (publicBatchWrapper86_copies1130 a h).2.2.2.1
  have c36164 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_570_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_18
  simp only [← c36159, ← c36160, ← c36161] at e_570_18
  have e_570_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_570_19
  simp only [← c36162, ← c36163, k2, ← c36164] at e_570_19
  have hr := e_570_19
  simp only [e_570_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 571 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2948)) := by
  have c36165 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.1
  have c36166 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.1
  have c36167 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.1
  have c36168 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.1
  have c36169 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.1
  have c36170 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_571_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_0
  simp only [← c36165, ← c36166, ← c36167] at e_571_0
  have e_571_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_1
  simp only [← c36168, ← c36169, k2, ← c36170] at e_571_1
  have hr := e_571_1
  simp only [e_571_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 571 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2949)) := by
  have c36171 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36172 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36173 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36174 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36175 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36176 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_571_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_2
  simp only [← c36171, ← c36172, ← c36173] at e_571_2
  have e_571_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_3
  simp only [← c36174, ← c36175, k2, ← c36176] at e_571_3
  have hr := e_571_3
  simp only [e_571_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 571 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2950)) := by
  have c36177 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36178 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36179 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36180 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36181 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36182 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_571_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_4
  simp only [← c36177, ← c36178, ← c36179] at e_571_4
  have e_571_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_5
  simp only [← c36180, ← c36181, k2, ← c36182] at e_571_5
  have hr := e_571_5
  simp only [e_571_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 571 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2951)) := by
  have c36183 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36184 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36185 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36186 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36187 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36188 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_571_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_6
  simp only [← c36183, ← c36184, ← c36185] at e_571_6
  have e_571_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_7
  simp only [← c36186, ← c36187, k2, ← c36188] at e_571_7
  have hr := e_571_7
  simp only [e_571_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 571 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2952)) := by
  have c36189 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36190 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36191 := (publicBatchWrapper86_copies1130 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36192 := (publicBatchWrapper86_copies1131 a h).1
  have c36193 := (publicBatchWrapper86_copies1131 a h).2.1
  have c36194 := (publicBatchWrapper86_copies1131 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_571_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_8
  simp only [← c36189, ← c36190, ← c36191] at e_571_8
  have e_571_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_9
  simp only [← c36192, ← c36193, k2, ← c36194] at e_571_9
  have hr := e_571_9
  simp only [e_571_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 571 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2953)) := by
  have c36195 := (publicBatchWrapper86_copies1131 a h).2.2.2.1
  have c36196 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.1
  have c36197 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.1
  have c36198 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.1
  have c36199 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.1
  have c36200 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_571_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_10
  simp only [← c36195, ← c36196, ← c36197] at e_571_10
  have e_571_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_11
  simp only [← c36198, ← c36199, k2, ← c36200] at e_571_11
  have hr := e_571_11
  simp only [e_571_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 571 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2954)) := by
  have c36201 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.1
  have c36202 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36203 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36204 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36205 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36206 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_571_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_12
  simp only [← c36201, ← c36202, ← c36203] at e_571_12
  have e_571_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_13
  simp only [← c36204, ← c36205, k2, ← c36206] at e_571_13
  have hr := e_571_13
  simp only [e_571_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 571 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2955)) := by
  have c36207 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36208 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36209 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36210 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36211 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36212 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_571_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_14
  simp only [← c36207, ← c36208, ← c36209] at e_571_14
  have e_571_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_15
  simp only [← c36210, ← c36211, k2, ← c36212] at e_571_15
  have hr := e_571_15
  simp only [e_571_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 571 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2956)) := by
  have c36213 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36214 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36215 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36216 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36217 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36218 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_571_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_16
  simp only [← c36213, ← c36214, ← c36215] at e_571_16
  have e_571_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_17
  simp only [← c36216, ← c36217, k2, ← c36218] at e_571_17
  have hr := e_571_17
  simp only [e_571_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 571 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2957)) := by
  have c36219 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36220 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36221 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36222 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36223 := (publicBatchWrapper86_copies1131 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36224 := (publicBatchWrapper86_copies1132 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_571_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_18
  simp only [← c36219, ← c36220, ← c36221] at e_571_18
  have e_571_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_571_19
  simp only [← c36222, ← c36223, k2, ← c36224] at e_571_19
  have hr := e_571_19
  simp only [e_571_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 572 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2958)) := by
  have c36225 := (publicBatchWrapper86_copies1132 a h).2.1
  have c36226 := (publicBatchWrapper86_copies1132 a h).2.2.1
  have c36227 := (publicBatchWrapper86_copies1132 a h).2.2.2.1
  have c36228 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.1
  have c36229 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.1
  have c36230 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_572_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_0
  simp only [← c36225, ← c36226, ← c36227] at e_572_0
  have e_572_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_1
  simp only [← c36228, ← c36229, k2, ← c36230] at e_572_1
  have hr := e_572_1
  simp only [e_572_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 572 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2959)) := by
  have c36231 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.1
  have c36232 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.1
  have c36233 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.1
  have c36234 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36235 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36236 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_572_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_2
  simp only [← c36231, ← c36232, ← c36233] at e_572_2
  have e_572_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_3
  simp only [← c36234, ← c36235, k2, ← c36236] at e_572_3
  have hr := e_572_3
  simp only [e_572_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 572 23) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2960)) := by
  have c36237 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36238 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36239 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36240 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36241 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36242 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_572_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_4
  simp only [← c36237, ← c36238, ← c36239] at e_572_4
  have e_572_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_5
  simp only [← c36240, ← c36241, k2, ← c36242] at e_572_5
  have hr := e_572_5
  simp only [e_572_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 572 31) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2961)) := by
  have c36243 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36244 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36245 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36246 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36247 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36248 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_572_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_6
  simp only [← c36243, ← c36244, ← c36245] at e_572_6
  have e_572_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_7
  simp only [← c36246, ← c36247, k2, ← c36248] at e_572_7
  have hr := e_572_7
  simp only [e_572_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 572 39) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2962)) := by
  have c36249 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36250 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36251 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36252 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36253 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36254 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_572_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_8
  simp only [← c36249, ← c36250, ← c36251] at e_572_8
  have e_572_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_9
  simp only [← c36252, ← c36253, k2, ← c36254] at e_572_9
  have hr := e_572_9
  simp only [e_572_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 572 47) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2963)) := by
  have c36255 := (publicBatchWrapper86_copies1132 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36256 := (publicBatchWrapper86_copies1133 a h).1
  have c36257 := (publicBatchWrapper86_copies1133 a h).2.1
  have c36258 := (publicBatchWrapper86_copies1133 a h).2.2.1
  have c36259 := (publicBatchWrapper86_copies1133 a h).2.2.2.1
  have c36260 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_572_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_10
  simp only [← c36255, ← c36256, ← c36257] at e_572_10
  have e_572_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_11
  simp only [← c36258, ← c36259, k2, ← c36260] at e_572_11
  have hr := e_572_11
  simp only [e_572_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 572 55) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2964)) := by
  have c36261 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.1
  have c36262 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.1
  have c36263 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.1
  have c36264 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.1
  have c36265 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.1
  have c36266 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_572_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_12
  simp only [← c36261, ← c36262, ← c36263] at e_572_12
  have e_572_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_13
  simp only [← c36264, ← c36265, k2, ← c36266] at e_572_13
  have hr := e_572_13
  simp only [e_572_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 572 63) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2965)) := by
  have c36267 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36268 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36269 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36270 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36271 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36272 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_572_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_14
  simp only [← c36267, ← c36268, ← c36269] at e_572_14
  have e_572_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_15
  simp only [← c36270, ← c36271, k2, ← c36272] at e_572_15
  have hr := e_572_15
  simp only [e_572_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 572 71) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2966)) := by
  have c36273 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36274 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36275 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36276 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36277 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36278 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_572_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_16
  simp only [← c36273, ← c36274, ← c36275] at e_572_16
  have e_572_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_17
  simp only [← c36276, ← c36277, k2, ← c36278] at e_572_17
  have hr := e_572_17
  simp only [e_572_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 572 79) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2967)) := by
  have c36279 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36280 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36281 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36282 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36283 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36284 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_572_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_18
  simp only [← c36279, ← c36280, ← c36281] at e_572_18
  have e_572_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_572_19
  simp only [← c36282, ← c36283, k2, ← c36284] at e_572_19
  have hr := e_572_19
  simp only [e_572_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 573 7) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2968)) := by
  have c36285 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36286 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36287 := (publicBatchWrapper86_copies1133 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36288 := (publicBatchWrapper86_copies1134 a h).1
  have c36289 := (publicBatchWrapper86_copies1134 a h).2.1
  have c36290 := (publicBatchWrapper86_copies1134 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_573_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_0
  simp only [← c36285, ← c36286, ← c36287] at e_573_0
  have e_573_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_1
  simp only [← c36288, ← c36289, k2, ← c36290] at e_573_1
  have hr := e_573_1
  simp only [e_573_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 573 15) = bselect (a (.wire 16 3)) (a (.virt 200)) (a (.virt 2969)) := by
  have c36291 := (publicBatchWrapper86_copies1134 a h).2.2.2.1
  have c36292 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.1
  have c36293 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.1
  have c36294 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.1
  have c36295 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.1
  have c36296 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_573_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_2
  simp only [← c36291, ← c36292, ← c36293] at e_573_2
  have e_573_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_3
  simp only [← c36294, ← c36295, k2, ← c36296] at e_573_3
  have hr := e_573_3
  simp only [e_573_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 573 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3010)) := by
  have c36297 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.1
  have c36298 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36299 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36300 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36301 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36302 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_573_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_4
  simp only [← c36297, ← c36298, ← c36299] at e_573_4
  have e_573_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_5
  simp only [← c36300, ← c36301, k2, ← c36302] at e_573_5
  have hr := e_573_5
  simp only [e_573_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 573 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3011)) := by
  have c36303 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36304 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36305 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36306 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36307 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36308 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_573_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_6
  simp only [← c36303, ← c36304, ← c36305] at e_573_6
  have e_573_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_7
  simp only [← c36306, ← c36307, k2, ← c36308] at e_573_7
  have hr := e_573_7
  simp only [e_573_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 573 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3012)) := by
  have c36309 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36310 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36311 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36312 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36313 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36314 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_573_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_8
  simp only [← c36309, ← c36310, ← c36311] at e_573_8
  have e_573_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_9
  simp only [← c36312, ← c36313, k2, ← c36314] at e_573_9
  have hr := e_573_9
  simp only [e_573_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 573 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3013)) := by
  have c36315 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36316 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36317 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36318 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36319 := (publicBatchWrapper86_copies1134 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36320 := (publicBatchWrapper86_copies1135 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_573_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_10
  simp only [← c36315, ← c36316, ← c36317] at e_573_10
  have e_573_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_11
  simp only [← c36318, ← c36319, k2, ← c36320] at e_573_11
  have hr := e_573_11
  simp only [e_573_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 573 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3014)) := by
  have c36321 := (publicBatchWrapper86_copies1135 a h).2.1
  have c36322 := (publicBatchWrapper86_copies1135 a h).2.2.1
  have c36323 := (publicBatchWrapper86_copies1135 a h).2.2.2.1
  have c36324 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.1
  have c36325 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.1
  have c36326 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_573_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_12
  simp only [← c36321, ← c36322, ← c36323] at e_573_12
  have e_573_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_13
  simp only [← c36324, ← c36325, k2, ← c36326] at e_573_13
  have hr := e_573_13
  simp only [e_573_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 573 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3015)) := by
  have c36327 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.1
  have c36328 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.1
  have c36329 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.1
  have c36330 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36331 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36332 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_573_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_14
  simp only [← c36327, ← c36328, ← c36329] at e_573_14
  have e_573_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_15
  simp only [← c36330, ← c36331, k2, ← c36332] at e_573_15
  have hr := e_573_15
  simp only [e_573_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 573 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3016)) := by
  have c36333 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36334 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36335 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36336 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36337 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36338 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_573_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_16
  simp only [← c36333, ← c36334, ← c36335] at e_573_16
  have e_573_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_17
  simp only [← c36336, ← c36337, k2, ← c36338] at e_573_17
  have hr := e_573_17
  simp only [e_573_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 573 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3017)) := by
  have c36339 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36340 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36341 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36342 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36343 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36344 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_573_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_18
  simp only [← c36339, ← c36340, ← c36341] at e_573_18
  have e_573_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_573_19
  simp only [← c36342, ← c36343, k2, ← c36344] at e_573_19
  have hr := e_573_19
  simp only [e_573_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 574 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3018)) := by
  have c36345 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36346 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36347 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36348 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36349 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36350 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_574_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_0
  simp only [← c36345, ← c36346, ← c36347] at e_574_0
  have e_574_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_1
  simp only [← c36348, ← c36349, k2, ← c36350] at e_574_1
  have hr := e_574_1
  simp only [e_574_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 574 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3019)) := by
  have c36351 := (publicBatchWrapper86_copies1135 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36352 := (publicBatchWrapper86_copies1136 a h).1
  have c36353 := (publicBatchWrapper86_copies1136 a h).2.1
  have c36354 := (publicBatchWrapper86_copies1136 a h).2.2.1
  have c36355 := (publicBatchWrapper86_copies1136 a h).2.2.2.1
  have c36356 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_574_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_2
  simp only [← c36351, ← c36352, ← c36353] at e_574_2
  have e_574_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_3
  simp only [← c36354, ← c36355, k2, ← c36356] at e_574_3
  have hr := e_574_3
  simp only [e_574_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 574 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3020)) := by
  have c36357 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.1
  have c36358 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.1
  have c36359 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.1
  have c36360 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.1
  have c36361 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.1
  have c36362 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_574_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_4
  simp only [← c36357, ← c36358, ← c36359] at e_574_4
  have e_574_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_5
  simp only [← c36360, ← c36361, k2, ← c36362] at e_574_5
  have hr := e_574_5
  simp only [e_574_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 574 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3021)) := by
  have c36363 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36364 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36365 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36366 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36367 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36368 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_574_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_6
  simp only [← c36363, ← c36364, ← c36365] at e_574_6
  have e_574_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_7
  simp only [← c36366, ← c36367, k2, ← c36368] at e_574_7
  have hr := e_574_7
  simp only [e_574_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 574 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3022)) := by
  have c36369 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36370 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36371 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36372 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36373 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36374 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_574_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_8
  simp only [← c36369, ← c36370, ← c36371] at e_574_8
  have e_574_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_9
  simp only [← c36372, ← c36373, k2, ← c36374] at e_574_9
  have hr := e_574_9
  simp only [e_574_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 574 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3023)) := by
  have c36375 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36376 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36377 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36378 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36379 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36380 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_574_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_10
  simp only [← c36375, ← c36376, ← c36377] at e_574_10
  have e_574_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_11
  simp only [← c36378, ← c36379, k2, ← c36380] at e_574_11
  have hr := e_574_11
  simp only [e_574_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 574 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3024)) := by
  have c36381 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36382 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36383 := (publicBatchWrapper86_copies1136 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36384 := (publicBatchWrapper86_copies1137 a h).1
  have c36385 := (publicBatchWrapper86_copies1137 a h).2.1
  have c36386 := (publicBatchWrapper86_copies1137 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_574_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_12
  simp only [← c36381, ← c36382, ← c36383] at e_574_12
  have e_574_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_13
  simp only [← c36384, ← c36385, k2, ← c36386] at e_574_13
  have hr := e_574_13
  simp only [e_574_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 574 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3025)) := by
  have c36387 := (publicBatchWrapper86_copies1137 a h).2.2.2.1
  have c36388 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.1
  have c36389 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.1
  have c36390 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.1
  have c36391 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.1
  have c36392 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_574_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_14
  simp only [← c36387, ← c36388, ← c36389] at e_574_14
  have e_574_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_15
  simp only [← c36390, ← c36391, k2, ← c36392] at e_574_15
  have hr := e_574_15
  simp only [e_574_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 574 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3026)) := by
  have c36393 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.1
  have c36394 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36395 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36396 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36397 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36398 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_574_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_16
  simp only [← c36393, ← c36394, ← c36395] at e_574_16
  have e_574_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_17
  simp only [← c36396, ← c36397, k2, ← c36398] at e_574_17
  have hr := e_574_17
  simp only [e_574_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 574 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3027)) := by
  have c36399 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36400 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36401 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36402 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36403 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36404 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_574_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_18
  simp only [← c36399, ← c36400, ← c36401] at e_574_18
  have e_574_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_574_19
  simp only [← c36402, ← c36403, k2, ← c36404] at e_574_19
  have hr := e_574_19
  simp only [e_574_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 575 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3028)) := by
  have c36405 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36406 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36407 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36408 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36409 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36410 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_575_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_0
  simp only [← c36405, ← c36406, ← c36407] at e_575_0
  have e_575_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_1
  simp only [← c36408, ← c36409, k2, ← c36410] at e_575_1
  have hr := e_575_1
  simp only [e_575_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 575 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3029)) := by
  have c36411 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36412 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36413 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36414 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36415 := (publicBatchWrapper86_copies1137 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36416 := (publicBatchWrapper86_copies1138 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_575_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_2
  simp only [← c36411, ← c36412, ← c36413] at e_575_2
  have e_575_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_3
  simp only [← c36414, ← c36415, k2, ← c36416] at e_575_3
  have hr := e_575_3
  simp only [e_575_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 575 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3030)) := by
  have c36417 := (publicBatchWrapper86_copies1138 a h).2.1
  have c36418 := (publicBatchWrapper86_copies1138 a h).2.2.1
  have c36419 := (publicBatchWrapper86_copies1138 a h).2.2.2.1
  have c36420 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.1
  have c36421 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.1
  have c36422 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_575_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_4
  simp only [← c36417, ← c36418, ← c36419] at e_575_4
  have e_575_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_5
  simp only [← c36420, ← c36421, k2, ← c36422] at e_575_5
  have hr := e_575_5
  simp only [e_575_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 575 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3031)) := by
  have c36423 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.1
  have c36424 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.1
  have c36425 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.1
  have c36426 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36427 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36428 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_575_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_6
  simp only [← c36423, ← c36424, ← c36425] at e_575_6
  have e_575_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_7
  simp only [← c36426, ← c36427, k2, ← c36428] at e_575_7
  have hr := e_575_7
  simp only [e_575_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 575 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3032)) := by
  have c36429 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36430 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36431 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36432 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36433 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36434 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_575_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_8
  simp only [← c36429, ← c36430, ← c36431] at e_575_8
  have e_575_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_9
  simp only [← c36432, ← c36433, k2, ← c36434] at e_575_9
  have hr := e_575_9
  simp only [e_575_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 575 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3033)) := by
  have c36435 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36436 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36437 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36438 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36439 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36440 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_575_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_10
  simp only [← c36435, ← c36436, ← c36437] at e_575_10
  have e_575_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_11
  simp only [← c36438, ← c36439, k2, ← c36440] at e_575_11
  have hr := e_575_11
  simp only [e_575_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 575 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3034)) := by
  have c36441 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36442 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36443 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36444 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36445 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36446 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_575_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_12
  simp only [← c36441, ← c36442, ← c36443] at e_575_12
  have e_575_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_13
  simp only [← c36444, ← c36445, k2, ← c36446] at e_575_13
  have hr := e_575_13
  simp only [e_575_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 575 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3035)) := by
  have c36447 := (publicBatchWrapper86_copies1138 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36448 := (publicBatchWrapper86_copies1139 a h).1
  have c36449 := (publicBatchWrapper86_copies1139 a h).2.1
  have c36450 := (publicBatchWrapper86_copies1139 a h).2.2.1
  have c36451 := (publicBatchWrapper86_copies1139 a h).2.2.2.1
  have c36452 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_575_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_14
  simp only [← c36447, ← c36448, ← c36449] at e_575_14
  have e_575_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_15
  simp only [← c36450, ← c36451, k2, ← c36452] at e_575_15
  have hr := e_575_15
  simp only [e_575_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 575 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3036)) := by
  have c36453 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.1
  have c36454 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.1
  have c36455 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.1
  have c36456 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.1
  have c36457 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.1
  have c36458 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_575_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_16
  simp only [← c36453, ← c36454, ← c36455] at e_575_16
  have e_575_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_17
  simp only [← c36456, ← c36457, k2, ← c36458] at e_575_17
  have hr := e_575_17
  simp only [e_575_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 575 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3037)) := by
  have c36459 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36460 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36461 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36462 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36463 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36464 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_575_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_18
  simp only [← c36459, ← c36460, ← c36461] at e_575_18
  have e_575_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_575_19
  simp only [← c36462, ← c36463, k2, ← c36464] at e_575_19
  have hr := e_575_19
  simp only [e_575_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 576 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3038)) := by
  have c36465 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36466 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36467 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36468 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36469 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36470 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_576_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_0
  simp only [← c36465, ← c36466, ← c36467] at e_576_0
  have e_576_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_1
  simp only [← c36468, ← c36469, k2, ← c36470] at e_576_1
  have hr := e_576_1
  simp only [e_576_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 576 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3039)) := by
  have c36471 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36472 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36473 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36474 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36475 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36476 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_576_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_2
  simp only [← c36471, ← c36472, ← c36473] at e_576_2
  have e_576_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_3
  simp only [← c36474, ← c36475, k2, ← c36476] at e_576_3
  have hr := e_576_3
  simp only [e_576_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 576 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3040)) := by
  have c36477 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36478 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36479 := (publicBatchWrapper86_copies1139 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36480 := (publicBatchWrapper86_copies1140 a h).1
  have c36481 := (publicBatchWrapper86_copies1140 a h).2.1
  have c36482 := (publicBatchWrapper86_copies1140 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_576_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_4
  simp only [← c36477, ← c36478, ← c36479] at e_576_4
  have e_576_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_5
  simp only [← c36480, ← c36481, k2, ← c36482] at e_576_5
  have hr := e_576_5
  simp only [e_576_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 576 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3041)) := by
  have c36483 := (publicBatchWrapper86_copies1140 a h).2.2.2.1
  have c36484 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.1
  have c36485 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.1
  have c36486 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.1
  have c36487 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.1
  have c36488 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_576_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_6
  simp only [← c36483, ← c36484, ← c36485] at e_576_6
  have e_576_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_7
  simp only [← c36486, ← c36487, k2, ← c36488] at e_576_7
  have hr := e_576_7
  simp only [e_576_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 576 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3042)) := by
  have c36489 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.1
  have c36490 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36491 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36492 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36493 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36494 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_576_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_8
  simp only [← c36489, ← c36490, ← c36491] at e_576_8
  have e_576_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_9
  simp only [← c36492, ← c36493, k2, ← c36494] at e_576_9
  have hr := e_576_9
  simp only [e_576_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 576 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3043)) := by
  have c36495 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36496 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36497 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36498 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36499 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36500 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_576_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_10
  simp only [← c36495, ← c36496, ← c36497] at e_576_10
  have e_576_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_11
  simp only [← c36498, ← c36499, k2, ← c36500] at e_576_11
  have hr := e_576_11
  simp only [e_576_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 576 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3044)) := by
  have c36501 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36502 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36503 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36504 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36505 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36506 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_576_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_12
  simp only [← c36501, ← c36502, ← c36503] at e_576_12
  have e_576_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_13
  simp only [← c36504, ← c36505, k2, ← c36506] at e_576_13
  have hr := e_576_13
  simp only [e_576_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 576 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3045)) := by
  have c36507 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36508 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36509 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36510 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36511 := (publicBatchWrapper86_copies1140 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36512 := (publicBatchWrapper86_copies1141 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_576_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_14
  simp only [← c36507, ← c36508, ← c36509] at e_576_14
  have e_576_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_15
  simp only [← c36510, ← c36511, k2, ← c36512] at e_576_15
  have hr := e_576_15
  simp only [e_576_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 576 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3046)) := by
  have c36513 := (publicBatchWrapper86_copies1141 a h).2.1
  have c36514 := (publicBatchWrapper86_copies1141 a h).2.2.1
  have c36515 := (publicBatchWrapper86_copies1141 a h).2.2.2.1
  have c36516 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.1
  have c36517 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.1
  have c36518 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_576_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_16
  simp only [← c36513, ← c36514, ← c36515] at e_576_16
  have e_576_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_17
  simp only [← c36516, ← c36517, k2, ← c36518] at e_576_17
  have hr := e_576_17
  simp only [e_576_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 576 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3047)) := by
  have c36519 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.1
  have c36520 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.1
  have c36521 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.1
  have c36522 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36523 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36524 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_576_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_18
  simp only [← c36519, ← c36520, ← c36521] at e_576_18
  have e_576_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_576_19
  simp only [← c36522, ← c36523, k2, ← c36524] at e_576_19
  have hr := e_576_19
  simp only [e_576_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 577 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3048)) := by
  have c36525 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36526 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36527 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36528 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36529 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36530 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_577_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_0
  simp only [← c36525, ← c36526, ← c36527] at e_577_0
  have e_577_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_1
  simp only [← c36528, ← c36529, k2, ← c36530] at e_577_1
  have hr := e_577_1
  simp only [e_577_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 577 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3049)) := by
  have c36531 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36532 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36533 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36534 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36535 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36536 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_577_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_2
  simp only [← c36531, ← c36532, ← c36533] at e_577_2
  have e_577_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_3
  simp only [← c36534, ← c36535, k2, ← c36536] at e_577_3
  have hr := e_577_3
  simp only [e_577_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 577 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3050)) := by
  have c36537 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36538 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36539 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36540 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36541 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36542 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_577_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_4
  simp only [← c36537, ← c36538, ← c36539] at e_577_4
  have e_577_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_5
  simp only [← c36540, ← c36541, k2, ← c36542] at e_577_5
  have hr := e_577_5
  simp only [e_577_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 577 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3051)) := by
  have c36543 := (publicBatchWrapper86_copies1141 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36544 := (publicBatchWrapper86_copies1142 a h).1
  have c36545 := (publicBatchWrapper86_copies1142 a h).2.1
  have c36546 := (publicBatchWrapper86_copies1142 a h).2.2.1
  have c36547 := (publicBatchWrapper86_copies1142 a h).2.2.2.1
  have c36548 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_577_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_6
  simp only [← c36543, ← c36544, ← c36545] at e_577_6
  have e_577_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_7
  simp only [← c36546, ← c36547, k2, ← c36548] at e_577_7
  have hr := e_577_7
  simp only [e_577_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 577 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3052)) := by
  have c36549 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.1
  have c36550 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.1
  have c36551 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.1
  have c36552 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.1
  have c36553 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.1
  have c36554 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_577_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_8
  simp only [← c36549, ← c36550, ← c36551] at e_577_8
  have e_577_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_9
  simp only [← c36552, ← c36553, k2, ← c36554] at e_577_9
  have hr := e_577_9
  simp only [e_577_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 577 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3053)) := by
  have c36555 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36556 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36557 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36558 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36559 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36560 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_577_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_10
  simp only [← c36555, ← c36556, ← c36557] at e_577_10
  have e_577_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_11
  simp only [← c36558, ← c36559, k2, ← c36560] at e_577_11
  have hr := e_577_11
  simp only [e_577_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 577 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3054)) := by
  have c36561 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36562 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36563 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36564 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36565 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36566 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_577_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_12
  simp only [← c36561, ← c36562, ← c36563] at e_577_12
  have e_577_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_13
  simp only [← c36564, ← c36565, k2, ← c36566] at e_577_13
  have hr := e_577_13
  simp only [e_577_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 577 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3055)) := by
  have c36567 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36568 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36569 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36570 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36571 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36572 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_577_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_14
  simp only [← c36567, ← c36568, ← c36569] at e_577_14
  have e_577_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_15
  simp only [← c36570, ← c36571, k2, ← c36572] at e_577_15
  have hr := e_577_15
  simp only [e_577_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 577 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3056)) := by
  have c36573 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36574 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36575 := (publicBatchWrapper86_copies1142 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36576 := (publicBatchWrapper86_copies1143 a h).1
  have c36577 := (publicBatchWrapper86_copies1143 a h).2.1
  have c36578 := (publicBatchWrapper86_copies1143 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_577_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_16
  simp only [← c36573, ← c36574, ← c36575] at e_577_16
  have e_577_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_17
  simp only [← c36576, ← c36577, k2, ← c36578] at e_577_17
  have hr := e_577_17
  simp only [e_577_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 577 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3057)) := by
  have c36579 := (publicBatchWrapper86_copies1143 a h).2.2.2.1
  have c36580 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.1
  have c36581 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.1
  have c36582 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.1
  have c36583 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.1
  have c36584 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_577_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_18
  simp only [← c36579, ← c36580, ← c36581] at e_577_18
  have e_577_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_577_19
  simp only [← c36582, ← c36583, k2, ← c36584] at e_577_19
  have hr := e_577_19
  simp only [e_577_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 578 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3058)) := by
  have c36585 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.1
  have c36586 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36587 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36588 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36589 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36590 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_578_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_0
  simp only [← c36585, ← c36586, ← c36587] at e_578_0
  have e_578_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_1
  simp only [← c36588, ← c36589, k2, ← c36590] at e_578_1
  have hr := e_578_1
  simp only [e_578_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 578 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3059)) := by
  have c36591 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36592 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36593 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36594 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36595 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36596 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_578_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_2
  simp only [← c36591, ← c36592, ← c36593] at e_578_2
  have e_578_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_3
  simp only [← c36594, ← c36595, k2, ← c36596] at e_578_3
  have hr := e_578_3
  simp only [e_578_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 578 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3060)) := by
  have c36597 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36598 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36599 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36600 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36601 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36602 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_578_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_4
  simp only [← c36597, ← c36598, ← c36599] at e_578_4
  have e_578_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_5
  simp only [← c36600, ← c36601, k2, ← c36602] at e_578_5
  have hr := e_578_5
  simp only [e_578_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 578 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3061)) := by
  have c36603 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36604 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36605 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36606 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36607 := (publicBatchWrapper86_copies1143 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36608 := (publicBatchWrapper86_copies1144 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_578_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_6
  simp only [← c36603, ← c36604, ← c36605] at e_578_6
  have e_578_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_7
  simp only [← c36606, ← c36607, k2, ← c36608] at e_578_7
  have hr := e_578_7
  simp only [e_578_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 578 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3062)) := by
  have c36609 := (publicBatchWrapper86_copies1144 a h).2.1
  have c36610 := (publicBatchWrapper86_copies1144 a h).2.2.1
  have c36611 := (publicBatchWrapper86_copies1144 a h).2.2.2.1
  have c36612 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.1
  have c36613 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.1
  have c36614 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_578_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_8
  simp only [← c36609, ← c36610, ← c36611] at e_578_8
  have e_578_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_9
  simp only [← c36612, ← c36613, k2, ← c36614] at e_578_9
  have hr := e_578_9
  simp only [e_578_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 578 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3063)) := by
  have c36615 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.1
  have c36616 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.1
  have c36617 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.1
  have c36618 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36619 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36620 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_578_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_10
  simp only [← c36615, ← c36616, ← c36617] at e_578_10
  have e_578_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_11
  simp only [← c36618, ← c36619, k2, ← c36620] at e_578_11
  have hr := e_578_11
  simp only [e_578_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 578 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3064)) := by
  have c36621 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36622 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36623 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36624 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36625 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36626 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_578_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_12
  simp only [← c36621, ← c36622, ← c36623] at e_578_12
  have e_578_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_13
  simp only [← c36624, ← c36625, k2, ← c36626] at e_578_13
  have hr := e_578_13
  simp only [e_578_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 578 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3065)) := by
  have c36627 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36628 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36629 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36630 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36631 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36632 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_578_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_14
  simp only [← c36627, ← c36628, ← c36629] at e_578_14
  have e_578_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_15
  simp only [← c36630, ← c36631, k2, ← c36632] at e_578_15
  have hr := e_578_15
  simp only [e_578_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 578 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3066)) := by
  have c36633 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36634 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36635 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36636 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36637 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36638 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_578_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_16
  simp only [← c36633, ← c36634, ← c36635] at e_578_16
  have e_578_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_17
  simp only [← c36636, ← c36637, k2, ← c36638] at e_578_17
  have hr := e_578_17
  simp only [e_578_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 578 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3067)) := by
  have c36639 := (publicBatchWrapper86_copies1144 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36640 := (publicBatchWrapper86_copies1145 a h).1
  have c36641 := (publicBatchWrapper86_copies1145 a h).2.1
  have c36642 := (publicBatchWrapper86_copies1145 a h).2.2.1
  have c36643 := (publicBatchWrapper86_copies1145 a h).2.2.2.1
  have c36644 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_578_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_18
  simp only [← c36639, ← c36640, ← c36641] at e_578_18
  have e_578_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_578_19
  simp only [← c36642, ← c36643, k2, ← c36644] at e_578_19
  have hr := e_578_19
  simp only [e_578_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 579 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3068)) := by
  have c36645 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.1
  have c36646 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.1
  have c36647 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.1
  have c36648 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.1
  have c36649 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.1
  have c36650 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_579_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_0
  simp only [← c36645, ← c36646, ← c36647] at e_579_0
  have e_579_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_1
  simp only [← c36648, ← c36649, k2, ← c36650] at e_579_1
  have hr := e_579_1
  simp only [e_579_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 579 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3069)) := by
  have c36651 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36652 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36653 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36654 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36655 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36656 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_579_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_2
  simp only [← c36651, ← c36652, ← c36653] at e_579_2
  have e_579_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_3
  simp only [← c36654, ← c36655, k2, ← c36656] at e_579_3
  have hr := e_579_3
  simp only [e_579_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 579 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3070)) := by
  have c36657 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36658 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36659 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36660 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36661 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36662 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_579_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_4
  simp only [← c36657, ← c36658, ← c36659] at e_579_4
  have e_579_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_5
  simp only [← c36660, ← c36661, k2, ← c36662] at e_579_5
  have hr := e_579_5
  simp only [e_579_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 579 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3071)) := by
  have c36663 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36664 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36665 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36666 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36667 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36668 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_579_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_6
  simp only [← c36663, ← c36664, ← c36665] at e_579_6
  have e_579_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_7
  simp only [← c36666, ← c36667, k2, ← c36668] at e_579_7
  have hr := e_579_7
  simp only [e_579_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 579 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3072)) := by
  have c36669 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36670 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36671 := (publicBatchWrapper86_copies1145 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36672 := (publicBatchWrapper86_copies1146 a h).1
  have c36673 := (publicBatchWrapper86_copies1146 a h).2.1
  have c36674 := (publicBatchWrapper86_copies1146 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_579_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_8
  simp only [← c36669, ← c36670, ← c36671] at e_579_8
  have e_579_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_9
  simp only [← c36672, ← c36673, k2, ← c36674] at e_579_9
  have hr := e_579_9
  simp only [e_579_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 579 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3073)) := by
  have c36675 := (publicBatchWrapper86_copies1146 a h).2.2.2.1
  have c36676 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.1
  have c36677 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.1
  have c36678 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.1
  have c36679 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.1
  have c36680 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_579_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_10
  simp only [← c36675, ← c36676, ← c36677] at e_579_10
  have e_579_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_11
  simp only [← c36678, ← c36679, k2, ← c36680] at e_579_11
  have hr := e_579_11
  simp only [e_579_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 579 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3074)) := by
  have c36681 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.1
  have c36682 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36683 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36684 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36685 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36686 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_579_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_12
  simp only [← c36681, ← c36682, ← c36683] at e_579_12
  have e_579_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_13
  simp only [← c36684, ← c36685, k2, ← c36686] at e_579_13
  have hr := e_579_13
  simp only [e_579_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 579 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3075)) := by
  have c36687 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36688 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36689 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36690 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36691 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36692 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_579_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_14
  simp only [← c36687, ← c36688, ← c36689] at e_579_14
  have e_579_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_15
  simp only [← c36690, ← c36691, k2, ← c36692] at e_579_15
  have hr := e_579_15
  simp only [e_579_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 579 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3076)) := by
  have c36693 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36694 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36695 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36696 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36697 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36698 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_579_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_16
  simp only [← c36693, ← c36694, ← c36695] at e_579_16
  have e_579_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_17
  simp only [← c36696, ← c36697, k2, ← c36698] at e_579_17
  have hr := e_579_17
  simp only [e_579_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 579 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3077)) := by
  have c36699 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36700 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36701 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36702 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36703 := (publicBatchWrapper86_copies1146 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36704 := (publicBatchWrapper86_copies1147 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_579_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_18
  simp only [← c36699, ← c36700, ← c36701] at e_579_18
  have e_579_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_579_19
  simp only [← c36702, ← c36703, k2, ← c36704] at e_579_19
  have hr := e_579_19
  simp only [e_579_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 580 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3078)) := by
  have c36705 := (publicBatchWrapper86_copies1147 a h).2.1
  have c36706 := (publicBatchWrapper86_copies1147 a h).2.2.1
  have c36707 := (publicBatchWrapper86_copies1147 a h).2.2.2.1
  have c36708 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.1
  have c36709 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.1
  have c36710 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_580_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_0
  simp only [← c36705, ← c36706, ← c36707] at e_580_0
  have e_580_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_1
  simp only [← c36708, ← c36709, k2, ← c36710] at e_580_1
  have hr := e_580_1
  simp only [e_580_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 580 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3079)) := by
  have c36711 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.1
  have c36712 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.1
  have c36713 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.1
  have c36714 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36715 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36716 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_580_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_2
  simp only [← c36711, ← c36712, ← c36713] at e_580_2
  have e_580_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_3
  simp only [← c36714, ← c36715, k2, ← c36716] at e_580_3
  have hr := e_580_3
  simp only [e_580_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 580 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3080)) := by
  have c36717 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36718 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36719 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36720 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36721 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36722 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_580_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_4
  simp only [← c36717, ← c36718, ← c36719] at e_580_4
  have e_580_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_5
  simp only [← c36720, ← c36721, k2, ← c36722] at e_580_5
  have hr := e_580_5
  simp only [e_580_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 580 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3081)) := by
  have c36723 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36724 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36725 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36726 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36727 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36728 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_580_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_6
  simp only [← c36723, ← c36724, ← c36725] at e_580_6
  have e_580_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_7
  simp only [← c36726, ← c36727, k2, ← c36728] at e_580_7
  have hr := e_580_7
  simp only [e_580_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 580 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3082)) := by
  have c36729 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36730 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36731 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36732 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36733 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36734 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_580_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_8
  simp only [← c36729, ← c36730, ← c36731] at e_580_8
  have e_580_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_9
  simp only [← c36732, ← c36733, k2, ← c36734] at e_580_9
  have hr := e_580_9
  simp only [e_580_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 580 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3083)) := by
  have c36735 := (publicBatchWrapper86_copies1147 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36736 := (publicBatchWrapper86_copies1148 a h).1
  have c36737 := (publicBatchWrapper86_copies1148 a h).2.1
  have c36738 := (publicBatchWrapper86_copies1148 a h).2.2.1
  have c36739 := (publicBatchWrapper86_copies1148 a h).2.2.2.1
  have c36740 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_580_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_10
  simp only [← c36735, ← c36736, ← c36737] at e_580_10
  have e_580_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_11
  simp only [← c36738, ← c36739, k2, ← c36740] at e_580_11
  have hr := e_580_11
  simp only [e_580_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 580 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3084)) := by
  have c36741 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.1
  have c36742 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.1
  have c36743 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.1
  have c36744 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.1
  have c36745 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.1
  have c36746 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_580_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_12
  simp only [← c36741, ← c36742, ← c36743] at e_580_12
  have e_580_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_13
  simp only [← c36744, ← c36745, k2, ← c36746] at e_580_13
  have hr := e_580_13
  simp only [e_580_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 580 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3085)) := by
  have c36747 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36748 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36749 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36750 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36751 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36752 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_580_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_14
  simp only [← c36747, ← c36748, ← c36749] at e_580_14
  have e_580_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_15
  simp only [← c36750, ← c36751, k2, ← c36752] at e_580_15
  have hr := e_580_15
  simp only [e_580_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 580 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3086)) := by
  have c36753 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36754 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36755 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36756 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36757 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36758 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_580_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_16
  simp only [← c36753, ← c36754, ← c36755] at e_580_16
  have e_580_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_17
  simp only [← c36756, ← c36757, k2, ← c36758] at e_580_17
  have hr := e_580_17
  simp only [e_580_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 580 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3087)) := by
  have c36759 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36760 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36761 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36762 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36763 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36764 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_580_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_18
  simp only [← c36759, ← c36760, ← c36761] at e_580_18
  have e_580_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_580_19
  simp only [← c36762, ← c36763, k2, ← c36764] at e_580_19
  have hr := e_580_19
  simp only [e_580_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 581 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3088)) := by
  have c36765 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36766 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36767 := (publicBatchWrapper86_copies1148 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36768 := (publicBatchWrapper86_copies1149 a h).1
  have c36769 := (publicBatchWrapper86_copies1149 a h).2.1
  have c36770 := (publicBatchWrapper86_copies1149 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_581_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_0
  simp only [← c36765, ← c36766, ← c36767] at e_581_0
  have e_581_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_1
  simp only [← c36768, ← c36769, k2, ← c36770] at e_581_1
  have hr := e_581_1
  simp only [e_581_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 581 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3089)) := by
  have c36771 := (publicBatchWrapper86_copies1149 a h).2.2.2.1
  have c36772 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.1
  have c36773 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.1
  have c36774 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.1
  have c36775 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.1
  have c36776 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_581_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_2
  simp only [← c36771, ← c36772, ← c36773] at e_581_2
  have e_581_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_3
  simp only [← c36774, ← c36775, k2, ← c36776] at e_581_3
  have hr := e_581_3
  simp only [e_581_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 581 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3090)) := by
  have c36777 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.1
  have c36778 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36779 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36780 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36781 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36782 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_581_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_4
  simp only [← c36777, ← c36778, ← c36779] at e_581_4
  have e_581_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_5
  simp only [← c36780, ← c36781, k2, ← c36782] at e_581_5
  have hr := e_581_5
  simp only [e_581_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 581 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3091)) := by
  have c36783 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36784 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36785 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36786 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36787 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36788 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_581_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_6
  simp only [← c36783, ← c36784, ← c36785] at e_581_6
  have e_581_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_7
  simp only [← c36786, ← c36787, k2, ← c36788] at e_581_7
  have hr := e_581_7
  simp only [e_581_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 581 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3092)) := by
  have c36789 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36790 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36791 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36792 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36793 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36794 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_581_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_8
  simp only [← c36789, ← c36790, ← c36791] at e_581_8
  have e_581_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_9
  simp only [← c36792, ← c36793, k2, ← c36794] at e_581_9
  have hr := e_581_9
  simp only [e_581_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 581 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3093)) := by
  have c36795 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36796 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36797 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36798 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36799 := (publicBatchWrapper86_copies1149 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36800 := (publicBatchWrapper86_copies1150 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_581_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_10
  simp only [← c36795, ← c36796, ← c36797] at e_581_10
  have e_581_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_11
  simp only [← c36798, ← c36799, k2, ← c36800] at e_581_11
  have hr := e_581_11
  simp only [e_581_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 581 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3094)) := by
  have c36801 := (publicBatchWrapper86_copies1150 a h).2.1
  have c36802 := (publicBatchWrapper86_copies1150 a h).2.2.1
  have c36803 := (publicBatchWrapper86_copies1150 a h).2.2.2.1
  have c36804 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.1
  have c36805 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.1
  have c36806 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_581_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_12
  simp only [← c36801, ← c36802, ← c36803] at e_581_12
  have e_581_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_13
  simp only [← c36804, ← c36805, k2, ← c36806] at e_581_13
  have hr := e_581_13
  simp only [e_581_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 581 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3095)) := by
  have c36807 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.1
  have c36808 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.1
  have c36809 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.1
  have c36810 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36811 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36812 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_581_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_14
  simp only [← c36807, ← c36808, ← c36809] at e_581_14
  have e_581_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_15
  simp only [← c36810, ← c36811, k2, ← c36812] at e_581_15
  have hr := e_581_15
  simp only [e_581_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 581 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3096)) := by
  have c36813 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36814 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36815 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36816 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36817 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36818 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_581_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_16
  simp only [← c36813, ← c36814, ← c36815] at e_581_16
  have e_581_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_17
  simp only [← c36816, ← c36817, k2, ← c36818] at e_581_17
  have hr := e_581_17
  simp only [e_581_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 581 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3097)) := by
  have c36819 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36820 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36821 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36822 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36823 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36824 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_581_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_18
  simp only [← c36819, ← c36820, ← c36821] at e_581_18
  have e_581_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_581_19
  simp only [← c36822, ← c36823, k2, ← c36824] at e_581_19
  have hr := e_581_19
  simp only [e_581_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 582 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3098)) := by
  have c36825 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36826 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36827 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36828 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36829 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36830 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_582_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_0
  simp only [← c36825, ← c36826, ← c36827] at e_582_0
  have e_582_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_1
  simp only [← c36828, ← c36829, k2, ← c36830] at e_582_1
  have hr := e_582_1
  simp only [e_582_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 582 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3099)) := by
  have c36831 := (publicBatchWrapper86_copies1150 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36832 := (publicBatchWrapper86_copies1151 a h).1
  have c36833 := (publicBatchWrapper86_copies1151 a h).2.1
  have c36834 := (publicBatchWrapper86_copies1151 a h).2.2.1
  have c36835 := (publicBatchWrapper86_copies1151 a h).2.2.2.1
  have c36836 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_582_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_2
  simp only [← c36831, ← c36832, ← c36833] at e_582_2
  have e_582_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_3
  simp only [← c36834, ← c36835, k2, ← c36836] at e_582_3
  have hr := e_582_3
  simp only [e_582_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 582 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3100)) := by
  have c36837 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.1
  have c36838 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.1
  have c36839 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.1
  have c36840 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.1
  have c36841 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.1
  have c36842 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_582_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_4
  simp only [← c36837, ← c36838, ← c36839] at e_582_4
  have e_582_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_5
  simp only [← c36840, ← c36841, k2, ← c36842] at e_582_5
  have hr := e_582_5
  simp only [e_582_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 582 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3101)) := by
  have c36843 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36844 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36845 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36846 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36847 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36848 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_582_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_6
  simp only [← c36843, ← c36844, ← c36845] at e_582_6
  have e_582_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_7
  simp only [← c36846, ← c36847, k2, ← c36848] at e_582_7
  have hr := e_582_7
  simp only [e_582_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 582 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3102)) := by
  have c36849 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36850 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36851 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36852 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36853 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36854 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_582_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_8
  simp only [← c36849, ← c36850, ← c36851] at e_582_8
  have e_582_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_9
  simp only [← c36852, ← c36853, k2, ← c36854] at e_582_9
  have hr := e_582_9
  simp only [e_582_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 582 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3103)) := by
  have c36855 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36856 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36857 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36858 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36859 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36860 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_582_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_10
  simp only [← c36855, ← c36856, ← c36857] at e_582_10
  have e_582_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_11
  simp only [← c36858, ← c36859, k2, ← c36860] at e_582_11
  have hr := e_582_11
  simp only [e_582_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 582 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3104)) := by
  have c36861 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36862 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36863 := (publicBatchWrapper86_copies1151 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36864 := (publicBatchWrapper86_copies1152 a h).1
  have c36865 := (publicBatchWrapper86_copies1152 a h).2.1
  have c36866 := (publicBatchWrapper86_copies1152 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_582_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_12
  simp only [← c36861, ← c36862, ← c36863] at e_582_12
  have e_582_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_13
  simp only [← c36864, ← c36865, k2, ← c36866] at e_582_13
  have hr := e_582_13
  simp only [e_582_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 582 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3105)) := by
  have c36867 := (publicBatchWrapper86_copies1152 a h).2.2.2.1
  have c36868 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.1
  have c36869 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.1
  have c36870 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.1
  have c36871 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.1
  have c36872 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_582_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_14
  simp only [← c36867, ← c36868, ← c36869] at e_582_14
  have e_582_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_15
  simp only [← c36870, ← c36871, k2, ← c36872] at e_582_15
  have hr := e_582_15
  simp only [e_582_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 582 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3106)) := by
  have c36873 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.1
  have c36874 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36875 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36876 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36877 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36878 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_582_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_16
  simp only [← c36873, ← c36874, ← c36875] at e_582_16
  have e_582_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_17
  simp only [← c36876, ← c36877, k2, ← c36878] at e_582_17
  have hr := e_582_17
  simp only [e_582_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 582 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3107)) := by
  have c36879 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36880 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36881 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36882 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36883 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36884 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_582_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_18
  simp only [← c36879, ← c36880, ← c36881] at e_582_18
  have e_582_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_582_19
  simp only [← c36882, ← c36883, k2, ← c36884] at e_582_19
  have hr := e_582_19
  simp only [e_582_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 583 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3108)) := by
  have c36885 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36886 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36887 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36888 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36889 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36890 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_583_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_0
  simp only [← c36885, ← c36886, ← c36887] at e_583_0
  have e_583_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_1
  simp only [← c36888, ← c36889, k2, ← c36890] at e_583_1
  have hr := e_583_1
  simp only [e_583_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 583 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3109)) := by
  have c36891 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36892 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36893 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36894 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36895 := (publicBatchWrapper86_copies1152 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36896 := (publicBatchWrapper86_copies1153 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_583_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_2
  simp only [← c36891, ← c36892, ← c36893] at e_583_2
  have e_583_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_3
  simp only [← c36894, ← c36895, k2, ← c36896] at e_583_3
  have hr := e_583_3
  simp only [e_583_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 583 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3110)) := by
  have c36897 := (publicBatchWrapper86_copies1153 a h).2.1
  have c36898 := (publicBatchWrapper86_copies1153 a h).2.2.1
  have c36899 := (publicBatchWrapper86_copies1153 a h).2.2.2.1
  have c36900 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.1
  have c36901 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.1
  have c36902 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_583_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_4
  simp only [← c36897, ← c36898, ← c36899] at e_583_4
  have e_583_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_5
  simp only [← c36900, ← c36901, k2, ← c36902] at e_583_5
  have hr := e_583_5
  simp only [e_583_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 583 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3111)) := by
  have c36903 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.1
  have c36904 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.1
  have c36905 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.1
  have c36906 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36907 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36908 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_583_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_6
  simp only [← c36903, ← c36904, ← c36905] at e_583_6
  have e_583_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_7
  simp only [← c36906, ← c36907, k2, ← c36908] at e_583_7
  have hr := e_583_7
  simp only [e_583_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 583 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3112)) := by
  have c36909 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36910 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36911 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36912 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36913 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36914 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_583_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_8
  simp only [← c36909, ← c36910, ← c36911] at e_583_8
  have e_583_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_9
  simp only [← c36912, ← c36913, k2, ← c36914] at e_583_9
  have hr := e_583_9
  simp only [e_583_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 583 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3113)) := by
  have c36915 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36916 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36917 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36918 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36919 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36920 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_583_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_10
  simp only [← c36915, ← c36916, ← c36917] at e_583_10
  have e_583_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_11
  simp only [← c36918, ← c36919, k2, ← c36920] at e_583_11
  have hr := e_583_11
  simp only [e_583_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 583 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3114)) := by
  have c36921 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36922 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36923 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36924 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36925 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36926 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_583_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_12
  simp only [← c36921, ← c36922, ← c36923] at e_583_12
  have e_583_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_13
  simp only [← c36924, ← c36925, k2, ← c36926] at e_583_13
  have hr := e_583_13
  simp only [e_583_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 583 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3115)) := by
  have c36927 := (publicBatchWrapper86_copies1153 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36928 := (publicBatchWrapper86_copies1154 a h).1
  have c36929 := (publicBatchWrapper86_copies1154 a h).2.1
  have c36930 := (publicBatchWrapper86_copies1154 a h).2.2.1
  have c36931 := (publicBatchWrapper86_copies1154 a h).2.2.2.1
  have c36932 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_583_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_14
  simp only [← c36927, ← c36928, ← c36929] at e_583_14
  have e_583_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_15
  simp only [← c36930, ← c36931, k2, ← c36932] at e_583_15
  have hr := e_583_15
  simp only [e_583_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 583 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3116)) := by
  have c36933 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.1
  have c36934 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.1
  have c36935 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.1
  have c36936 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.1
  have c36937 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.1
  have c36938 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_583_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_16
  simp only [← c36933, ← c36934, ← c36935] at e_583_16
  have e_583_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_17
  simp only [← c36936, ← c36937, k2, ← c36938] at e_583_17
  have hr := e_583_17
  simp only [e_583_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 583 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3117)) := by
  have c36939 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36940 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36941 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36942 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36943 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36944 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_583_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_18
  simp only [← c36939, ← c36940, ← c36941] at e_583_18
  have e_583_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_583_19
  simp only [← c36942, ← c36943, k2, ← c36944] at e_583_19
  have hr := e_583_19
  simp only [e_583_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 584 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3118)) := by
  have c36945 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36946 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36947 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36948 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36949 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36950 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_584_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_0
  simp only [← c36945, ← c36946, ← c36947] at e_584_0
  have e_584_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_1
  simp only [← c36948, ← c36949, k2, ← c36950] at e_584_1
  have hr := e_584_1
  simp only [e_584_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 584 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3119)) := by
  have c36951 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36952 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36953 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36954 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36955 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36956 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_584_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_2
  simp only [← c36951, ← c36952, ← c36953] at e_584_2
  have e_584_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_3
  simp only [← c36954, ← c36955, k2, ← c36956] at e_584_3
  have hr := e_584_3
  simp only [e_584_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 584 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3120)) := by
  have c36957 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36958 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36959 := (publicBatchWrapper86_copies1154 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36960 := (publicBatchWrapper86_copies1155 a h).1
  have c36961 := (publicBatchWrapper86_copies1155 a h).2.1
  have c36962 := (publicBatchWrapper86_copies1155 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_584_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_4
  simp only [← c36957, ← c36958, ← c36959] at e_584_4
  have e_584_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_5
  simp only [← c36960, ← c36961, k2, ← c36962] at e_584_5
  have hr := e_584_5
  simp only [e_584_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 584 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3121)) := by
  have c36963 := (publicBatchWrapper86_copies1155 a h).2.2.2.1
  have c36964 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.1
  have c36965 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.1
  have c36966 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.1
  have c36967 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.1
  have c36968 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_584_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_6
  simp only [← c36963, ← c36964, ← c36965] at e_584_6
  have e_584_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_7
  simp only [← c36966, ← c36967, k2, ← c36968] at e_584_7
  have hr := e_584_7
  simp only [e_584_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 584 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3122)) := by
  have c36969 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.1
  have c36970 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.1
  have c36971 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c36972 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36973 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36974 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_584_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_8
  simp only [← c36969, ← c36970, ← c36971] at e_584_8
  have e_584_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_9
  simp only [← c36972, ← c36973, k2, ← c36974] at e_584_9
  have hr := e_584_9
  simp only [e_584_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 584 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3123)) := by
  have c36975 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36976 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36977 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36978 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36979 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36980 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_584_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_10
  simp only [← c36975, ← c36976, ← c36977] at e_584_10
  have e_584_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_11
  simp only [← c36978, ← c36979, k2, ← c36980] at e_584_11
  have hr := e_584_11
  simp only [e_584_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 584 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3124)) := by
  have c36981 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36982 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36983 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36984 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36985 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36986 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_584_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_12
  simp only [← c36981, ← c36982, ← c36983] at e_584_12
  have e_584_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_13
  simp only [← c36984, ← c36985, k2, ← c36986] at e_584_13
  have hr := e_584_13
  simp only [e_584_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 584 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3125)) := by
  have c36987 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36988 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36989 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36990 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c36991 := (publicBatchWrapper86_copies1155 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c36992 := (publicBatchWrapper86_copies1156 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_584_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_14
  simp only [← c36987, ← c36988, ← c36989] at e_584_14
  have e_584_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_15
  simp only [← c36990, ← c36991, k2, ← c36992] at e_584_15
  have hr := e_584_15
  simp only [e_584_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 584 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3126)) := by
  have c36993 := (publicBatchWrapper86_copies1156 a h).2.1
  have c36994 := (publicBatchWrapper86_copies1156 a h).2.2.1
  have c36995 := (publicBatchWrapper86_copies1156 a h).2.2.2.1
  have c36996 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.1
  have c36997 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.1
  have c36998 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_584_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_16
  simp only [← c36993, ← c36994, ← c36995] at e_584_16
  have e_584_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_17
  simp only [← c36996, ← c36997, k2, ← c36998] at e_584_17
  have hr := e_584_17
  simp only [e_584_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 584 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3127)) := by
  have c36999 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.1
  have c37000 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.1
  have c37001 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.1
  have c37002 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37003 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37004 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_584_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_18
  simp only [← c36999, ← c37000, ← c37001] at e_584_18
  have e_584_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_584_19
  simp only [← c37002, ← c37003, k2, ← c37004] at e_584_19
  have hr := e_584_19
  simp only [e_584_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 585 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3128)) := by
  have c37005 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37006 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37007 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37008 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37009 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37010 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_585_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_0
  simp only [← c37005, ← c37006, ← c37007] at e_585_0
  have e_585_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_1
  simp only [← c37008, ← c37009, k2, ← c37010] at e_585_1
  have hr := e_585_1
  simp only [e_585_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 585 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3129)) := by
  have c37011 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37012 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37013 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37014 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37015 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37016 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_585_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_2
  simp only [← c37011, ← c37012, ← c37013] at e_585_2
  have e_585_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_3
  simp only [← c37014, ← c37015, k2, ← c37016] at e_585_3
  have hr := e_585_3
  simp only [e_585_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 585 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3130)) := by
  have c37017 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37018 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37019 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37020 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37021 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37022 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_585_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_4
  simp only [← c37017, ← c37018, ← c37019] at e_585_4
  have e_585_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_5
  simp only [← c37020, ← c37021, k2, ← c37022] at e_585_5
  have hr := e_585_5
  simp only [e_585_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 585 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3131)) := by
  have c37023 := (publicBatchWrapper86_copies1156 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37024 := (publicBatchWrapper86_copies1157 a h).1
  have c37025 := (publicBatchWrapper86_copies1157 a h).2.1
  have c37026 := (publicBatchWrapper86_copies1157 a h).2.2.1
  have c37027 := (publicBatchWrapper86_copies1157 a h).2.2.2.1
  have c37028 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_585_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_6
  simp only [← c37023, ← c37024, ← c37025] at e_585_6
  have e_585_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_7
  simp only [← c37026, ← c37027, k2, ← c37028] at e_585_7
  have hr := e_585_7
  simp only [e_585_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 585 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3132)) := by
  have c37029 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.1
  have c37030 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.1
  have c37031 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.1
  have c37032 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.1
  have c37033 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.1
  have c37034 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_585_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_8
  simp only [← c37029, ← c37030, ← c37031] at e_585_8
  have e_585_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_9
  simp only [← c37032, ← c37033, k2, ← c37034] at e_585_9
  have hr := e_585_9
  simp only [e_585_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 585 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3133)) := by
  have c37035 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37036 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37037 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37038 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37039 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37040 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_585_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_10
  simp only [← c37035, ← c37036, ← c37037] at e_585_10
  have e_585_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_11
  simp only [← c37038, ← c37039, k2, ← c37040] at e_585_11
  have hr := e_585_11
  simp only [e_585_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 585 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3134)) := by
  have c37041 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37042 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37043 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37044 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37045 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37046 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_585_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_12
  simp only [← c37041, ← c37042, ← c37043] at e_585_12
  have e_585_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_13
  simp only [← c37044, ← c37045, k2, ← c37046] at e_585_13
  have hr := e_585_13
  simp only [e_585_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 585 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3135)) := by
  have c37047 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37048 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37049 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37050 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37051 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37052 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_585_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_14
  simp only [← c37047, ← c37048, ← c37049] at e_585_14
  have e_585_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_15
  simp only [← c37050, ← c37051, k2, ← c37052] at e_585_15
  have hr := e_585_15
  simp only [e_585_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 585 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3136)) := by
  have c37053 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37054 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37055 := (publicBatchWrapper86_copies1157 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37056 := (publicBatchWrapper86_copies1158 a h).1
  have c37057 := (publicBatchWrapper86_copies1158 a h).2.1
  have c37058 := (publicBatchWrapper86_copies1158 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_585_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_16
  simp only [← c37053, ← c37054, ← c37055] at e_585_16
  have e_585_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_17
  simp only [← c37056, ← c37057, k2, ← c37058] at e_585_17
  have hr := e_585_17
  simp only [e_585_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 585 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3137)) := by
  have c37059 := (publicBatchWrapper86_copies1158 a h).2.2.2.1
  have c37060 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.1
  have c37061 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.1
  have c37062 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.1
  have c37063 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.1
  have c37064 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_585_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_18
  simp only [← c37059, ← c37060, ← c37061] at e_585_18
  have e_585_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_585_19
  simp only [← c37062, ← c37063, k2, ← c37064] at e_585_19
  have hr := e_585_19
  simp only [e_585_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 586 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3138)) := by
  have c37065 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.1
  have c37066 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37067 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37068 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37069 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37070 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_586_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_0
  simp only [← c37065, ← c37066, ← c37067] at e_586_0
  have e_586_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_1
  simp only [← c37068, ← c37069, k2, ← c37070] at e_586_1
  have hr := e_586_1
  simp only [e_586_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 586 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3139)) := by
  have c37071 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37072 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37073 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37074 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37075 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37076 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_586_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_2
  simp only [← c37071, ← c37072, ← c37073] at e_586_2
  have e_586_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_3
  simp only [← c37074, ← c37075, k2, ← c37076] at e_586_3
  have hr := e_586_3
  simp only [e_586_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 586 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3140)) := by
  have c37077 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37078 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37079 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37080 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37081 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37082 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_586_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_4
  simp only [← c37077, ← c37078, ← c37079] at e_586_4
  have e_586_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_5
  simp only [← c37080, ← c37081, k2, ← c37082] at e_586_5
  have hr := e_586_5
  simp only [e_586_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 586 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3141)) := by
  have c37083 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37084 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37085 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37086 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37087 := (publicBatchWrapper86_copies1158 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37088 := (publicBatchWrapper86_copies1159 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_586_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_6
  simp only [← c37083, ← c37084, ← c37085] at e_586_6
  have e_586_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_7
  simp only [← c37086, ← c37087, k2, ← c37088] at e_586_7
  have hr := e_586_7
  simp only [e_586_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 586 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3142)) := by
  have c37089 := (publicBatchWrapper86_copies1159 a h).2.1
  have c37090 := (publicBatchWrapper86_copies1159 a h).2.2.1
  have c37091 := (publicBatchWrapper86_copies1159 a h).2.2.2.1
  have c37092 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.1
  have c37093 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.1
  have c37094 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_586_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_8
  simp only [← c37089, ← c37090, ← c37091] at e_586_8
  have e_586_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_9
  simp only [← c37092, ← c37093, k2, ← c37094] at e_586_9
  have hr := e_586_9
  simp only [e_586_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 586 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3143)) := by
  have c37095 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.1
  have c37096 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.1
  have c37097 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.1
  have c37098 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37099 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37100 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_586_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_10
  simp only [← c37095, ← c37096, ← c37097] at e_586_10
  have e_586_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_11
  simp only [← c37098, ← c37099, k2, ← c37100] at e_586_11
  have hr := e_586_11
  simp only [e_586_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 586 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3144)) := by
  have c37101 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37102 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37103 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37104 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37105 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37106 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_586_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_12
  simp only [← c37101, ← c37102, ← c37103] at e_586_12
  have e_586_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_13
  simp only [← c37104, ← c37105, k2, ← c37106] at e_586_13
  have hr := e_586_13
  simp only [e_586_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 586 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3145)) := by
  have c37107 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37108 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37109 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37110 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37111 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37112 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_586_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_14
  simp only [← c37107, ← c37108, ← c37109] at e_586_14
  have e_586_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_15
  simp only [← c37110, ← c37111, k2, ← c37112] at e_586_15
  have hr := e_586_15
  simp only [e_586_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 586 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3146)) := by
  have c37113 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37114 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37115 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37116 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37117 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37118 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_586_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_16
  simp only [← c37113, ← c37114, ← c37115] at e_586_16
  have e_586_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_17
  simp only [← c37116, ← c37117, k2, ← c37118] at e_586_17
  have hr := e_586_17
  simp only [e_586_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 586 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3147)) := by
  have c37119 := (publicBatchWrapper86_copies1159 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37120 := (publicBatchWrapper86_copies1160 a h).1
  have c37121 := (publicBatchWrapper86_copies1160 a h).2.1
  have c37122 := (publicBatchWrapper86_copies1160 a h).2.2.1
  have c37123 := (publicBatchWrapper86_copies1160 a h).2.2.2.1
  have c37124 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_586_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_18
  simp only [← c37119, ← c37120, ← c37121] at e_586_18
  have e_586_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_586_19
  simp only [← c37122, ← c37123, k2, ← c37124] at e_586_19
  have hr := e_586_19
  simp only [e_586_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 587 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3148)) := by
  have c37125 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.1
  have c37126 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.1
  have c37127 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.1
  have c37128 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.1
  have c37129 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.1
  have c37130 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_587_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_0
  simp only [← c37125, ← c37126, ← c37127] at e_587_0
  have e_587_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_1
  simp only [← c37128, ← c37129, k2, ← c37130] at e_587_1
  have hr := e_587_1
  simp only [e_587_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 587 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3149)) := by
  have c37131 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37132 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37133 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37134 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37135 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37136 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_587_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_2
  simp only [← c37131, ← c37132, ← c37133] at e_587_2
  have e_587_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_3
  simp only [← c37134, ← c37135, k2, ← c37136] at e_587_3
  have hr := e_587_3
  simp only [e_587_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 587 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3150)) := by
  have c37137 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37138 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37139 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37140 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37141 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37142 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_587_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_4
  simp only [← c37137, ← c37138, ← c37139] at e_587_4
  have e_587_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_5
  simp only [← c37140, ← c37141, k2, ← c37142] at e_587_5
  have hr := e_587_5
  simp only [e_587_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 587 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3151)) := by
  have c37143 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37144 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37145 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37146 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37147 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37148 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_587_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_6
  simp only [← c37143, ← c37144, ← c37145] at e_587_6
  have e_587_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_7
  simp only [← c37146, ← c37147, k2, ← c37148] at e_587_7
  have hr := e_587_7
  simp only [e_587_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 587 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3152)) := by
  have c37149 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37150 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37151 := (publicBatchWrapper86_copies1160 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37152 := (publicBatchWrapper86_copies1161 a h).1
  have c37153 := (publicBatchWrapper86_copies1161 a h).2.1
  have c37154 := (publicBatchWrapper86_copies1161 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_587_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_8
  simp only [← c37149, ← c37150, ← c37151] at e_587_8
  have e_587_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_9
  simp only [← c37152, ← c37153, k2, ← c37154] at e_587_9
  have hr := e_587_9
  simp only [e_587_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 587 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3153)) := by
  have c37155 := (publicBatchWrapper86_copies1161 a h).2.2.2.1
  have c37156 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.1
  have c37157 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.1
  have c37158 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.1
  have c37159 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.1
  have c37160 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_587_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_10
  simp only [← c37155, ← c37156, ← c37157] at e_587_10
  have e_587_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_11
  simp only [← c37158, ← c37159, k2, ← c37160] at e_587_11
  have hr := e_587_11
  simp only [e_587_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 587 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3154)) := by
  have c37161 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.1
  have c37162 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37163 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37164 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37165 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37166 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_587_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_12
  simp only [← c37161, ← c37162, ← c37163] at e_587_12
  have e_587_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_13
  simp only [← c37164, ← c37165, k2, ← c37166] at e_587_13
  have hr := e_587_13
  simp only [e_587_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 587 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3155)) := by
  have c37167 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37168 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37169 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37170 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37171 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37172 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_587_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_14
  simp only [← c37167, ← c37168, ← c37169] at e_587_14
  have e_587_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_15
  simp only [← c37170, ← c37171, k2, ← c37172] at e_587_15
  have hr := e_587_15
  simp only [e_587_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 587 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3156)) := by
  have c37173 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37174 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37175 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37176 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37177 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37178 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_587_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_16
  simp only [← c37173, ← c37174, ← c37175] at e_587_16
  have e_587_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_17
  simp only [← c37176, ← c37177, k2, ← c37178] at e_587_17
  have hr := e_587_17
  simp only [e_587_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 587 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3157)) := by
  have c37179 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37180 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37181 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37182 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37183 := (publicBatchWrapper86_copies1161 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37184 := (publicBatchWrapper86_copies1162 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_587_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_18
  simp only [← c37179, ← c37180, ← c37181] at e_587_18
  have e_587_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_587_19
  simp only [← c37182, ← c37183, k2, ← c37184] at e_587_19
  have hr := e_587_19
  simp only [e_587_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 588 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3158)) := by
  have c37185 := (publicBatchWrapper86_copies1162 a h).2.1
  have c37186 := (publicBatchWrapper86_copies1162 a h).2.2.1
  have c37187 := (publicBatchWrapper86_copies1162 a h).2.2.2.1
  have c37188 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.1
  have c37189 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.1
  have c37190 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_588_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_0
  simp only [← c37185, ← c37186, ← c37187] at e_588_0
  have e_588_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_1
  simp only [← c37188, ← c37189, k2, ← c37190] at e_588_1
  have hr := e_588_1
  simp only [e_588_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 588 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3159)) := by
  have c37191 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.1
  have c37192 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.1
  have c37193 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.1
  have c37194 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37195 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37196 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_588_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_2
  simp only [← c37191, ← c37192, ← c37193] at e_588_2
  have e_588_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_3
  simp only [← c37194, ← c37195, k2, ← c37196] at e_588_3
  have hr := e_588_3
  simp only [e_588_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 588 23) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3160)) := by
  have c37197 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37198 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37199 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37200 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37201 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37202 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_588_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_4
  simp only [← c37197, ← c37198, ← c37199] at e_588_4
  have e_588_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_5
  simp only [← c37200, ← c37201, k2, ← c37202] at e_588_5
  have hr := e_588_5
  simp only [e_588_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 588 31) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3161)) := by
  have c37203 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37204 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37205 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37206 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37207 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37208 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_588_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_6
  simp only [← c37203, ← c37204, ← c37205] at e_588_6
  have e_588_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_7
  simp only [← c37206, ← c37207, k2, ← c37208] at e_588_7
  have hr := e_588_7
  simp only [e_588_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 588 39) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3162)) := by
  have c37209 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37210 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37211 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37212 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37213 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37214 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_588_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_8
  simp only [← c37209, ← c37210, ← c37211] at e_588_8
  have e_588_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_9
  simp only [← c37212, ← c37213, k2, ← c37214] at e_588_9
  have hr := e_588_9
  simp only [e_588_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 588 47) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3163)) := by
  have c37215 := (publicBatchWrapper86_copies1162 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37216 := (publicBatchWrapper86_copies1163 a h).1
  have c37217 := (publicBatchWrapper86_copies1163 a h).2.1
  have c37218 := (publicBatchWrapper86_copies1163 a h).2.2.1
  have c37219 := (publicBatchWrapper86_copies1163 a h).2.2.2.1
  have c37220 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_588_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_10
  simp only [← c37215, ← c37216, ← c37217] at e_588_10
  have e_588_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_11
  simp only [← c37218, ← c37219, k2, ← c37220] at e_588_11
  have hr := e_588_11
  simp only [e_588_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 588 55) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3164)) := by
  have c37221 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.1
  have c37222 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.1
  have c37223 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.1
  have c37224 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.1
  have c37225 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.1
  have c37226 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_588_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_12
  simp only [← c37221, ← c37222, ← c37223] at e_588_12
  have e_588_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_13
  simp only [← c37224, ← c37225, k2, ← c37226] at e_588_13
  have hr := e_588_13
  simp only [e_588_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 588 63) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3165)) := by
  have c37227 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37228 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37229 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37230 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37231 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37232 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_588_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_14
  simp only [← c37227, ← c37228, ← c37229] at e_588_14
  have e_588_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_15
  simp only [← c37230, ← c37231, k2, ← c37232] at e_588_15
  have hr := e_588_15
  simp only [e_588_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 588 71) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3166)) := by
  have c37233 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37234 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37235 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37236 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37237 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37238 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_588_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_16
  simp only [← c37233, ← c37234, ← c37235] at e_588_16
  have e_588_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_17
  simp only [← c37236, ← c37237, k2, ← c37238] at e_588_17
  have hr := e_588_17
  simp only [e_588_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 588 79) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3167)) := by
  have c37239 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37240 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37241 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37242 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37243 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37244 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_588_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_18
  simp only [← c37239, ← c37240, ← c37241] at e_588_18
  have e_588_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_588_19
  simp only [← c37242, ← c37243, k2, ← c37244] at e_588_19
  have hr := e_588_19
  simp only [e_588_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 589 7) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3168)) := by
  have c37245 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37246 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37247 := (publicBatchWrapper86_copies1163 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37248 := (publicBatchWrapper86_copies1164 a h).1
  have c37249 := (publicBatchWrapper86_copies1164 a h).2.1
  have c37250 := (publicBatchWrapper86_copies1164 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_589_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_0
  simp only [← c37245, ← c37246, ← c37247] at e_589_0
  have e_589_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_1
  simp only [← c37248, ← c37249, k2, ← c37250] at e_589_1
  have hr := e_589_1
  simp only [e_589_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 589 15) = bselect (a (.wire 16 47)) (a (.virt 200)) (a (.virt 3169)) := by
  have c37251 := (publicBatchWrapper86_copies1164 a h).2.2.2.1
  have c37252 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.1
  have c37253 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.1
  have c37254 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.1
  have c37255 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.1
  have c37256 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_589_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_2
  simp only [← c37251, ← c37252, ← c37253] at e_589_2
  have e_589_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_3
  simp only [← c37254, ← c37255, k2, ← c37256] at e_589_3
  have hr := e_589_3
  simp only [e_589_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 589 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3210)) := by
  have c37257 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.1
  have c37258 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37259 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37260 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37261 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37262 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_589_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_4
  simp only [← c37257, ← c37258, ← c37259] at e_589_4
  have e_589_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_5
  simp only [← c37260, ← c37261, k2, ← c37262] at e_589_5
  have hr := e_589_5
  simp only [e_589_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 589 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3211)) := by
  have c37263 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37264 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37265 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37266 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37267 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37268 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_589_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_6
  simp only [← c37263, ← c37264, ← c37265] at e_589_6
  have e_589_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_7
  simp only [← c37266, ← c37267, k2, ← c37268] at e_589_7
  have hr := e_589_7
  simp only [e_589_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 589 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3212)) := by
  have c37269 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37270 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37271 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37272 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37273 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37274 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_589_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_8
  simp only [← c37269, ← c37270, ← c37271] at e_589_8
  have e_589_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_9
  simp only [← c37272, ← c37273, k2, ← c37274] at e_589_9
  have hr := e_589_9
  simp only [e_589_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 589 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3213)) := by
  have c37275 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37276 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37277 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37278 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37279 := (publicBatchWrapper86_copies1164 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37280 := (publicBatchWrapper86_copies1165 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_589_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_10
  simp only [← c37275, ← c37276, ← c37277] at e_589_10
  have e_589_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_11
  simp only [← c37278, ← c37279, k2, ← c37280] at e_589_11
  have hr := e_589_11
  simp only [e_589_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 589 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3214)) := by
  have c37281 := (publicBatchWrapper86_copies1165 a h).2.1
  have c37282 := (publicBatchWrapper86_copies1165 a h).2.2.1
  have c37283 := (publicBatchWrapper86_copies1165 a h).2.2.2.1
  have c37284 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.1
  have c37285 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.1
  have c37286 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_589_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_12
  simp only [← c37281, ← c37282, ← c37283] at e_589_12
  have e_589_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_13
  simp only [← c37284, ← c37285, k2, ← c37286] at e_589_13
  have hr := e_589_13
  simp only [e_589_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 589 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3215)) := by
  have c37287 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.1
  have c37288 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.1
  have c37289 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.1
  have c37290 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37291 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37292 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_589_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_14
  simp only [← c37287, ← c37288, ← c37289] at e_589_14
  have e_589_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_15
  simp only [← c37290, ← c37291, k2, ← c37292] at e_589_15
  have hr := e_589_15
  simp only [e_589_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 589 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3216)) := by
  have c37293 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37294 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37295 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37296 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37297 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37298 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_589_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_16
  simp only [← c37293, ← c37294, ← c37295] at e_589_16
  have e_589_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_17
  simp only [← c37296, ← c37297, k2, ← c37298] at e_589_17
  have hr := e_589_17
  simp only [e_589_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 589 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3217)) := by
  have c37299 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37300 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37301 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37302 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37303 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37304 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_589_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_18
  simp only [← c37299, ← c37300, ← c37301] at e_589_18
  have e_589_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_589_19
  simp only [← c37302, ← c37303, k2, ← c37304] at e_589_19
  have hr := e_589_19
  simp only [e_589_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 590 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3218)) := by
  have c37305 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37306 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37307 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37308 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37309 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37310 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_590_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_0
  simp only [← c37305, ← c37306, ← c37307] at e_590_0
  have e_590_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_1
  simp only [← c37308, ← c37309, k2, ← c37310] at e_590_1
  have hr := e_590_1
  simp only [e_590_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 590 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3219)) := by
  have c37311 := (publicBatchWrapper86_copies1165 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37312 := (publicBatchWrapper86_copies1166 a h).1
  have c37313 := (publicBatchWrapper86_copies1166 a h).2.1
  have c37314 := (publicBatchWrapper86_copies1166 a h).2.2.1
  have c37315 := (publicBatchWrapper86_copies1166 a h).2.2.2.1
  have c37316 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_590_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_2
  simp only [← c37311, ← c37312, ← c37313] at e_590_2
  have e_590_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_3
  simp only [← c37314, ← c37315, k2, ← c37316] at e_590_3
  have hr := e_590_3
  simp only [e_590_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 590 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3220)) := by
  have c37317 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.1
  have c37318 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.1
  have c37319 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.1
  have c37320 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.1
  have c37321 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.1
  have c37322 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_590_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_4
  simp only [← c37317, ← c37318, ← c37319] at e_590_4
  have e_590_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_5
  simp only [← c37320, ← c37321, k2, ← c37322] at e_590_5
  have hr := e_590_5
  simp only [e_590_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 590 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3221)) := by
  have c37323 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37324 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37325 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37326 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37327 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37328 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_590_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_6
  simp only [← c37323, ← c37324, ← c37325] at e_590_6
  have e_590_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_7
  simp only [← c37326, ← c37327, k2, ← c37328] at e_590_7
  have hr := e_590_7
  simp only [e_590_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 590 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3222)) := by
  have c37329 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37330 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37331 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37332 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37333 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37334 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_590_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_8
  simp only [← c37329, ← c37330, ← c37331] at e_590_8
  have e_590_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_9
  simp only [← c37332, ← c37333, k2, ← c37334] at e_590_9
  have hr := e_590_9
  simp only [e_590_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 590 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3223)) := by
  have c37335 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37336 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37337 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37338 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37339 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37340 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_590_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_10
  simp only [← c37335, ← c37336, ← c37337] at e_590_10
  have e_590_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_11
  simp only [← c37338, ← c37339, k2, ← c37340] at e_590_11
  have hr := e_590_11
  simp only [e_590_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 590 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3224)) := by
  have c37341 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37342 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37343 := (publicBatchWrapper86_copies1166 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37344 := (publicBatchWrapper86_copies1167 a h).1
  have c37345 := (publicBatchWrapper86_copies1167 a h).2.1
  have c37346 := (publicBatchWrapper86_copies1167 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_590_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_12
  simp only [← c37341, ← c37342, ← c37343] at e_590_12
  have e_590_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_13
  simp only [← c37344, ← c37345, k2, ← c37346] at e_590_13
  have hr := e_590_13
  simp only [e_590_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 590 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3225)) := by
  have c37347 := (publicBatchWrapper86_copies1167 a h).2.2.2.1
  have c37348 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.1
  have c37349 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.1
  have c37350 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.1
  have c37351 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.1
  have c37352 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_590_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_14
  simp only [← c37347, ← c37348, ← c37349] at e_590_14
  have e_590_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_15
  simp only [← c37350, ← c37351, k2, ← c37352] at e_590_15
  have hr := e_590_15
  simp only [e_590_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 590 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3226)) := by
  have c37353 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.1
  have c37354 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37355 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37356 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37357 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37358 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_590_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_16
  simp only [← c37353, ← c37354, ← c37355] at e_590_16
  have e_590_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_17
  simp only [← c37356, ← c37357, k2, ← c37358] at e_590_17
  have hr := e_590_17
  simp only [e_590_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 590 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3227)) := by
  have c37359 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37360 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37361 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37362 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37363 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37364 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_590_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_18
  simp only [← c37359, ← c37360, ← c37361] at e_590_18
  have e_590_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_590_19
  simp only [← c37362, ← c37363, k2, ← c37364] at e_590_19
  have hr := e_590_19
  simp only [e_590_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 591 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3228)) := by
  have c37365 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37366 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37367 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37368 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37369 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37370 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_591_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_0
  simp only [← c37365, ← c37366, ← c37367] at e_591_0
  have e_591_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_1
  simp only [← c37368, ← c37369, k2, ← c37370] at e_591_1
  have hr := e_591_1
  simp only [e_591_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 591 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3229)) := by
  have c37371 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37372 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37373 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37374 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37375 := (publicBatchWrapper86_copies1167 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37376 := (publicBatchWrapper86_copies1168 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_591_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_2
  simp only [← c37371, ← c37372, ← c37373] at e_591_2
  have e_591_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_3
  simp only [← c37374, ← c37375, k2, ← c37376] at e_591_3
  have hr := e_591_3
  simp only [e_591_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 591 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3230)) := by
  have c37377 := (publicBatchWrapper86_copies1168 a h).2.1
  have c37378 := (publicBatchWrapper86_copies1168 a h).2.2.1
  have c37379 := (publicBatchWrapper86_copies1168 a h).2.2.2.1
  have c37380 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.1
  have c37381 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.1
  have c37382 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_591_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_4
  simp only [← c37377, ← c37378, ← c37379] at e_591_4
  have e_591_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_5
  simp only [← c37380, ← c37381, k2, ← c37382] at e_591_5
  have hr := e_591_5
  simp only [e_591_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 591 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3231)) := by
  have c37383 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.1
  have c37384 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.1
  have c37385 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.1
  have c37386 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37387 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37388 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_591_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_6
  simp only [← c37383, ← c37384, ← c37385] at e_591_6
  have e_591_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_7
  simp only [← c37386, ← c37387, k2, ← c37388] at e_591_7
  have hr := e_591_7
  simp only [e_591_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 591 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3232)) := by
  have c37389 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37390 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37391 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37392 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37393 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37394 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_591_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_8
  simp only [← c37389, ← c37390, ← c37391] at e_591_8
  have e_591_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_9
  simp only [← c37392, ← c37393, k2, ← c37394] at e_591_9
  have hr := e_591_9
  simp only [e_591_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 591 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3233)) := by
  have c37395 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37396 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37397 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37398 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37399 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37400 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_591_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_10
  simp only [← c37395, ← c37396, ← c37397] at e_591_10
  have e_591_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_11
  simp only [← c37398, ← c37399, k2, ← c37400] at e_591_11
  have hr := e_591_11
  simp only [e_591_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 591 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3234)) := by
  have c37401 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37402 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37403 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37404 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37405 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37406 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_591_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_12
  simp only [← c37401, ← c37402, ← c37403] at e_591_12
  have e_591_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_13
  simp only [← c37404, ← c37405, k2, ← c37406] at e_591_13
  have hr := e_591_13
  simp only [e_591_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 591 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3235)) := by
  have c37407 := (publicBatchWrapper86_copies1168 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37408 := (publicBatchWrapper86_copies1169 a h).1
  have c37409 := (publicBatchWrapper86_copies1169 a h).2.1
  have c37410 := (publicBatchWrapper86_copies1169 a h).2.2.1
  have c37411 := (publicBatchWrapper86_copies1169 a h).2.2.2.1
  have c37412 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_591_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_14
  simp only [← c37407, ← c37408, ← c37409] at e_591_14
  have e_591_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_15
  simp only [← c37410, ← c37411, k2, ← c37412] at e_591_15
  have hr := e_591_15
  simp only [e_591_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 591 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3236)) := by
  have c37413 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.1
  have c37414 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.1
  have c37415 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.1
  have c37416 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.1
  have c37417 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.1
  have c37418 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_591_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_16
  simp only [← c37413, ← c37414, ← c37415] at e_591_16
  have e_591_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_17
  simp only [← c37416, ← c37417, k2, ← c37418] at e_591_17
  have hr := e_591_17
  simp only [e_591_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 591 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3237)) := by
  have c37419 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37420 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37421 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37422 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37423 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37424 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_591_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_18
  simp only [← c37419, ← c37420, ← c37421] at e_591_18
  have e_591_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_591_19
  simp only [← c37422, ← c37423, k2, ← c37424] at e_591_19
  have hr := e_591_19
  simp only [e_591_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 592 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3238)) := by
  have c37425 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37426 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37427 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37428 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37429 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37430 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_592_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_0
  simp only [← c37425, ← c37426, ← c37427] at e_592_0
  have e_592_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_1
  simp only [← c37428, ← c37429, k2, ← c37430] at e_592_1
  have hr := e_592_1
  simp only [e_592_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 592 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3239)) := by
  have c37431 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37432 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37433 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37434 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37435 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37436 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_592_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_2
  simp only [← c37431, ← c37432, ← c37433] at e_592_2
  have e_592_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_3
  simp only [← c37434, ← c37435, k2, ← c37436] at e_592_3
  have hr := e_592_3
  simp only [e_592_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 592 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3240)) := by
  have c37437 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37438 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37439 := (publicBatchWrapper86_copies1169 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37440 := (publicBatchWrapper86_copies1170 a h).1
  have c37441 := (publicBatchWrapper86_copies1170 a h).2.1
  have c37442 := (publicBatchWrapper86_copies1170 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_592_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_4
  simp only [← c37437, ← c37438, ← c37439] at e_592_4
  have e_592_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_5
  simp only [← c37440, ← c37441, k2, ← c37442] at e_592_5
  have hr := e_592_5
  simp only [e_592_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 592 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3241)) := by
  have c37443 := (publicBatchWrapper86_copies1170 a h).2.2.2.1
  have c37444 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.1
  have c37445 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.1
  have c37446 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.1
  have c37447 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.1
  have c37448 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_592_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_6
  simp only [← c37443, ← c37444, ← c37445] at e_592_6
  have e_592_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_7
  simp only [← c37446, ← c37447, k2, ← c37448] at e_592_7
  have hr := e_592_7
  simp only [e_592_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 592 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3242)) := by
  have c37449 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.1
  have c37450 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37451 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37452 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37453 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37454 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_592_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_8
  simp only [← c37449, ← c37450, ← c37451] at e_592_8
  have e_592_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_9
  simp only [← c37452, ← c37453, k2, ← c37454] at e_592_9
  have hr := e_592_9
  simp only [e_592_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 592 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3243)) := by
  have c37455 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37456 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37457 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37458 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37459 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37460 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_592_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_10
  simp only [← c37455, ← c37456, ← c37457] at e_592_10
  have e_592_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_11
  simp only [← c37458, ← c37459, k2, ← c37460] at e_592_11
  have hr := e_592_11
  simp only [e_592_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 592 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3244)) := by
  have c37461 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37462 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37463 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37464 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37465 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37466 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_592_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_12
  simp only [← c37461, ← c37462, ← c37463] at e_592_12
  have e_592_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_13
  simp only [← c37464, ← c37465, k2, ← c37466] at e_592_13
  have hr := e_592_13
  simp only [e_592_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 592 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3245)) := by
  have c37467 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37468 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37469 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37470 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37471 := (publicBatchWrapper86_copies1170 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37472 := (publicBatchWrapper86_copies1171 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_592_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_14
  simp only [← c37467, ← c37468, ← c37469] at e_592_14
  have e_592_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_15
  simp only [← c37470, ← c37471, k2, ← c37472] at e_592_15
  have hr := e_592_15
  simp only [e_592_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 592 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3246)) := by
  have c37473 := (publicBatchWrapper86_copies1171 a h).2.1
  have c37474 := (publicBatchWrapper86_copies1171 a h).2.2.1
  have c37475 := (publicBatchWrapper86_copies1171 a h).2.2.2.1
  have c37476 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.1
  have c37477 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.1
  have c37478 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_592_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_16
  simp only [← c37473, ← c37474, ← c37475] at e_592_16
  have e_592_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_17
  simp only [← c37476, ← c37477, k2, ← c37478] at e_592_17
  have hr := e_592_17
  simp only [e_592_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 592 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3247)) := by
  have c37479 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.1
  have c37480 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.1
  have c37481 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.1
  have c37482 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37483 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37484 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_592_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_18
  simp only [← c37479, ← c37480, ← c37481] at e_592_18
  have e_592_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_592_19
  simp only [← c37482, ← c37483, k2, ← c37484] at e_592_19
  have hr := e_592_19
  simp only [e_592_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 593 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3248)) := by
  have c37485 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37486 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37487 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37488 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37489 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37490 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_593_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_0
  simp only [← c37485, ← c37486, ← c37487] at e_593_0
  have e_593_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_1
  simp only [← c37488, ← c37489, k2, ← c37490] at e_593_1
  have hr := e_593_1
  simp only [e_593_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 593 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3249)) := by
  have c37491 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37492 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37493 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37494 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37495 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37496 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_593_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_2
  simp only [← c37491, ← c37492, ← c37493] at e_593_2
  have e_593_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_3
  simp only [← c37494, ← c37495, k2, ← c37496] at e_593_3
  have hr := e_593_3
  simp only [e_593_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 593 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3250)) := by
  have c37497 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37498 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37499 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37500 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37501 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37502 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_593_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_4
  simp only [← c37497, ← c37498, ← c37499] at e_593_4
  have e_593_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_5
  simp only [← c37500, ← c37501, k2, ← c37502] at e_593_5
  have hr := e_593_5
  simp only [e_593_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 593 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3251)) := by
  have c37503 := (publicBatchWrapper86_copies1171 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37504 := (publicBatchWrapper86_copies1172 a h).1
  have c37505 := (publicBatchWrapper86_copies1172 a h).2.1
  have c37506 := (publicBatchWrapper86_copies1172 a h).2.2.1
  have c37507 := (publicBatchWrapper86_copies1172 a h).2.2.2.1
  have c37508 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_593_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_6
  simp only [← c37503, ← c37504, ← c37505] at e_593_6
  have e_593_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_7
  simp only [← c37506, ← c37507, k2, ← c37508] at e_593_7
  have hr := e_593_7
  simp only [e_593_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 593 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3252)) := by
  have c37509 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.1
  have c37510 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.1
  have c37511 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.1
  have c37512 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.1
  have c37513 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.1
  have c37514 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_593_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_8
  simp only [← c37509, ← c37510, ← c37511] at e_593_8
  have e_593_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_9
  simp only [← c37512, ← c37513, k2, ← c37514] at e_593_9
  have hr := e_593_9
  simp only [e_593_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 593 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3253)) := by
  have c37515 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37516 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37517 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37518 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37519 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37520 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_593_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_10
  simp only [← c37515, ← c37516, ← c37517] at e_593_10
  have e_593_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_11
  simp only [← c37518, ← c37519, k2, ← c37520] at e_593_11
  have hr := e_593_11
  simp only [e_593_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 593 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3254)) := by
  have c37521 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37522 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37523 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37524 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37525 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37526 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_593_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_12
  simp only [← c37521, ← c37522, ← c37523] at e_593_12
  have e_593_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_13
  simp only [← c37524, ← c37525, k2, ← c37526] at e_593_13
  have hr := e_593_13
  simp only [e_593_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 593 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3255)) := by
  have c37527 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37528 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37529 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37530 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37531 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37532 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_593_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_14
  simp only [← c37527, ← c37528, ← c37529] at e_593_14
  have e_593_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_15
  simp only [← c37530, ← c37531, k2, ← c37532] at e_593_15
  have hr := e_593_15
  simp only [e_593_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 593 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3256)) := by
  have c37533 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37534 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37535 := (publicBatchWrapper86_copies1172 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37536 := (publicBatchWrapper86_copies1173 a h).1
  have c37537 := (publicBatchWrapper86_copies1173 a h).2.1
  have c37538 := (publicBatchWrapper86_copies1173 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_593_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_16
  simp only [← c37533, ← c37534, ← c37535] at e_593_16
  have e_593_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_17
  simp only [← c37536, ← c37537, k2, ← c37538] at e_593_17
  have hr := e_593_17
  simp only [e_593_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 593 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3257)) := by
  have c37539 := (publicBatchWrapper86_copies1173 a h).2.2.2.1
  have c37540 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.1
  have c37541 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.1
  have c37542 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.1
  have c37543 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.1
  have c37544 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_593_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_18
  simp only [← c37539, ← c37540, ← c37541] at e_593_18
  have e_593_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_593_19
  simp only [← c37542, ← c37543, k2, ← c37544] at e_593_19
  have hr := e_593_19
  simp only [e_593_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 594 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3258)) := by
  have c37545 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.1
  have c37546 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37547 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37548 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37549 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37550 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_594_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_0
  simp only [← c37545, ← c37546, ← c37547] at e_594_0
  have e_594_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_1
  simp only [← c37548, ← c37549, k2, ← c37550] at e_594_1
  have hr := e_594_1
  simp only [e_594_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 594 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3259)) := by
  have c37551 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37552 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37553 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37554 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37555 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37556 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_594_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_2
  simp only [← c37551, ← c37552, ← c37553] at e_594_2
  have e_594_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_3
  simp only [← c37554, ← c37555, k2, ← c37556] at e_594_3
  have hr := e_594_3
  simp only [e_594_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 594 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3260)) := by
  have c37557 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37558 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37559 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37560 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37561 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37562 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_594_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_4
  simp only [← c37557, ← c37558, ← c37559] at e_594_4
  have e_594_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_5
  simp only [← c37560, ← c37561, k2, ← c37562] at e_594_5
  have hr := e_594_5
  simp only [e_594_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 594 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3261)) := by
  have c37563 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37564 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37565 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37566 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37567 := (publicBatchWrapper86_copies1173 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37568 := (publicBatchWrapper86_copies1174 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_594_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_6
  simp only [← c37563, ← c37564, ← c37565] at e_594_6
  have e_594_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_7
  simp only [← c37566, ← c37567, k2, ← c37568] at e_594_7
  have hr := e_594_7
  simp only [e_594_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 594 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3262)) := by
  have c37569 := (publicBatchWrapper86_copies1174 a h).2.1
  have c37570 := (publicBatchWrapper86_copies1174 a h).2.2.1
  have c37571 := (publicBatchWrapper86_copies1174 a h).2.2.2.1
  have c37572 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.1
  have c37573 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.1
  have c37574 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_594_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_8
  simp only [← c37569, ← c37570, ← c37571] at e_594_8
  have e_594_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_9
  simp only [← c37572, ← c37573, k2, ← c37574] at e_594_9
  have hr := e_594_9
  simp only [e_594_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 594 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3263)) := by
  have c37575 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.1
  have c37576 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.1
  have c37577 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.1
  have c37578 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37579 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37580 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_594_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_10
  simp only [← c37575, ← c37576, ← c37577] at e_594_10
  have e_594_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_11
  simp only [← c37578, ← c37579, k2, ← c37580] at e_594_11
  have hr := e_594_11
  simp only [e_594_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 594 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3264)) := by
  have c37581 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37582 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37583 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37584 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37585 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37586 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_594_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_12
  simp only [← c37581, ← c37582, ← c37583] at e_594_12
  have e_594_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_13
  simp only [← c37584, ← c37585, k2, ← c37586] at e_594_13
  have hr := e_594_13
  simp only [e_594_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 594 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3265)) := by
  have c37587 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37588 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37589 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37590 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37591 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37592 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_594_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_14
  simp only [← c37587, ← c37588, ← c37589] at e_594_14
  have e_594_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_15
  simp only [← c37590, ← c37591, k2, ← c37592] at e_594_15
  have hr := e_594_15
  simp only [e_594_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 594 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3266)) := by
  have c37593 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37594 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37595 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37596 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37597 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37598 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_594_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_16
  simp only [← c37593, ← c37594, ← c37595] at e_594_16
  have e_594_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_17
  simp only [← c37596, ← c37597, k2, ← c37598] at e_594_17
  have hr := e_594_17
  simp only [e_594_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 594 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3267)) := by
  have c37599 := (publicBatchWrapper86_copies1174 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37600 := (publicBatchWrapper86_copies1175 a h).1
  have c37601 := (publicBatchWrapper86_copies1175 a h).2.1
  have c37602 := (publicBatchWrapper86_copies1175 a h).2.2.1
  have c37603 := (publicBatchWrapper86_copies1175 a h).2.2.2.1
  have c37604 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_594_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_18
  simp only [← c37599, ← c37600, ← c37601] at e_594_18
  have e_594_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_594_19
  simp only [← c37602, ← c37603, k2, ← c37604] at e_594_19
  have hr := e_594_19
  simp only [e_594_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 595 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3268)) := by
  have c37605 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.1
  have c37606 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.1
  have c37607 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.1
  have c37608 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.1
  have c37609 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.1
  have c37610 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_595_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_0
  simp only [← c37605, ← c37606, ← c37607] at e_595_0
  have e_595_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_1
  simp only [← c37608, ← c37609, k2, ← c37610] at e_595_1
  have hr := e_595_1
  simp only [e_595_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 595 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3269)) := by
  have c37611 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37612 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37613 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37614 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37615 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37616 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_595_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_2
  simp only [← c37611, ← c37612, ← c37613] at e_595_2
  have e_595_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_3
  simp only [← c37614, ← c37615, k2, ← c37616] at e_595_3
  have hr := e_595_3
  simp only [e_595_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 595 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3270)) := by
  have c37617 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37618 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37619 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37620 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37621 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37622 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_595_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_4
  simp only [← c37617, ← c37618, ← c37619] at e_595_4
  have e_595_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_5
  simp only [← c37620, ← c37621, k2, ← c37622] at e_595_5
  have hr := e_595_5
  simp only [e_595_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 595 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3271)) := by
  have c37623 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37624 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37625 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37626 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37627 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37628 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_595_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_6
  simp only [← c37623, ← c37624, ← c37625] at e_595_6
  have e_595_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_7
  simp only [← c37626, ← c37627, k2, ← c37628] at e_595_7
  have hr := e_595_7
  simp only [e_595_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 595 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3272)) := by
  have c37629 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37630 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37631 := (publicBatchWrapper86_copies1175 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37632 := (publicBatchWrapper86_copies1176 a h).1
  have c37633 := (publicBatchWrapper86_copies1176 a h).2.1
  have c37634 := (publicBatchWrapper86_copies1176 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_595_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_8
  simp only [← c37629, ← c37630, ← c37631] at e_595_8
  have e_595_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_9
  simp only [← c37632, ← c37633, k2, ← c37634] at e_595_9
  have hr := e_595_9
  simp only [e_595_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 595 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3273)) := by
  have c37635 := (publicBatchWrapper86_copies1176 a h).2.2.2.1
  have c37636 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.1
  have c37637 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.1
  have c37638 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.1
  have c37639 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.1
  have c37640 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_595_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_10
  simp only [← c37635, ← c37636, ← c37637] at e_595_10
  have e_595_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_11
  simp only [← c37638, ← c37639, k2, ← c37640] at e_595_11
  have hr := e_595_11
  simp only [e_595_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 595 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3274)) := by
  have c37641 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.1
  have c37642 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37643 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37644 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37645 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37646 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_595_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_12
  simp only [← c37641, ← c37642, ← c37643] at e_595_12
  have e_595_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_13
  simp only [← c37644, ← c37645, k2, ← c37646] at e_595_13
  have hr := e_595_13
  simp only [e_595_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 595 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3275)) := by
  have c37647 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37648 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37649 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37650 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37651 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37652 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_595_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_14
  simp only [← c37647, ← c37648, ← c37649] at e_595_14
  have e_595_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_15
  simp only [← c37650, ← c37651, k2, ← c37652] at e_595_15
  have hr := e_595_15
  simp only [e_595_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 595 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3276)) := by
  have c37653 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37654 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37655 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37656 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37657 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37658 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_595_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_16
  simp only [← c37653, ← c37654, ← c37655] at e_595_16
  have e_595_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_17
  simp only [← c37656, ← c37657, k2, ← c37658] at e_595_17
  have hr := e_595_17
  simp only [e_595_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 595 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3277)) := by
  have c37659 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37660 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37661 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37662 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37663 := (publicBatchWrapper86_copies1176 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37664 := (publicBatchWrapper86_copies1177 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_595_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_18
  simp only [← c37659, ← c37660, ← c37661] at e_595_18
  have e_595_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_595_19
  simp only [← c37662, ← c37663, k2, ← c37664] at e_595_19
  have hr := e_595_19
  simp only [e_595_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 596 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3278)) := by
  have c37665 := (publicBatchWrapper86_copies1177 a h).2.1
  have c37666 := (publicBatchWrapper86_copies1177 a h).2.2.1
  have c37667 := (publicBatchWrapper86_copies1177 a h).2.2.2.1
  have c37668 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.1
  have c37669 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.1
  have c37670 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_596_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_0
  simp only [← c37665, ← c37666, ← c37667] at e_596_0
  have e_596_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_1
  simp only [← c37668, ← c37669, k2, ← c37670] at e_596_1
  have hr := e_596_1
  simp only [e_596_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 596 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3279)) := by
  have c37671 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.1
  have c37672 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.1
  have c37673 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.1
  have c37674 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37675 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37676 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_596_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_2
  simp only [← c37671, ← c37672, ← c37673] at e_596_2
  have e_596_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_3
  simp only [← c37674, ← c37675, k2, ← c37676] at e_596_3
  have hr := e_596_3
  simp only [e_596_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 596 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3280)) := by
  have c37677 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37678 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37679 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37680 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37681 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37682 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_596_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_4
  simp only [← c37677, ← c37678, ← c37679] at e_596_4
  have e_596_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_5
  simp only [← c37680, ← c37681, k2, ← c37682] at e_596_5
  have hr := e_596_5
  simp only [e_596_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 596 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3281)) := by
  have c37683 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37684 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37685 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37686 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37687 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37688 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_596_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_6
  simp only [← c37683, ← c37684, ← c37685] at e_596_6
  have e_596_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_7
  simp only [← c37686, ← c37687, k2, ← c37688] at e_596_7
  have hr := e_596_7
  simp only [e_596_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 596 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3282)) := by
  have c37689 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37690 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37691 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37692 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37693 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37694 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_596_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_8
  simp only [← c37689, ← c37690, ← c37691] at e_596_8
  have e_596_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_9
  simp only [← c37692, ← c37693, k2, ← c37694] at e_596_9
  have hr := e_596_9
  simp only [e_596_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 596 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3283)) := by
  have c37695 := (publicBatchWrapper86_copies1177 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37696 := (publicBatchWrapper86_copies1178 a h).1
  have c37697 := (publicBatchWrapper86_copies1178 a h).2.1
  have c37698 := (publicBatchWrapper86_copies1178 a h).2.2.1
  have c37699 := (publicBatchWrapper86_copies1178 a h).2.2.2.1
  have c37700 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_596_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_10
  simp only [← c37695, ← c37696, ← c37697] at e_596_10
  have e_596_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_11
  simp only [← c37698, ← c37699, k2, ← c37700] at e_596_11
  have hr := e_596_11
  simp only [e_596_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 596 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3284)) := by
  have c37701 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.1
  have c37702 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.1
  have c37703 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.1
  have c37704 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.1
  have c37705 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.1
  have c37706 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_596_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_12
  simp only [← c37701, ← c37702, ← c37703] at e_596_12
  have e_596_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_13
  simp only [← c37704, ← c37705, k2, ← c37706] at e_596_13
  have hr := e_596_13
  simp only [e_596_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 596 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3285)) := by
  have c37707 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37708 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37709 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37710 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37711 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37712 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_596_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_14
  simp only [← c37707, ← c37708, ← c37709] at e_596_14
  have e_596_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_15
  simp only [← c37710, ← c37711, k2, ← c37712] at e_596_15
  have hr := e_596_15
  simp only [e_596_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 596 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3286)) := by
  have c37713 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37714 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37715 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37716 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37717 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37718 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_596_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_16
  simp only [← c37713, ← c37714, ← c37715] at e_596_16
  have e_596_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_17
  simp only [← c37716, ← c37717, k2, ← c37718] at e_596_17
  have hr := e_596_17
  simp only [e_596_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 596 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3287)) := by
  have c37719 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37720 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37721 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37722 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37723 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37724 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_596_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_18
  simp only [← c37719, ← c37720, ← c37721] at e_596_18
  have e_596_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_596_19
  simp only [← c37722, ← c37723, k2, ← c37724] at e_596_19
  have hr := e_596_19
  simp only [e_596_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 597 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3288)) := by
  have c37725 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37726 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37727 := (publicBatchWrapper86_copies1178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37728 := (publicBatchWrapper86_copies1179 a h).1
  have c37729 := (publicBatchWrapper86_copies1179 a h).2.1
  have c37730 := (publicBatchWrapper86_copies1179 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_597_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_0
  simp only [← c37725, ← c37726, ← c37727] at e_597_0
  have e_597_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_1
  simp only [← c37728, ← c37729, k2, ← c37730] at e_597_1
  have hr := e_597_1
  simp only [e_597_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 597 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3289)) := by
  have c37731 := (publicBatchWrapper86_copies1179 a h).2.2.2.1
  have c37732 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.1
  have c37733 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.1
  have c37734 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.1
  have c37735 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.1
  have c37736 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_597_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_2
  simp only [← c37731, ← c37732, ← c37733] at e_597_2
  have e_597_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_3
  simp only [← c37734, ← c37735, k2, ← c37736] at e_597_3
  have hr := e_597_3
  simp only [e_597_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 597 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3290)) := by
  have c37737 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.1
  have c37738 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37739 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37740 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37741 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37742 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_597_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_4
  simp only [← c37737, ← c37738, ← c37739] at e_597_4
  have e_597_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_5
  simp only [← c37740, ← c37741, k2, ← c37742] at e_597_5
  have hr := e_597_5
  simp only [e_597_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 597 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3291)) := by
  have c37743 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37744 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37745 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37746 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37747 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37748 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_597_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_6
  simp only [← c37743, ← c37744, ← c37745] at e_597_6
  have e_597_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_7
  simp only [← c37746, ← c37747, k2, ← c37748] at e_597_7
  have hr := e_597_7
  simp only [e_597_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 597 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3292)) := by
  have c37749 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37750 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37751 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37752 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37753 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37754 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_597_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_8
  simp only [← c37749, ← c37750, ← c37751] at e_597_8
  have e_597_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_9
  simp only [← c37752, ← c37753, k2, ← c37754] at e_597_9
  have hr := e_597_9
  simp only [e_597_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 597 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3293)) := by
  have c37755 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37756 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37757 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37758 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37759 := (publicBatchWrapper86_copies1179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37760 := (publicBatchWrapper86_copies1180 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_597_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_10
  simp only [← c37755, ← c37756, ← c37757] at e_597_10
  have e_597_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_11
  simp only [← c37758, ← c37759, k2, ← c37760] at e_597_11
  have hr := e_597_11
  simp only [e_597_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 597 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3294)) := by
  have c37761 := (publicBatchWrapper86_copies1180 a h).2.1
  have c37762 := (publicBatchWrapper86_copies1180 a h).2.2.1
  have c37763 := (publicBatchWrapper86_copies1180 a h).2.2.2.1
  have c37764 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.1
  have c37765 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.1
  have c37766 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_597_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_12
  simp only [← c37761, ← c37762, ← c37763] at e_597_12
  have e_597_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_13
  simp only [← c37764, ← c37765, k2, ← c37766] at e_597_13
  have hr := e_597_13
  simp only [e_597_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 597 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3295)) := by
  have c37767 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.1
  have c37768 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.1
  have c37769 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.1
  have c37770 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37771 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37772 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_597_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_14
  simp only [← c37767, ← c37768, ← c37769] at e_597_14
  have e_597_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_15
  simp only [← c37770, ← c37771, k2, ← c37772] at e_597_15
  have hr := e_597_15
  simp only [e_597_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 597 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3296)) := by
  have c37773 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37774 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37775 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37776 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37777 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37778 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_597_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_16
  simp only [← c37773, ← c37774, ← c37775] at e_597_16
  have e_597_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_17
  simp only [← c37776, ← c37777, k2, ← c37778] at e_597_17
  have hr := e_597_17
  simp only [e_597_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 597 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3297)) := by
  have c37779 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37780 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37781 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37782 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37783 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37784 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_597_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_18
  simp only [← c37779, ← c37780, ← c37781] at e_597_18
  have e_597_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_597_19
  simp only [← c37782, ← c37783, k2, ← c37784] at e_597_19
  have hr := e_597_19
  simp only [e_597_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 598 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3298)) := by
  have c37785 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37786 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37787 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37788 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37789 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37790 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_598_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_0
  simp only [← c37785, ← c37786, ← c37787] at e_598_0
  have e_598_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_1
  simp only [← c37788, ← c37789, k2, ← c37790] at e_598_1
  have hr := e_598_1
  simp only [e_598_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 598 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3299)) := by
  have c37791 := (publicBatchWrapper86_copies1180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37792 := (publicBatchWrapper86_copies1181 a h).1
  have c37793 := (publicBatchWrapper86_copies1181 a h).2.1
  have c37794 := (publicBatchWrapper86_copies1181 a h).2.2.1
  have c37795 := (publicBatchWrapper86_copies1181 a h).2.2.2.1
  have c37796 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_598_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_2
  simp only [← c37791, ← c37792, ← c37793] at e_598_2
  have e_598_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_3
  simp only [← c37794, ← c37795, k2, ← c37796] at e_598_3
  have hr := e_598_3
  simp only [e_598_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 598 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3300)) := by
  have c37797 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.1
  have c37798 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.1
  have c37799 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.1
  have c37800 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.1
  have c37801 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.1
  have c37802 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_598_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_4
  simp only [← c37797, ← c37798, ← c37799] at e_598_4
  have e_598_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_5
  simp only [← c37800, ← c37801, k2, ← c37802] at e_598_5
  have hr := e_598_5
  simp only [e_598_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 598 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3301)) := by
  have c37803 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37804 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37805 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37806 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37807 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37808 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_598_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_6
  simp only [← c37803, ← c37804, ← c37805] at e_598_6
  have e_598_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_7
  simp only [← c37806, ← c37807, k2, ← c37808] at e_598_7
  have hr := e_598_7
  simp only [e_598_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 598 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3302)) := by
  have c37809 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37810 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37811 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37812 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37813 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37814 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_598_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_8
  simp only [← c37809, ← c37810, ← c37811] at e_598_8
  have e_598_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_9
  simp only [← c37812, ← c37813, k2, ← c37814] at e_598_9
  have hr := e_598_9
  simp only [e_598_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 598 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3303)) := by
  have c37815 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37816 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37817 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37818 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37819 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37820 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_598_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_10
  simp only [← c37815, ← c37816, ← c37817] at e_598_10
  have e_598_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_11
  simp only [← c37818, ← c37819, k2, ← c37820] at e_598_11
  have hr := e_598_11
  simp only [e_598_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 598 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3304)) := by
  have c37821 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37822 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37823 := (publicBatchWrapper86_copies1181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37824 := (publicBatchWrapper86_copies1182 a h).1
  have c37825 := (publicBatchWrapper86_copies1182 a h).2.1
  have c37826 := (publicBatchWrapper86_copies1182 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_598_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_12
  simp only [← c37821, ← c37822, ← c37823] at e_598_12
  have e_598_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_13
  simp only [← c37824, ← c37825, k2, ← c37826] at e_598_13
  have hr := e_598_13
  simp only [e_598_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 598 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3305)) := by
  have c37827 := (publicBatchWrapper86_copies1182 a h).2.2.2.1
  have c37828 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.1
  have c37829 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.1
  have c37830 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.1
  have c37831 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.1
  have c37832 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_598_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_14
  simp only [← c37827, ← c37828, ← c37829] at e_598_14
  have e_598_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_15
  simp only [← c37830, ← c37831, k2, ← c37832] at e_598_15
  have hr := e_598_15
  simp only [e_598_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 598 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3306)) := by
  have c37833 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.1
  have c37834 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37835 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37836 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37837 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37838 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_598_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_16
  simp only [← c37833, ← c37834, ← c37835] at e_598_16
  have e_598_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_17
  simp only [← c37836, ← c37837, k2, ← c37838] at e_598_17
  have hr := e_598_17
  simp only [e_598_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 598 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3307)) := by
  have c37839 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37840 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37841 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37842 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37843 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37844 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_598_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_18
  simp only [← c37839, ← c37840, ← c37841] at e_598_18
  have e_598_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_598_19
  simp only [← c37842, ← c37843, k2, ← c37844] at e_598_19
  have hr := e_598_19
  simp only [e_598_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 599 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3308)) := by
  have c37845 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37846 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37847 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37848 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37849 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37850 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_599_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_0
  simp only [← c37845, ← c37846, ← c37847] at e_599_0
  have e_599_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_1
  simp only [← c37848, ← c37849, k2, ← c37850] at e_599_1
  have hr := e_599_1
  simp only [e_599_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 599 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3309)) := by
  have c37851 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37852 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37853 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37854 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37855 := (publicBatchWrapper86_copies1182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37856 := (publicBatchWrapper86_copies1183 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_599_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_2
  simp only [← c37851, ← c37852, ← c37853] at e_599_2
  have e_599_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_3
  simp only [← c37854, ← c37855, k2, ← c37856] at e_599_3
  have hr := e_599_3
  simp only [e_599_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 599 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3310)) := by
  have c37857 := (publicBatchWrapper86_copies1183 a h).2.1
  have c37858 := (publicBatchWrapper86_copies1183 a h).2.2.1
  have c37859 := (publicBatchWrapper86_copies1183 a h).2.2.2.1
  have c37860 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.1
  have c37861 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.1
  have c37862 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_599_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_4
  simp only [← c37857, ← c37858, ← c37859] at e_599_4
  have e_599_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_5
  simp only [← c37860, ← c37861, k2, ← c37862] at e_599_5
  have hr := e_599_5
  simp only [e_599_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 599 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3311)) := by
  have c37863 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.1
  have c37864 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.1
  have c37865 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.1
  have c37866 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37867 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37868 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_599_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_6
  simp only [← c37863, ← c37864, ← c37865] at e_599_6
  have e_599_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_7
  simp only [← c37866, ← c37867, k2, ← c37868] at e_599_7
  have hr := e_599_7
  simp only [e_599_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 599 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3312)) := by
  have c37869 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37870 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37871 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37872 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37873 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37874 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_599_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_8
  simp only [← c37869, ← c37870, ← c37871] at e_599_8
  have e_599_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_9
  simp only [← c37872, ← c37873, k2, ← c37874] at e_599_9
  have hr := e_599_9
  simp only [e_599_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 599 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3313)) := by
  have c37875 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37876 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37877 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37878 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37879 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37880 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_599_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_10
  simp only [← c37875, ← c37876, ← c37877] at e_599_10
  have e_599_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_11
  simp only [← c37878, ← c37879, k2, ← c37880] at e_599_11
  have hr := e_599_11
  simp only [e_599_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 599 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3314)) := by
  have c37881 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37882 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37883 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37884 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37885 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37886 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_599_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_12
  simp only [← c37881, ← c37882, ← c37883] at e_599_12
  have e_599_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_13
  simp only [← c37884, ← c37885, k2, ← c37886] at e_599_13
  have hr := e_599_13
  simp only [e_599_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 599 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3315)) := by
  have c37887 := (publicBatchWrapper86_copies1183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37888 := (publicBatchWrapper86_copies1184 a h).1
  have c37889 := (publicBatchWrapper86_copies1184 a h).2.1
  have c37890 := (publicBatchWrapper86_copies1184 a h).2.2.1
  have c37891 := (publicBatchWrapper86_copies1184 a h).2.2.2.1
  have c37892 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_599_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_14
  simp only [← c37887, ← c37888, ← c37889] at e_599_14
  have e_599_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_15
  simp only [← c37890, ← c37891, k2, ← c37892] at e_599_15
  have hr := e_599_15
  simp only [e_599_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 599 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3316)) := by
  have c37893 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.1
  have c37894 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.1
  have c37895 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.1
  have c37896 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.1
  have c37897 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.1
  have c37898 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_599_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_16
  simp only [← c37893, ← c37894, ← c37895] at e_599_16
  have e_599_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_17
  simp only [← c37896, ← c37897, k2, ← c37898] at e_599_17
  have hr := e_599_17
  simp only [e_599_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 599 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3317)) := by
  have c37899 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37900 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37901 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37902 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37903 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37904 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_599_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_18
  simp only [← c37899, ← c37900, ← c37901] at e_599_18
  have e_599_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_599_19
  simp only [← c37902, ← c37903, k2, ← c37904] at e_599_19
  have hr := e_599_19
  simp only [e_599_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 600 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3318)) := by
  have c37905 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37906 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37907 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37908 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37909 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37910 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_600_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_0
  simp only [← c37905, ← c37906, ← c37907] at e_600_0
  have e_600_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_1
  simp only [← c37908, ← c37909, k2, ← c37910] at e_600_1
  have hr := e_600_1
  simp only [e_600_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 600 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3319)) := by
  have c37911 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37912 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37913 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37914 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37915 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37916 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_600_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_2
  simp only [← c37911, ← c37912, ← c37913] at e_600_2
  have e_600_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_3
  simp only [← c37914, ← c37915, k2, ← c37916] at e_600_3
  have hr := e_600_3
  simp only [e_600_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 600 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3320)) := by
  have c37917 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37918 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37919 := (publicBatchWrapper86_copies1184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37920 := (publicBatchWrapper86_copies1185 a h).1
  have c37921 := (publicBatchWrapper86_copies1185 a h).2.1
  have c37922 := (publicBatchWrapper86_copies1185 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_600_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_4
  simp only [← c37917, ← c37918, ← c37919] at e_600_4
  have e_600_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_5
  simp only [← c37920, ← c37921, k2, ← c37922] at e_600_5
  have hr := e_600_5
  simp only [e_600_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 600 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3321)) := by
  have c37923 := (publicBatchWrapper86_copies1185 a h).2.2.2.1
  have c37924 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.1
  have c37925 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.1
  have c37926 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.1
  have c37927 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.1
  have c37928 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_600_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_6
  simp only [← c37923, ← c37924, ← c37925] at e_600_6
  have e_600_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_7
  simp only [← c37926, ← c37927, k2, ← c37928] at e_600_7
  have hr := e_600_7
  simp only [e_600_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 600 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3322)) := by
  have c37929 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.1
  have c37930 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37931 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37932 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37933 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37934 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_600_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_8
  simp only [← c37929, ← c37930, ← c37931] at e_600_8
  have e_600_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_9
  simp only [← c37932, ← c37933, k2, ← c37934] at e_600_9
  have hr := e_600_9
  simp only [e_600_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 600 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3323)) := by
  have c37935 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37936 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37937 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37938 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37939 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37940 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_600_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_10
  simp only [← c37935, ← c37936, ← c37937] at e_600_10
  have e_600_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_11
  simp only [← c37938, ← c37939, k2, ← c37940] at e_600_11
  have hr := e_600_11
  simp only [e_600_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 600 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3324)) := by
  have c37941 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37942 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37943 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37944 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37945 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37946 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_600_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_12
  simp only [← c37941, ← c37942, ← c37943] at e_600_12
  have e_600_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_13
  simp only [← c37944, ← c37945, k2, ← c37946] at e_600_13
  have hr := e_600_13
  simp only [e_600_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 600 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3325)) := by
  have c37947 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37948 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37949 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37950 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37951 := (publicBatchWrapper86_copies1185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37952 := (publicBatchWrapper86_copies1186 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_600_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_14
  simp only [← c37947, ← c37948, ← c37949] at e_600_14
  have e_600_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_15
  simp only [← c37950, ← c37951, k2, ← c37952] at e_600_15
  have hr := e_600_15
  simp only [e_600_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 600 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3326)) := by
  have c37953 := (publicBatchWrapper86_copies1186 a h).2.1
  have c37954 := (publicBatchWrapper86_copies1186 a h).2.2.1
  have c37955 := (publicBatchWrapper86_copies1186 a h).2.2.2.1
  have c37956 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.1
  have c37957 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.1
  have c37958 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_600_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_16
  simp only [← c37953, ← c37954, ← c37955] at e_600_16
  have e_600_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_17
  simp only [← c37956, ← c37957, k2, ← c37958] at e_600_17
  have hr := e_600_17
  simp only [e_600_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 600 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3327)) := by
  have c37959 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.1
  have c37960 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.1
  have c37961 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.1
  have c37962 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.1
  have c37963 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37964 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_600_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_18
  simp only [← c37959, ← c37960, ← c37961] at e_600_18
  have e_600_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_600_19
  simp only [← c37962, ← c37963, k2, ← c37964] at e_600_19
  have hr := e_600_19
  simp only [e_600_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 601 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3328)) := by
  have c37965 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37966 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37967 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37968 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37969 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37970 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_601_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_0
  simp only [← c37965, ← c37966, ← c37967] at e_601_0
  have e_601_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_1
  simp only [← c37968, ← c37969, k2, ← c37970] at e_601_1
  have hr := e_601_1
  simp only [e_601_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 601 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3329)) := by
  have c37971 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37972 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37973 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37974 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37975 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37976 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_601_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_2
  simp only [← c37971, ← c37972, ← c37973] at e_601_2
  have e_601_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_3
  simp only [← c37974, ← c37975, k2, ← c37976] at e_601_3
  have hr := e_601_3
  simp only [e_601_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 601 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3330)) := by
  have c37977 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37978 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37979 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37980 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37981 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37982 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_601_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_4
  simp only [← c37977, ← c37978, ← c37979] at e_601_4
  have e_601_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_5
  simp only [← c37980, ← c37981, k2, ← c37982] at e_601_5
  have hr := e_601_5
  simp only [e_601_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 601 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3331)) := by
  have c37983 := (publicBatchWrapper86_copies1186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c37984 := (publicBatchWrapper86_copies1187 a h).1
  have c37985 := (publicBatchWrapper86_copies1187 a h).2.1
  have c37986 := (publicBatchWrapper86_copies1187 a h).2.2.1
  have c37987 := (publicBatchWrapper86_copies1187 a h).2.2.2.1
  have c37988 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_601_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_6
  simp only [← c37983, ← c37984, ← c37985] at e_601_6
  have e_601_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_7
  simp only [← c37986, ← c37987, k2, ← c37988] at e_601_7
  have hr := e_601_7
  simp only [e_601_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 601 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3332)) := by
  have c37989 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.1
  have c37990 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.1
  have c37991 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.1
  have c37992 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.1
  have c37993 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.1
  have c37994 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_601_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_8
  simp only [← c37989, ← c37990, ← c37991] at e_601_8
  have e_601_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_9
  simp only [← c37992, ← c37993, k2, ← c37994] at e_601_9
  have hr := e_601_9
  simp only [e_601_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 601 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3333)) := by
  have c37995 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c37996 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37997 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37998 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c37999 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38000 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_601_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_10
  simp only [← c37995, ← c37996, ← c37997] at e_601_10
  have e_601_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_11
  simp only [← c37998, ← c37999, k2, ← c38000] at e_601_11
  have hr := e_601_11
  simp only [e_601_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 601 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3334)) := by
  have c38001 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38002 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38003 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38004 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38005 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38006 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_601_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_12
  simp only [← c38001, ← c38002, ← c38003] at e_601_12
  have e_601_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_13
  simp only [← c38004, ← c38005, k2, ← c38006] at e_601_13
  have hr := e_601_13
  simp only [e_601_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 601 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3335)) := by
  have c38007 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38008 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38009 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38010 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38011 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38012 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_601_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_14
  simp only [← c38007, ← c38008, ← c38009] at e_601_14
  have e_601_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_15
  simp only [← c38010, ← c38011, k2, ← c38012] at e_601_15
  have hr := e_601_15
  simp only [e_601_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 601 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3336)) := by
  have c38013 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38014 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38015 := (publicBatchWrapper86_copies1187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38016 := (publicBatchWrapper86_copies1188 a h).1
  have c38017 := (publicBatchWrapper86_copies1188 a h).2.1
  have c38018 := (publicBatchWrapper86_copies1188 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_601_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_16
  simp only [← c38013, ← c38014, ← c38015] at e_601_16
  have e_601_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_17
  simp only [← c38016, ← c38017, k2, ← c38018] at e_601_17
  have hr := e_601_17
  simp only [e_601_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 601 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3337)) := by
  have c38019 := (publicBatchWrapper86_copies1188 a h).2.2.2.1
  have c38020 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.1
  have c38021 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.1
  have c38022 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.1
  have c38023 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.1
  have c38024 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_601_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_18
  simp only [← c38019, ← c38020, ← c38021] at e_601_18
  have e_601_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_601_19
  simp only [← c38022, ← c38023, k2, ← c38024] at e_601_19
  have hr := e_601_19
  simp only [e_601_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 602 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3338)) := by
  have c38025 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.1
  have c38026 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38027 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38028 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38029 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38030 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_602_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_0
  simp only [← c38025, ← c38026, ← c38027] at e_602_0
  have e_602_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_1
  simp only [← c38028, ← c38029, k2, ← c38030] at e_602_1
  have hr := e_602_1
  simp only [e_602_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 602 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3339)) := by
  have c38031 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38032 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38033 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38034 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38035 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38036 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_602_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_2
  simp only [← c38031, ← c38032, ← c38033] at e_602_2
  have e_602_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_3
  simp only [← c38034, ← c38035, k2, ← c38036] at e_602_3
  have hr := e_602_3
  simp only [e_602_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 602 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3340)) := by
  have c38037 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38038 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38039 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38040 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38041 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38042 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_602_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_4
  simp only [← c38037, ← c38038, ← c38039] at e_602_4
  have e_602_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_5
  simp only [← c38040, ← c38041, k2, ← c38042] at e_602_5
  have hr := e_602_5
  simp only [e_602_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 602 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3341)) := by
  have c38043 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38044 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38045 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38046 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38047 := (publicBatchWrapper86_copies1188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38048 := (publicBatchWrapper86_copies1189 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_602_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_6
  simp only [← c38043, ← c38044, ← c38045] at e_602_6
  have e_602_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_7
  simp only [← c38046, ← c38047, k2, ← c38048] at e_602_7
  have hr := e_602_7
  simp only [e_602_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 602 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3342)) := by
  have c38049 := (publicBatchWrapper86_copies1189 a h).2.1
  have c38050 := (publicBatchWrapper86_copies1189 a h).2.2.1
  have c38051 := (publicBatchWrapper86_copies1189 a h).2.2.2.1
  have c38052 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.1
  have c38053 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.1
  have c38054 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_602_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_8
  simp only [← c38049, ← c38050, ← c38051] at e_602_8
  have e_602_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_9
  simp only [← c38052, ← c38053, k2, ← c38054] at e_602_9
  have hr := e_602_9
  simp only [e_602_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 602 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3343)) := by
  have c38055 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.1
  have c38056 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.1
  have c38057 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.1
  have c38058 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38059 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38060 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_602_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_10
  simp only [← c38055, ← c38056, ← c38057] at e_602_10
  have e_602_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_11
  simp only [← c38058, ← c38059, k2, ← c38060] at e_602_11
  have hr := e_602_11
  simp only [e_602_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 602 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3344)) := by
  have c38061 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38062 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38063 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38064 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38065 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38066 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_602_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_12
  simp only [← c38061, ← c38062, ← c38063] at e_602_12
  have e_602_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_13
  simp only [← c38064, ← c38065, k2, ← c38066] at e_602_13
  have hr := e_602_13
  simp only [e_602_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 602 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3345)) := by
  have c38067 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38068 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38069 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38070 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38071 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38072 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_602_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_14
  simp only [← c38067, ← c38068, ← c38069] at e_602_14
  have e_602_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_15
  simp only [← c38070, ← c38071, k2, ← c38072] at e_602_15
  have hr := e_602_15
  simp only [e_602_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 602 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3346)) := by
  have c38073 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38074 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38075 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38076 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38077 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38078 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_602_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_16
  simp only [← c38073, ← c38074, ← c38075] at e_602_16
  have e_602_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_17
  simp only [← c38076, ← c38077, k2, ← c38078] at e_602_17
  have hr := e_602_17
  simp only [e_602_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 602 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3347)) := by
  have c38079 := (publicBatchWrapper86_copies1189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38080 := (publicBatchWrapper86_copies1190 a h).1
  have c38081 := (publicBatchWrapper86_copies1190 a h).2.1
  have c38082 := (publicBatchWrapper86_copies1190 a h).2.2.1
  have c38083 := (publicBatchWrapper86_copies1190 a h).2.2.2.1
  have c38084 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_602_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_18
  simp only [← c38079, ← c38080, ← c38081] at e_602_18
  have e_602_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_602_19
  simp only [← c38082, ← c38083, k2, ← c38084] at e_602_19
  have hr := e_602_19
  simp only [e_602_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 603 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3348)) := by
  have c38085 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.1
  have c38086 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.1
  have c38087 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.1
  have c38088 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.1
  have c38089 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.1
  have c38090 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_603_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_0
  simp only [← c38085, ← c38086, ← c38087] at e_603_0
  have e_603_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_1
  simp only [← c38088, ← c38089, k2, ← c38090] at e_603_1
  have hr := e_603_1
  simp only [e_603_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 603 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3349)) := by
  have c38091 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38092 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38093 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38094 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38095 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38096 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_603_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_2
  simp only [← c38091, ← c38092, ← c38093] at e_603_2
  have e_603_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_3
  simp only [← c38094, ← c38095, k2, ← c38096] at e_603_3
  have hr := e_603_3
  simp only [e_603_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 603 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3350)) := by
  have c38097 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38098 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38099 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38100 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38101 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38102 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_603_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_4
  simp only [← c38097, ← c38098, ← c38099] at e_603_4
  have e_603_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_5
  simp only [← c38100, ← c38101, k2, ← c38102] at e_603_5
  have hr := e_603_5
  simp only [e_603_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 603 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3351)) := by
  have c38103 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38104 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38105 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38106 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38107 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38108 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_603_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_6
  simp only [← c38103, ← c38104, ← c38105] at e_603_6
  have e_603_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_7
  simp only [← c38106, ← c38107, k2, ← c38108] at e_603_7
  have hr := e_603_7
  simp only [e_603_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 603 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3352)) := by
  have c38109 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38110 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38111 := (publicBatchWrapper86_copies1190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38112 := (publicBatchWrapper86_copies1191 a h).1
  have c38113 := (publicBatchWrapper86_copies1191 a h).2.1
  have c38114 := (publicBatchWrapper86_copies1191 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_603_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_8
  simp only [← c38109, ← c38110, ← c38111] at e_603_8
  have e_603_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_9
  simp only [← c38112, ← c38113, k2, ← c38114] at e_603_9
  have hr := e_603_9
  simp only [e_603_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 603 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3353)) := by
  have c38115 := (publicBatchWrapper86_copies1191 a h).2.2.2.1
  have c38116 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.1
  have c38117 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.1
  have c38118 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.1
  have c38119 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.1
  have c38120 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_603_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_10
  simp only [← c38115, ← c38116, ← c38117] at e_603_10
  have e_603_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_11
  simp only [← c38118, ← c38119, k2, ← c38120] at e_603_11
  have hr := e_603_11
  simp only [e_603_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 603 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3354)) := by
  have c38121 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.1
  have c38122 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38123 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38124 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38125 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38126 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_603_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_12
  simp only [← c38121, ← c38122, ← c38123] at e_603_12
  have e_603_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_13
  simp only [← c38124, ← c38125, k2, ← c38126] at e_603_13
  have hr := e_603_13
  simp only [e_603_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 603 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3355)) := by
  have c38127 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38128 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38129 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38130 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38131 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38132 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_603_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_14
  simp only [← c38127, ← c38128, ← c38129] at e_603_14
  have e_603_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_15
  simp only [← c38130, ← c38131, k2, ← c38132] at e_603_15
  have hr := e_603_15
  simp only [e_603_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 603 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3356)) := by
  have c38133 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38134 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38135 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38136 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38137 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38138 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_603_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_16
  simp only [← c38133, ← c38134, ← c38135] at e_603_16
  have e_603_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_17
  simp only [← c38136, ← c38137, k2, ← c38138] at e_603_17
  have hr := e_603_17
  simp only [e_603_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 603 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3357)) := by
  have c38139 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38140 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38141 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38142 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38143 := (publicBatchWrapper86_copies1191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38144 := (publicBatchWrapper86_copies1192 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_603_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_18
  simp only [← c38139, ← c38140, ← c38141] at e_603_18
  have e_603_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_603_19
  simp only [← c38142, ← c38143, k2, ← c38144] at e_603_19
  have hr := e_603_19
  simp only [e_603_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 604 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3358)) := by
  have c38145 := (publicBatchWrapper86_copies1192 a h).2.1
  have c38146 := (publicBatchWrapper86_copies1192 a h).2.2.1
  have c38147 := (publicBatchWrapper86_copies1192 a h).2.2.2.1
  have c38148 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.1
  have c38149 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.1
  have c38150 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_604_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_0
  simp only [← c38145, ← c38146, ← c38147] at e_604_0
  have e_604_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_1
  simp only [← c38148, ← c38149, k2, ← c38150] at e_604_1
  have hr := e_604_1
  simp only [e_604_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 604 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3359)) := by
  have c38151 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.1
  have c38152 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.1
  have c38153 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.1
  have c38154 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38155 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38156 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_604_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_2
  simp only [← c38151, ← c38152, ← c38153] at e_604_2
  have e_604_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_3
  simp only [← c38154, ← c38155, k2, ← c38156] at e_604_3
  have hr := e_604_3
  simp only [e_604_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 604 23) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3360)) := by
  have c38157 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38158 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38159 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38160 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38161 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38162 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_604_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_4
  simp only [← c38157, ← c38158, ← c38159] at e_604_4
  have e_604_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_5
  simp only [← c38160, ← c38161, k2, ← c38162] at e_604_5
  have hr := e_604_5
  simp only [e_604_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 604 31) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3361)) := by
  have c38163 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38164 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38165 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38166 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38167 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38168 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_604_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_6
  simp only [← c38163, ← c38164, ← c38165] at e_604_6
  have e_604_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_7
  simp only [← c38166, ← c38167, k2, ← c38168] at e_604_7
  have hr := e_604_7
  simp only [e_604_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 604 39) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3362)) := by
  have c38169 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38170 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38171 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38172 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38173 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38174 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_604_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_8
  simp only [← c38169, ← c38170, ← c38171] at e_604_8
  have e_604_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_9
  simp only [← c38172, ← c38173, k2, ← c38174] at e_604_9
  have hr := e_604_9
  simp only [e_604_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 604 47) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3363)) := by
  have c38175 := (publicBatchWrapper86_copies1192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38176 := (publicBatchWrapper86_copies1193 a h).1
  have c38177 := (publicBatchWrapper86_copies1193 a h).2.1
  have c38178 := (publicBatchWrapper86_copies1193 a h).2.2.1
  have c38179 := (publicBatchWrapper86_copies1193 a h).2.2.2.1
  have c38180 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_604_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_10
  simp only [← c38175, ← c38176, ← c38177] at e_604_10
  have e_604_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_11
  simp only [← c38178, ← c38179, k2, ← c38180] at e_604_11
  have hr := e_604_11
  simp only [e_604_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 604 55) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3364)) := by
  have c38181 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.1
  have c38182 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.1
  have c38183 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.1
  have c38184 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.1
  have c38185 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.1
  have c38186 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_604_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_12
  simp only [← c38181, ← c38182, ← c38183] at e_604_12
  have e_604_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_13
  simp only [← c38184, ← c38185, k2, ← c38186] at e_604_13
  have hr := e_604_13
  simp only [e_604_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 604 63) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3365)) := by
  have c38187 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38188 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38189 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38190 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38191 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38192 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_604_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_14
  simp only [← c38187, ← c38188, ← c38189] at e_604_14
  have e_604_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_15
  simp only [← c38190, ← c38191, k2, ← c38192] at e_604_15
  have hr := e_604_15
  simp only [e_604_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 604 71) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3366)) := by
  have c38193 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38194 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38195 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38196 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38197 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38198 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_604_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_16
  simp only [← c38193, ← c38194, ← c38195] at e_604_16
  have e_604_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_17
  simp only [← c38196, ← c38197, k2, ← c38198] at e_604_17
  have hr := e_604_17
  simp only [e_604_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 604 79) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3367)) := by
  have c38199 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38200 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38201 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38202 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38203 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38204 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_604_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_18
  simp only [← c38199, ← c38200, ← c38201] at e_604_18
  have e_604_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_604_19
  simp only [← c38202, ← c38203, k2, ← c38204] at e_604_19
  have hr := e_604_19
  simp only [e_604_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 605 7) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3368)) := by
  have c38205 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38206 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38207 := (publicBatchWrapper86_copies1193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38208 := (publicBatchWrapper86_copies1194 a h).1
  have c38209 := (publicBatchWrapper86_copies1194 a h).2.1
  have c38210 := (publicBatchWrapper86_copies1194 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_605_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_0
  simp only [← c38205, ← c38206, ← c38207] at e_605_0
  have e_605_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_1
  simp only [← c38208, ← c38209, k2, ← c38210] at e_605_1
  have hr := e_605_1
  simp only [e_605_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 605 15) = bselect (a (.wire 18 11)) (a (.virt 200)) (a (.virt 3369)) := by
  have c38211 := (publicBatchWrapper86_copies1194 a h).2.2.2.1
  have c38212 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.1
  have c38213 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.1
  have c38214 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.1
  have c38215 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.1
  have c38216 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_605_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_2
  simp only [← c38211, ← c38212, ← c38213] at e_605_2
  have e_605_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_3
  simp only [← c38214, ← c38215, k2, ← c38216] at e_605_3
  have hr := e_605_3
  simp only [e_605_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 605 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3410)) := by
  have c38217 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.1
  have c38218 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38219 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38220 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38221 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38222 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_605_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_4
  simp only [← c38217, ← c38218, ← c38219] at e_605_4
  have e_605_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_5
  simp only [← c38220, ← c38221, k2, ← c38222] at e_605_5
  have hr := e_605_5
  simp only [e_605_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 605 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3411)) := by
  have c38223 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38224 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38225 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38226 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38227 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38228 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_605_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_6
  simp only [← c38223, ← c38224, ← c38225] at e_605_6
  have e_605_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_7
  simp only [← c38226, ← c38227, k2, ← c38228] at e_605_7
  have hr := e_605_7
  simp only [e_605_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 605 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3412)) := by
  have c38229 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38230 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38231 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38232 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38233 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38234 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_605_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_8
  simp only [← c38229, ← c38230, ← c38231] at e_605_8
  have e_605_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_9
  simp only [← c38232, ← c38233, k2, ← c38234] at e_605_9
  have hr := e_605_9
  simp only [e_605_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 605 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3413)) := by
  have c38235 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38236 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38237 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38238 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38239 := (publicBatchWrapper86_copies1194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38240 := (publicBatchWrapper86_copies1195 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_605_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_10
  simp only [← c38235, ← c38236, ← c38237] at e_605_10
  have e_605_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_11
  simp only [← c38238, ← c38239, k2, ← c38240] at e_605_11
  have hr := e_605_11
  simp only [e_605_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 605 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3414)) := by
  have c38241 := (publicBatchWrapper86_copies1195 a h).2.1
  have c38242 := (publicBatchWrapper86_copies1195 a h).2.2.1
  have c38243 := (publicBatchWrapper86_copies1195 a h).2.2.2.1
  have c38244 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.1
  have c38245 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.1
  have c38246 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_605_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_12
  simp only [← c38241, ← c38242, ← c38243] at e_605_12
  have e_605_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_13
  simp only [← c38244, ← c38245, k2, ← c38246] at e_605_13
  have hr := e_605_13
  simp only [e_605_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 605 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3415)) := by
  have c38247 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.1
  have c38248 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.1
  have c38249 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.1
  have c38250 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38251 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38252 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_605_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_14
  simp only [← c38247, ← c38248, ← c38249] at e_605_14
  have e_605_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_15
  simp only [← c38250, ← c38251, k2, ← c38252] at e_605_15
  have hr := e_605_15
  simp only [e_605_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 605 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3416)) := by
  have c38253 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38254 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38255 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38256 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38257 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38258 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_605_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_16
  simp only [← c38253, ← c38254, ← c38255] at e_605_16
  have e_605_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_17
  simp only [← c38256, ← c38257, k2, ← c38258] at e_605_17
  have hr := e_605_17
  simp only [e_605_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 605 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3417)) := by
  have c38259 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38260 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38261 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38262 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38263 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38264 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_605_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_18
  simp only [← c38259, ← c38260, ← c38261] at e_605_18
  have e_605_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_605_19
  simp only [← c38262, ← c38263, k2, ← c38264] at e_605_19
  have hr := e_605_19
  simp only [e_605_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 606 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3418)) := by
  have c38265 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38266 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38267 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38268 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38269 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38270 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_606_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_0
  simp only [← c38265, ← c38266, ← c38267] at e_606_0
  have e_606_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_1
  simp only [← c38268, ← c38269, k2, ← c38270] at e_606_1
  have hr := e_606_1
  simp only [e_606_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 606 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3419)) := by
  have c38271 := (publicBatchWrapper86_copies1195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38272 := (publicBatchWrapper86_copies1196 a h).1
  have c38273 := (publicBatchWrapper86_copies1196 a h).2.1
  have c38274 := (publicBatchWrapper86_copies1196 a h).2.2.1
  have c38275 := (publicBatchWrapper86_copies1196 a h).2.2.2.1
  have c38276 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_606_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_2
  simp only [← c38271, ← c38272, ← c38273] at e_606_2
  have e_606_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_3
  simp only [← c38274, ← c38275, k2, ← c38276] at e_606_3
  have hr := e_606_3
  simp only [e_606_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 606 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3420)) := by
  have c38277 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.1
  have c38278 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.1
  have c38279 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.1
  have c38280 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.1
  have c38281 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.1
  have c38282 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_606_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_4
  simp only [← c38277, ← c38278, ← c38279] at e_606_4
  have e_606_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_5
  simp only [← c38280, ← c38281, k2, ← c38282] at e_606_5
  have hr := e_606_5
  simp only [e_606_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 606 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3421)) := by
  have c38283 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38284 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38285 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38286 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38287 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38288 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_606_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_6
  simp only [← c38283, ← c38284, ← c38285] at e_606_6
  have e_606_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_7
  simp only [← c38286, ← c38287, k2, ← c38288] at e_606_7
  have hr := e_606_7
  simp only [e_606_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 606 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3422)) := by
  have c38289 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38290 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38291 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38292 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38293 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38294 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_606_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_8
  simp only [← c38289, ← c38290, ← c38291] at e_606_8
  have e_606_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_9
  simp only [← c38292, ← c38293, k2, ← c38294] at e_606_9
  have hr := e_606_9
  simp only [e_606_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 606 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3423)) := by
  have c38295 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38296 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38297 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38298 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38299 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38300 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_606_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_10
  simp only [← c38295, ← c38296, ← c38297] at e_606_10
  have e_606_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_11
  simp only [← c38298, ← c38299, k2, ← c38300] at e_606_11
  have hr := e_606_11
  simp only [e_606_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 606 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3424)) := by
  have c38301 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38302 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38303 := (publicBatchWrapper86_copies1196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38304 := (publicBatchWrapper86_copies1197 a h).1
  have c38305 := (publicBatchWrapper86_copies1197 a h).2.1
  have c38306 := (publicBatchWrapper86_copies1197 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_606_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_12
  simp only [← c38301, ← c38302, ← c38303] at e_606_12
  have e_606_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_13
  simp only [← c38304, ← c38305, k2, ← c38306] at e_606_13
  have hr := e_606_13
  simp only [e_606_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 606 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3425)) := by
  have c38307 := (publicBatchWrapper86_copies1197 a h).2.2.2.1
  have c38308 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.1
  have c38309 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.1
  have c38310 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.1
  have c38311 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.1
  have c38312 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_606_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_14
  simp only [← c38307, ← c38308, ← c38309] at e_606_14
  have e_606_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_15
  simp only [← c38310, ← c38311, k2, ← c38312] at e_606_15
  have hr := e_606_15
  simp only [e_606_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 606 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3426)) := by
  have c38313 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.1
  have c38314 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38315 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38316 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38317 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38318 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_606_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_16
  simp only [← c38313, ← c38314, ← c38315] at e_606_16
  have e_606_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_17
  simp only [← c38316, ← c38317, k2, ← c38318] at e_606_17
  have hr := e_606_17
  simp only [e_606_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 606 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3427)) := by
  have c38319 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38320 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38321 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38322 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38323 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38324 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_606_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_18
  simp only [← c38319, ← c38320, ← c38321] at e_606_18
  have e_606_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_606_19
  simp only [← c38322, ← c38323, k2, ← c38324] at e_606_19
  have hr := e_606_19
  simp only [e_606_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 607 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3428)) := by
  have c38325 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38326 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38327 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38328 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38329 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38330 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_607_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_0
  simp only [← c38325, ← c38326, ← c38327] at e_607_0
  have e_607_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_1
  simp only [← c38328, ← c38329, k2, ← c38330] at e_607_1
  have hr := e_607_1
  simp only [e_607_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 607 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3429)) := by
  have c38331 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38332 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38333 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38334 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38335 := (publicBatchWrapper86_copies1197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38336 := (publicBatchWrapper86_copies1198 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_607_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_2
  simp only [← c38331, ← c38332, ← c38333] at e_607_2
  have e_607_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_3
  simp only [← c38334, ← c38335, k2, ← c38336] at e_607_3
  have hr := e_607_3
  simp only [e_607_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 607 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3430)) := by
  have c38337 := (publicBatchWrapper86_copies1198 a h).2.1
  have c38338 := (publicBatchWrapper86_copies1198 a h).2.2.1
  have c38339 := (publicBatchWrapper86_copies1198 a h).2.2.2.1
  have c38340 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.1
  have c38341 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.1
  have c38342 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_607_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_4
  simp only [← c38337, ← c38338, ← c38339] at e_607_4
  have e_607_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_5
  simp only [← c38340, ← c38341, k2, ← c38342] at e_607_5
  have hr := e_607_5
  simp only [e_607_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 607 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3431)) := by
  have c38343 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.1
  have c38344 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.1
  have c38345 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.1
  have c38346 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38347 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38348 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_607_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_6
  simp only [← c38343, ← c38344, ← c38345] at e_607_6
  have e_607_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_7
  simp only [← c38346, ← c38347, k2, ← c38348] at e_607_7
  have hr := e_607_7
  simp only [e_607_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 607 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3432)) := by
  have c38349 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38350 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38351 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38352 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38353 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38354 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_607_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_8
  simp only [← c38349, ← c38350, ← c38351] at e_607_8
  have e_607_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_9
  simp only [← c38352, ← c38353, k2, ← c38354] at e_607_9
  have hr := e_607_9
  simp only [e_607_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 607 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3433)) := by
  have c38355 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38356 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38357 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38358 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38359 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38360 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_607_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_10
  simp only [← c38355, ← c38356, ← c38357] at e_607_10
  have e_607_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_11
  simp only [← c38358, ← c38359, k2, ← c38360] at e_607_11
  have hr := e_607_11
  simp only [e_607_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 607 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3434)) := by
  have c38361 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38362 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38363 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38364 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38365 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38366 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_607_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_12
  simp only [← c38361, ← c38362, ← c38363] at e_607_12
  have e_607_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_13
  simp only [← c38364, ← c38365, k2, ← c38366] at e_607_13
  have hr := e_607_13
  simp only [e_607_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 607 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3435)) := by
  have c38367 := (publicBatchWrapper86_copies1198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38368 := (publicBatchWrapper86_copies1199 a h).1
  have c38369 := (publicBatchWrapper86_copies1199 a h).2.1
  have c38370 := (publicBatchWrapper86_copies1199 a h).2.2.1
  have c38371 := (publicBatchWrapper86_copies1199 a h).2.2.2.1
  have c38372 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_607_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_14
  simp only [← c38367, ← c38368, ← c38369] at e_607_14
  have e_607_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_15
  simp only [← c38370, ← c38371, k2, ← c38372] at e_607_15
  have hr := e_607_15
  simp only [e_607_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 607 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3436)) := by
  have c38373 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.1
  have c38374 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.1
  have c38375 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.1
  have c38376 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.1
  have c38377 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.1
  have c38378 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_607_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_16
  simp only [← c38373, ← c38374, ← c38375] at e_607_16
  have e_607_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_17
  simp only [← c38376, ← c38377, k2, ← c38378] at e_607_17
  have hr := e_607_17
  simp only [e_607_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 607 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3437)) := by
  have c38379 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38380 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38381 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38382 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38383 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38384 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_607_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_18
  simp only [← c38379, ← c38380, ← c38381] at e_607_18
  have e_607_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_607_19
  simp only [← c38382, ← c38383, k2, ← c38384] at e_607_19
  have hr := e_607_19
  simp only [e_607_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 608 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3438)) := by
  have c38385 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38386 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38387 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38388 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38389 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38390 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_608_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_0
  simp only [← c38385, ← c38386, ← c38387] at e_608_0
  have e_608_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_1
  simp only [← c38388, ← c38389, k2, ← c38390] at e_608_1
  have hr := e_608_1
  simp only [e_608_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 608 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3439)) := by
  have c38391 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38392 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38393 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38394 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38395 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38396 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_608_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_2
  simp only [← c38391, ← c38392, ← c38393] at e_608_2
  have e_608_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_3
  simp only [← c38394, ← c38395, k2, ← c38396] at e_608_3
  have hr := e_608_3
  simp only [e_608_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 608 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3440)) := by
  have c38397 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38398 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38399 := (publicBatchWrapper86_copies1199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38400 := (publicBatchWrapper86_copies1200 a h).1
  have c38401 := (publicBatchWrapper86_copies1200 a h).2.1
  have c38402 := (publicBatchWrapper86_copies1200 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_608_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_4
  simp only [← c38397, ← c38398, ← c38399] at e_608_4
  have e_608_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_5
  simp only [← c38400, ← c38401, k2, ← c38402] at e_608_5
  have hr := e_608_5
  simp only [e_608_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 608 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3441)) := by
  have c38403 := (publicBatchWrapper86_copies1200 a h).2.2.2.1
  have c38404 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.1
  have c38405 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.1
  have c38406 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.1
  have c38407 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.1
  have c38408 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_608_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_6
  simp only [← c38403, ← c38404, ← c38405] at e_608_6
  have e_608_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_7
  simp only [← c38406, ← c38407, k2, ← c38408] at e_608_7
  have hr := e_608_7
  simp only [e_608_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 608 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3442)) := by
  have c38409 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.1
  have c38410 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38411 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38412 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38413 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38414 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_608_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_8
  simp only [← c38409, ← c38410, ← c38411] at e_608_8
  have e_608_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_9
  simp only [← c38412, ← c38413, k2, ← c38414] at e_608_9
  have hr := e_608_9
  simp only [e_608_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 608 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3443)) := by
  have c38415 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38416 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38417 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38418 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38419 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38420 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_608_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_10
  simp only [← c38415, ← c38416, ← c38417] at e_608_10
  have e_608_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_11
  simp only [← c38418, ← c38419, k2, ← c38420] at e_608_11
  have hr := e_608_11
  simp only [e_608_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 608 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3444)) := by
  have c38421 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38422 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38423 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38424 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38425 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38426 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_608_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_12
  simp only [← c38421, ← c38422, ← c38423] at e_608_12
  have e_608_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_13
  simp only [← c38424, ← c38425, k2, ← c38426] at e_608_13
  have hr := e_608_13
  simp only [e_608_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 608 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3445)) := by
  have c38427 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38428 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38429 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38430 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38431 := (publicBatchWrapper86_copies1200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38432 := (publicBatchWrapper86_copies1201 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_608_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_14
  simp only [← c38427, ← c38428, ← c38429] at e_608_14
  have e_608_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_15
  simp only [← c38430, ← c38431, k2, ← c38432] at e_608_15
  have hr := e_608_15
  simp only [e_608_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 608 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3446)) := by
  have c38433 := (publicBatchWrapper86_copies1201 a h).2.1
  have c38434 := (publicBatchWrapper86_copies1201 a h).2.2.1
  have c38435 := (publicBatchWrapper86_copies1201 a h).2.2.2.1
  have c38436 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.1
  have c38437 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.1
  have c38438 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_608_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_16
  simp only [← c38433, ← c38434, ← c38435] at e_608_16
  have e_608_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_17
  simp only [← c38436, ← c38437, k2, ← c38438] at e_608_17
  have hr := e_608_17
  simp only [e_608_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 608 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3447)) := by
  have c38439 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.1
  have c38440 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.1
  have c38441 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.1
  have c38442 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38443 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38444 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_608_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_18
  simp only [← c38439, ← c38440, ← c38441] at e_608_18
  have e_608_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_608_19
  simp only [← c38442, ← c38443, k2, ← c38444] at e_608_19
  have hr := e_608_19
  simp only [e_608_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 609 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3448)) := by
  have c38445 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38446 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38447 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38448 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38449 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38450 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_609_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_0
  simp only [← c38445, ← c38446, ← c38447] at e_609_0
  have e_609_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_1
  simp only [← c38448, ← c38449, k2, ← c38450] at e_609_1
  have hr := e_609_1
  simp only [e_609_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 609 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3449)) := by
  have c38451 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38452 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38453 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38454 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38455 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38456 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_609_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_2
  simp only [← c38451, ← c38452, ← c38453] at e_609_2
  have e_609_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_3
  simp only [← c38454, ← c38455, k2, ← c38456] at e_609_3
  have hr := e_609_3
  simp only [e_609_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 609 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3450)) := by
  have c38457 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38458 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38459 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38460 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38461 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38462 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_609_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_4
  simp only [← c38457, ← c38458, ← c38459] at e_609_4
  have e_609_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_5
  simp only [← c38460, ← c38461, k2, ← c38462] at e_609_5
  have hr := e_609_5
  simp only [e_609_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 609 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3451)) := by
  have c38463 := (publicBatchWrapper86_copies1201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38464 := (publicBatchWrapper86_copies1202 a h).1
  have c38465 := (publicBatchWrapper86_copies1202 a h).2.1
  have c38466 := (publicBatchWrapper86_copies1202 a h).2.2.1
  have c38467 := (publicBatchWrapper86_copies1202 a h).2.2.2.1
  have c38468 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_609_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_6
  simp only [← c38463, ← c38464, ← c38465] at e_609_6
  have e_609_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_7
  simp only [← c38466, ← c38467, k2, ← c38468] at e_609_7
  have hr := e_609_7
  simp only [e_609_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 609 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3452)) := by
  have c38469 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.1
  have c38470 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.1
  have c38471 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.1
  have c38472 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.1
  have c38473 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.1
  have c38474 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_609_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_8
  simp only [← c38469, ← c38470, ← c38471] at e_609_8
  have e_609_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_9
  simp only [← c38472, ← c38473, k2, ← c38474] at e_609_9
  have hr := e_609_9
  simp only [e_609_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 609 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3453)) := by
  have c38475 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38476 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38477 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38478 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38479 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38480 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_609_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_10
  simp only [← c38475, ← c38476, ← c38477] at e_609_10
  have e_609_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_11
  simp only [← c38478, ← c38479, k2, ← c38480] at e_609_11
  have hr := e_609_11
  simp only [e_609_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 609 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3454)) := by
  have c38481 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38482 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38483 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38484 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38485 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38486 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_609_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_12
  simp only [← c38481, ← c38482, ← c38483] at e_609_12
  have e_609_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_13
  simp only [← c38484, ← c38485, k2, ← c38486] at e_609_13
  have hr := e_609_13
  simp only [e_609_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 609 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3455)) := by
  have c38487 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38488 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38489 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38490 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38491 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38492 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_609_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_14
  simp only [← c38487, ← c38488, ← c38489] at e_609_14
  have e_609_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_15
  simp only [← c38490, ← c38491, k2, ← c38492] at e_609_15
  have hr := e_609_15
  simp only [e_609_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 609 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3456)) := by
  have c38493 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38494 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38495 := (publicBatchWrapper86_copies1202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38496 := (publicBatchWrapper86_copies1203 a h).1
  have c38497 := (publicBatchWrapper86_copies1203 a h).2.1
  have c38498 := (publicBatchWrapper86_copies1203 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_609_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_16
  simp only [← c38493, ← c38494, ← c38495] at e_609_16
  have e_609_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_17
  simp only [← c38496, ← c38497, k2, ← c38498] at e_609_17
  have hr := e_609_17
  simp only [e_609_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 609 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3457)) := by
  have c38499 := (publicBatchWrapper86_copies1203 a h).2.2.2.1
  have c38500 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.1
  have c38501 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.1
  have c38502 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.1
  have c38503 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.1
  have c38504 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_609_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_18
  simp only [← c38499, ← c38500, ← c38501] at e_609_18
  have e_609_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_609_19
  simp only [← c38502, ← c38503, k2, ← c38504] at e_609_19
  have hr := e_609_19
  simp only [e_609_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 610 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3458)) := by
  have c38505 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.1
  have c38506 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38507 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38508 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38509 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38510 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_610_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_0
  simp only [← c38505, ← c38506, ← c38507] at e_610_0
  have e_610_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_1
  simp only [← c38508, ← c38509, k2, ← c38510] at e_610_1
  have hr := e_610_1
  simp only [e_610_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 610 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3459)) := by
  have c38511 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38512 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38513 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38514 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38515 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38516 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_610_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_2
  simp only [← c38511, ← c38512, ← c38513] at e_610_2
  have e_610_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_3
  simp only [← c38514, ← c38515, k2, ← c38516] at e_610_3
  have hr := e_610_3
  simp only [e_610_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 610 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3460)) := by
  have c38517 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38518 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38519 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38520 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38521 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38522 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_610_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_4
  simp only [← c38517, ← c38518, ← c38519] at e_610_4
  have e_610_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_5
  simp only [← c38520, ← c38521, k2, ← c38522] at e_610_5
  have hr := e_610_5
  simp only [e_610_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 610 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3461)) := by
  have c38523 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38524 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38525 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38526 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38527 := (publicBatchWrapper86_copies1203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38528 := (publicBatchWrapper86_copies1204 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_610_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_6
  simp only [← c38523, ← c38524, ← c38525] at e_610_6
  have e_610_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_7
  simp only [← c38526, ← c38527, k2, ← c38528] at e_610_7
  have hr := e_610_7
  simp only [e_610_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 610 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3462)) := by
  have c38529 := (publicBatchWrapper86_copies1204 a h).2.1
  have c38530 := (publicBatchWrapper86_copies1204 a h).2.2.1
  have c38531 := (publicBatchWrapper86_copies1204 a h).2.2.2.1
  have c38532 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.1
  have c38533 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.1
  have c38534 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_610_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_8
  simp only [← c38529, ← c38530, ← c38531] at e_610_8
  have e_610_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_9
  simp only [← c38532, ← c38533, k2, ← c38534] at e_610_9
  have hr := e_610_9
  simp only [e_610_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 610 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3463)) := by
  have c38535 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.1
  have c38536 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.1
  have c38537 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.1
  have c38538 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38539 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38540 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_610_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_10
  simp only [← c38535, ← c38536, ← c38537] at e_610_10
  have e_610_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_11
  simp only [← c38538, ← c38539, k2, ← c38540] at e_610_11
  have hr := e_610_11
  simp only [e_610_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 610 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3464)) := by
  have c38541 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38542 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38543 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38544 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38545 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38546 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_610_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_12
  simp only [← c38541, ← c38542, ← c38543] at e_610_12
  have e_610_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_13
  simp only [← c38544, ← c38545, k2, ← c38546] at e_610_13
  have hr := e_610_13
  simp only [e_610_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 610 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3465)) := by
  have c38547 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38548 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38549 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38550 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38551 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38552 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_610_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_14
  simp only [← c38547, ← c38548, ← c38549] at e_610_14
  have e_610_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_15
  simp only [← c38550, ← c38551, k2, ← c38552] at e_610_15
  have hr := e_610_15
  simp only [e_610_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 610 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3466)) := by
  have c38553 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38554 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38555 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38556 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38557 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38558 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_610_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_16
  simp only [← c38553, ← c38554, ← c38555] at e_610_16
  have e_610_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_17
  simp only [← c38556, ← c38557, k2, ← c38558] at e_610_17
  have hr := e_610_17
  simp only [e_610_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 610 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3467)) := by
  have c38559 := (publicBatchWrapper86_copies1204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38560 := (publicBatchWrapper86_copies1205 a h).1
  have c38561 := (publicBatchWrapper86_copies1205 a h).2.1
  have c38562 := (publicBatchWrapper86_copies1205 a h).2.2.1
  have c38563 := (publicBatchWrapper86_copies1205 a h).2.2.2.1
  have c38564 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_610_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_18
  simp only [← c38559, ← c38560, ← c38561] at e_610_18
  have e_610_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_610_19
  simp only [← c38562, ← c38563, k2, ← c38564] at e_610_19
  have hr := e_610_19
  simp only [e_610_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 611 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3468)) := by
  have c38565 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.1
  have c38566 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.1
  have c38567 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.1
  have c38568 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.1
  have c38569 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.1
  have c38570 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_611_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_0
  simp only [← c38565, ← c38566, ← c38567] at e_611_0
  have e_611_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_1
  simp only [← c38568, ← c38569, k2, ← c38570] at e_611_1
  have hr := e_611_1
  simp only [e_611_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 611 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3469)) := by
  have c38571 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38572 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38573 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38574 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38575 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38576 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_611_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_2
  simp only [← c38571, ← c38572, ← c38573] at e_611_2
  have e_611_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_3
  simp only [← c38574, ← c38575, k2, ← c38576] at e_611_3
  have hr := e_611_3
  simp only [e_611_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 611 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3470)) := by
  have c38577 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38578 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38579 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38580 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38581 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38582 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_611_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_4
  simp only [← c38577, ← c38578, ← c38579] at e_611_4
  have e_611_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_5
  simp only [← c38580, ← c38581, k2, ← c38582] at e_611_5
  have hr := e_611_5
  simp only [e_611_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 611 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3471)) := by
  have c38583 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38584 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38585 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38586 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38587 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38588 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_611_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_6
  simp only [← c38583, ← c38584, ← c38585] at e_611_6
  have e_611_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_7
  simp only [← c38586, ← c38587, k2, ← c38588] at e_611_7
  have hr := e_611_7
  simp only [e_611_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 611 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3472)) := by
  have c38589 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38590 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38591 := (publicBatchWrapper86_copies1205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38592 := (publicBatchWrapper86_copies1206 a h).1
  have c38593 := (publicBatchWrapper86_copies1206 a h).2.1
  have c38594 := (publicBatchWrapper86_copies1206 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_611_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_8
  simp only [← c38589, ← c38590, ← c38591] at e_611_8
  have e_611_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_9
  simp only [← c38592, ← c38593, k2, ← c38594] at e_611_9
  have hr := e_611_9
  simp only [e_611_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 611 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3473)) := by
  have c38595 := (publicBatchWrapper86_copies1206 a h).2.2.2.1
  have c38596 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.1
  have c38597 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.1
  have c38598 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.1
  have c38599 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.1
  have c38600 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_611_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_10
  simp only [← c38595, ← c38596, ← c38597] at e_611_10
  have e_611_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_11
  simp only [← c38598, ← c38599, k2, ← c38600] at e_611_11
  have hr := e_611_11
  simp only [e_611_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 611 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3474)) := by
  have c38601 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.1
  have c38602 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38603 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38604 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38605 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38606 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_611_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_12
  simp only [← c38601, ← c38602, ← c38603] at e_611_12
  have e_611_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_13
  simp only [← c38604, ← c38605, k2, ← c38606] at e_611_13
  have hr := e_611_13
  simp only [e_611_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 611 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3475)) := by
  have c38607 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38608 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38609 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38610 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38611 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38612 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_611_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_14
  simp only [← c38607, ← c38608, ← c38609] at e_611_14
  have e_611_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_15
  simp only [← c38610, ← c38611, k2, ← c38612] at e_611_15
  have hr := e_611_15
  simp only [e_611_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 611 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3476)) := by
  have c38613 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38614 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38615 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38616 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38617 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38618 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_611_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_16
  simp only [← c38613, ← c38614, ← c38615] at e_611_16
  have e_611_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_17
  simp only [← c38616, ← c38617, k2, ← c38618] at e_611_17
  have hr := e_611_17
  simp only [e_611_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 611 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3477)) := by
  have c38619 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38620 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38621 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38622 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38623 := (publicBatchWrapper86_copies1206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38624 := (publicBatchWrapper86_copies1207 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_611_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_18
  simp only [← c38619, ← c38620, ← c38621] at e_611_18
  have e_611_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_611_19
  simp only [← c38622, ← c38623, k2, ← c38624] at e_611_19
  have hr := e_611_19
  simp only [e_611_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 612 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3478)) := by
  have c38625 := (publicBatchWrapper86_copies1207 a h).2.1
  have c38626 := (publicBatchWrapper86_copies1207 a h).2.2.1
  have c38627 := (publicBatchWrapper86_copies1207 a h).2.2.2.1
  have c38628 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.1
  have c38629 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.1
  have c38630 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_612_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_0
  simp only [← c38625, ← c38626, ← c38627] at e_612_0
  have e_612_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_1
  simp only [← c38628, ← c38629, k2, ← c38630] at e_612_1
  have hr := e_612_1
  simp only [e_612_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 612 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3479)) := by
  have c38631 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.1
  have c38632 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.1
  have c38633 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.1
  have c38634 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38635 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38636 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_612_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_2
  simp only [← c38631, ← c38632, ← c38633] at e_612_2
  have e_612_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_3
  simp only [← c38634, ← c38635, k2, ← c38636] at e_612_3
  have hr := e_612_3
  simp only [e_612_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 612 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3480)) := by
  have c38637 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38638 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38639 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38640 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38641 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38642 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_612_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_4
  simp only [← c38637, ← c38638, ← c38639] at e_612_4
  have e_612_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_5
  simp only [← c38640, ← c38641, k2, ← c38642] at e_612_5
  have hr := e_612_5
  simp only [e_612_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 612 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3481)) := by
  have c38643 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38644 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38645 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38646 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38647 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38648 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_612_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_6
  simp only [← c38643, ← c38644, ← c38645] at e_612_6
  have e_612_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_7
  simp only [← c38646, ← c38647, k2, ← c38648] at e_612_7
  have hr := e_612_7
  simp only [e_612_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 612 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3482)) := by
  have c38649 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38650 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38651 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38652 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38653 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38654 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_612_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_8
  simp only [← c38649, ← c38650, ← c38651] at e_612_8
  have e_612_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_9
  simp only [← c38652, ← c38653, k2, ← c38654] at e_612_9
  have hr := e_612_9
  simp only [e_612_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 612 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3483)) := by
  have c38655 := (publicBatchWrapper86_copies1207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38656 := (publicBatchWrapper86_copies1208 a h).1
  have c38657 := (publicBatchWrapper86_copies1208 a h).2.1
  have c38658 := (publicBatchWrapper86_copies1208 a h).2.2.1
  have c38659 := (publicBatchWrapper86_copies1208 a h).2.2.2.1
  have c38660 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_612_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_10
  simp only [← c38655, ← c38656, ← c38657] at e_612_10
  have e_612_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_11
  simp only [← c38658, ← c38659, k2, ← c38660] at e_612_11
  have hr := e_612_11
  simp only [e_612_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 612 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3484)) := by
  have c38661 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.1
  have c38662 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.1
  have c38663 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.1
  have c38664 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.1
  have c38665 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.1
  have c38666 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_612_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_12
  simp only [← c38661, ← c38662, ← c38663] at e_612_12
  have e_612_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_13
  simp only [← c38664, ← c38665, k2, ← c38666] at e_612_13
  have hr := e_612_13
  simp only [e_612_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 612 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3485)) := by
  have c38667 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38668 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38669 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38670 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38671 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38672 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_612_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_14
  simp only [← c38667, ← c38668, ← c38669] at e_612_14
  have e_612_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_15
  simp only [← c38670, ← c38671, k2, ← c38672] at e_612_15
  have hr := e_612_15
  simp only [e_612_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 612 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3486)) := by
  have c38673 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38674 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38675 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38676 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38677 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38678 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_612_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_16
  simp only [← c38673, ← c38674, ← c38675] at e_612_16
  have e_612_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_17
  simp only [← c38676, ← c38677, k2, ← c38678] at e_612_17
  have hr := e_612_17
  simp only [e_612_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 612 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3487)) := by
  have c38679 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38680 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38681 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38682 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38683 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38684 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_612_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_18
  simp only [← c38679, ← c38680, ← c38681] at e_612_18
  have e_612_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_612_19
  simp only [← c38682, ← c38683, k2, ← c38684] at e_612_19
  have hr := e_612_19
  simp only [e_612_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 613 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3488)) := by
  have c38685 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38686 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38687 := (publicBatchWrapper86_copies1208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38688 := (publicBatchWrapper86_copies1209 a h).1
  have c38689 := (publicBatchWrapper86_copies1209 a h).2.1
  have c38690 := (publicBatchWrapper86_copies1209 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_613_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_0
  simp only [← c38685, ← c38686, ← c38687] at e_613_0
  have e_613_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_1
  simp only [← c38688, ← c38689, k2, ← c38690] at e_613_1
  have hr := e_613_1
  simp only [e_613_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 613 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3489)) := by
  have c38691 := (publicBatchWrapper86_copies1209 a h).2.2.2.1
  have c38692 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.1
  have c38693 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.1
  have c38694 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.1
  have c38695 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.1
  have c38696 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_613_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_2
  simp only [← c38691, ← c38692, ← c38693] at e_613_2
  have e_613_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_3
  simp only [← c38694, ← c38695, k2, ← c38696] at e_613_3
  have hr := e_613_3
  simp only [e_613_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 613 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3490)) := by
  have c38697 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.1
  have c38698 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38699 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38700 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38701 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38702 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_613_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_4
  simp only [← c38697, ← c38698, ← c38699] at e_613_4
  have e_613_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_5
  simp only [← c38700, ← c38701, k2, ← c38702] at e_613_5
  have hr := e_613_5
  simp only [e_613_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 613 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3491)) := by
  have c38703 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38704 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38705 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38706 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38707 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38708 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_613_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_6
  simp only [← c38703, ← c38704, ← c38705] at e_613_6
  have e_613_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_7
  simp only [← c38706, ← c38707, k2, ← c38708] at e_613_7
  have hr := e_613_7
  simp only [e_613_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 613 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3492)) := by
  have c38709 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38710 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38711 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38712 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38713 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38714 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_613_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_8
  simp only [← c38709, ← c38710, ← c38711] at e_613_8
  have e_613_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_9
  simp only [← c38712, ← c38713, k2, ← c38714] at e_613_9
  have hr := e_613_9
  simp only [e_613_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 613 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3493)) := by
  have c38715 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38716 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38717 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38718 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38719 := (publicBatchWrapper86_copies1209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38720 := (publicBatchWrapper86_copies1210 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_613_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_10
  simp only [← c38715, ← c38716, ← c38717] at e_613_10
  have e_613_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_11
  simp only [← c38718, ← c38719, k2, ← c38720] at e_613_11
  have hr := e_613_11
  simp only [e_613_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 613 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3494)) := by
  have c38721 := (publicBatchWrapper86_copies1210 a h).2.1
  have c38722 := (publicBatchWrapper86_copies1210 a h).2.2.1
  have c38723 := (publicBatchWrapper86_copies1210 a h).2.2.2.1
  have c38724 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.1
  have c38725 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.1
  have c38726 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_613_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_12
  simp only [← c38721, ← c38722, ← c38723] at e_613_12
  have e_613_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_13
  simp only [← c38724, ← c38725, k2, ← c38726] at e_613_13
  have hr := e_613_13
  simp only [e_613_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 613 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3495)) := by
  have c38727 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.1
  have c38728 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.1
  have c38729 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.1
  have c38730 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38731 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38732 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_613_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_14
  simp only [← c38727, ← c38728, ← c38729] at e_613_14
  have e_613_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_15
  simp only [← c38730, ← c38731, k2, ← c38732] at e_613_15
  have hr := e_613_15
  simp only [e_613_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 613 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3496)) := by
  have c38733 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38734 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38735 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38736 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38737 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38738 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_613_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_16
  simp only [← c38733, ← c38734, ← c38735] at e_613_16
  have e_613_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_17
  simp only [← c38736, ← c38737, k2, ← c38738] at e_613_17
  have hr := e_613_17
  simp only [e_613_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 613 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3497)) := by
  have c38739 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38740 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38741 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38742 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38743 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38744 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_613_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_18
  simp only [← c38739, ← c38740, ← c38741] at e_613_18
  have e_613_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_613_19
  simp only [← c38742, ← c38743, k2, ← c38744] at e_613_19
  have hr := e_613_19
  simp only [e_613_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 614 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3498)) := by
  have c38745 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38746 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38747 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38748 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38749 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38750 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_614_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_0
  simp only [← c38745, ← c38746, ← c38747] at e_614_0
  have e_614_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_1
  simp only [← c38748, ← c38749, k2, ← c38750] at e_614_1
  have hr := e_614_1
  simp only [e_614_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 614 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3499)) := by
  have c38751 := (publicBatchWrapper86_copies1210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38752 := (publicBatchWrapper86_copies1211 a h).1
  have c38753 := (publicBatchWrapper86_copies1211 a h).2.1
  have c38754 := (publicBatchWrapper86_copies1211 a h).2.2.1
  have c38755 := (publicBatchWrapper86_copies1211 a h).2.2.2.1
  have c38756 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_614_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_2
  simp only [← c38751, ← c38752, ← c38753] at e_614_2
  have e_614_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_3
  simp only [← c38754, ← c38755, k2, ← c38756] at e_614_3
  have hr := e_614_3
  simp only [e_614_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 614 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3500)) := by
  have c38757 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.1
  have c38758 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.1
  have c38759 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.1
  have c38760 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.1
  have c38761 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.1
  have c38762 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_614_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_4
  simp only [← c38757, ← c38758, ← c38759] at e_614_4
  have e_614_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_5
  simp only [← c38760, ← c38761, k2, ← c38762] at e_614_5
  have hr := e_614_5
  simp only [e_614_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 614 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3501)) := by
  have c38763 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38764 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38765 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38766 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38767 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38768 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_614_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_6
  simp only [← c38763, ← c38764, ← c38765] at e_614_6
  have e_614_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_7
  simp only [← c38766, ← c38767, k2, ← c38768] at e_614_7
  have hr := e_614_7
  simp only [e_614_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 614 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3502)) := by
  have c38769 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38770 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38771 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38772 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38773 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38774 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_614_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_8
  simp only [← c38769, ← c38770, ← c38771] at e_614_8
  have e_614_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_9
  simp only [← c38772, ← c38773, k2, ← c38774] at e_614_9
  have hr := e_614_9
  simp only [e_614_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 614 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3503)) := by
  have c38775 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38776 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38777 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38778 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38779 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38780 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_614_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_10
  simp only [← c38775, ← c38776, ← c38777] at e_614_10
  have e_614_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_11
  simp only [← c38778, ← c38779, k2, ← c38780] at e_614_11
  have hr := e_614_11
  simp only [e_614_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 614 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3504)) := by
  have c38781 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38782 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38783 := (publicBatchWrapper86_copies1211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38784 := (publicBatchWrapper86_copies1212 a h).1
  have c38785 := (publicBatchWrapper86_copies1212 a h).2.1
  have c38786 := (publicBatchWrapper86_copies1212 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_614_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_12
  simp only [← c38781, ← c38782, ← c38783] at e_614_12
  have e_614_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_13
  simp only [← c38784, ← c38785, k2, ← c38786] at e_614_13
  have hr := e_614_13
  simp only [e_614_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 614 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3505)) := by
  have c38787 := (publicBatchWrapper86_copies1212 a h).2.2.2.1
  have c38788 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.1
  have c38789 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.1
  have c38790 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.1
  have c38791 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.1
  have c38792 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_614_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_14
  simp only [← c38787, ← c38788, ← c38789] at e_614_14
  have e_614_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_15
  simp only [← c38790, ← c38791, k2, ← c38792] at e_614_15
  have hr := e_614_15
  simp only [e_614_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 614 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3506)) := by
  have c38793 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.1
  have c38794 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38795 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38796 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38797 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38798 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_614_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_16
  simp only [← c38793, ← c38794, ← c38795] at e_614_16
  have e_614_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_17
  simp only [← c38796, ← c38797, k2, ← c38798] at e_614_17
  have hr := e_614_17
  simp only [e_614_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 614 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3507)) := by
  have c38799 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38800 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38801 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38802 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38803 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38804 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_614_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_18
  simp only [← c38799, ← c38800, ← c38801] at e_614_18
  have e_614_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_614_19
  simp only [← c38802, ← c38803, k2, ← c38804] at e_614_19
  have hr := e_614_19
  simp only [e_614_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 615 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3508)) := by
  have c38805 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38806 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38807 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38808 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38809 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38810 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_615_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_0
  simp only [← c38805, ← c38806, ← c38807] at e_615_0
  have e_615_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_1
  simp only [← c38808, ← c38809, k2, ← c38810] at e_615_1
  have hr := e_615_1
  simp only [e_615_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 615 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3509)) := by
  have c38811 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38812 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38813 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38814 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38815 := (publicBatchWrapper86_copies1212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38816 := (publicBatchWrapper86_copies1213 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_615_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_2
  simp only [← c38811, ← c38812, ← c38813] at e_615_2
  have e_615_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_3
  simp only [← c38814, ← c38815, k2, ← c38816] at e_615_3
  have hr := e_615_3
  simp only [e_615_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 615 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3510)) := by
  have c38817 := (publicBatchWrapper86_copies1213 a h).2.1
  have c38818 := (publicBatchWrapper86_copies1213 a h).2.2.1
  have c38819 := (publicBatchWrapper86_copies1213 a h).2.2.2.1
  have c38820 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.1
  have c38821 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.1
  have c38822 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_615_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_4
  simp only [← c38817, ← c38818, ← c38819] at e_615_4
  have e_615_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_5
  simp only [← c38820, ← c38821, k2, ← c38822] at e_615_5
  have hr := e_615_5
  simp only [e_615_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 615 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3511)) := by
  have c38823 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.1
  have c38824 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.1
  have c38825 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.1
  have c38826 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38827 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38828 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_615_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_6
  simp only [← c38823, ← c38824, ← c38825] at e_615_6
  have e_615_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_7
  simp only [← c38826, ← c38827, k2, ← c38828] at e_615_7
  have hr := e_615_7
  simp only [e_615_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 615 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3512)) := by
  have c38829 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38830 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38831 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38832 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38833 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38834 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_615_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_8
  simp only [← c38829, ← c38830, ← c38831] at e_615_8
  have e_615_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_9
  simp only [← c38832, ← c38833, k2, ← c38834] at e_615_9
  have hr := e_615_9
  simp only [e_615_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 615 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3513)) := by
  have c38835 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38836 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38837 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38838 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38839 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38840 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_615_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_10
  simp only [← c38835, ← c38836, ← c38837] at e_615_10
  have e_615_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_11
  simp only [← c38838, ← c38839, k2, ← c38840] at e_615_11
  have hr := e_615_11
  simp only [e_615_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 615 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3514)) := by
  have c38841 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38842 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38843 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38844 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38845 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38846 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_615_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_12
  simp only [← c38841, ← c38842, ← c38843] at e_615_12
  have e_615_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_13
  simp only [← c38844, ← c38845, k2, ← c38846] at e_615_13
  have hr := e_615_13
  simp only [e_615_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 615 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3515)) := by
  have c38847 := (publicBatchWrapper86_copies1213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38848 := (publicBatchWrapper86_copies1214 a h).1
  have c38849 := (publicBatchWrapper86_copies1214 a h).2.1
  have c38850 := (publicBatchWrapper86_copies1214 a h).2.2.1
  have c38851 := (publicBatchWrapper86_copies1214 a h).2.2.2.1
  have c38852 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_615_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_14
  simp only [← c38847, ← c38848, ← c38849] at e_615_14
  have e_615_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_15
  simp only [← c38850, ← c38851, k2, ← c38852] at e_615_15
  have hr := e_615_15
  simp only [e_615_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 615 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3516)) := by
  have c38853 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.1
  have c38854 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.1
  have c38855 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.1
  have c38856 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.1
  have c38857 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.1
  have c38858 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_615_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_16
  simp only [← c38853, ← c38854, ← c38855] at e_615_16
  have e_615_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_17
  simp only [← c38856, ← c38857, k2, ← c38858] at e_615_17
  have hr := e_615_17
  simp only [e_615_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 615 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3517)) := by
  have c38859 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38860 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38861 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38862 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38863 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38864 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_615_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_18
  simp only [← c38859, ← c38860, ← c38861] at e_615_18
  have e_615_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_615_19
  simp only [← c38862, ← c38863, k2, ← c38864] at e_615_19
  have hr := e_615_19
  simp only [e_615_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 616 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3518)) := by
  have c38865 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38866 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38867 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38868 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38869 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38870 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_616_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_0
  simp only [← c38865, ← c38866, ← c38867] at e_616_0
  have e_616_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_1
  simp only [← c38868, ← c38869, k2, ← c38870] at e_616_1
  have hr := e_616_1
  simp only [e_616_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 616 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3519)) := by
  have c38871 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38872 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38873 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38874 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38875 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38876 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_616_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_2
  simp only [← c38871, ← c38872, ← c38873] at e_616_2
  have e_616_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_3
  simp only [← c38874, ← c38875, k2, ← c38876] at e_616_3
  have hr := e_616_3
  simp only [e_616_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 616 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3520)) := by
  have c38877 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38878 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38879 := (publicBatchWrapper86_copies1214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38880 := (publicBatchWrapper86_copies1215 a h).1
  have c38881 := (publicBatchWrapper86_copies1215 a h).2.1
  have c38882 := (publicBatchWrapper86_copies1215 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_616_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_4
  simp only [← c38877, ← c38878, ← c38879] at e_616_4
  have e_616_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_5
  simp only [← c38880, ← c38881, k2, ← c38882] at e_616_5
  have hr := e_616_5
  simp only [e_616_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 616 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3521)) := by
  have c38883 := (publicBatchWrapper86_copies1215 a h).2.2.2.1
  have c38884 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.1
  have c38885 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.1
  have c38886 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.1
  have c38887 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.1
  have c38888 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_616_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_6
  simp only [← c38883, ← c38884, ← c38885] at e_616_6
  have e_616_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_7
  simp only [← c38886, ← c38887, k2, ← c38888] at e_616_7
  have hr := e_616_7
  simp only [e_616_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 616 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3522)) := by
  have c38889 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.1
  have c38890 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38891 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38892 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38893 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38894 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_616_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_8
  simp only [← c38889, ← c38890, ← c38891] at e_616_8
  have e_616_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_9
  simp only [← c38892, ← c38893, k2, ← c38894] at e_616_9
  have hr := e_616_9
  simp only [e_616_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 616 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3523)) := by
  have c38895 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38896 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38897 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38898 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38899 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38900 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_616_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_10
  simp only [← c38895, ← c38896, ← c38897] at e_616_10
  have e_616_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_11
  simp only [← c38898, ← c38899, k2, ← c38900] at e_616_11
  have hr := e_616_11
  simp only [e_616_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 616 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3524)) := by
  have c38901 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38902 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38903 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38904 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38905 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38906 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_616_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_12
  simp only [← c38901, ← c38902, ← c38903] at e_616_12
  have e_616_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_13
  simp only [← c38904, ← c38905, k2, ← c38906] at e_616_13
  have hr := e_616_13
  simp only [e_616_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 616 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3525)) := by
  have c38907 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38908 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38909 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38910 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38911 := (publicBatchWrapper86_copies1215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38912 := (publicBatchWrapper86_copies1216 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_616_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_14
  simp only [← c38907, ← c38908, ← c38909] at e_616_14
  have e_616_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_15
  simp only [← c38910, ← c38911, k2, ← c38912] at e_616_15
  have hr := e_616_15
  simp only [e_616_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 616 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3526)) := by
  have c38913 := (publicBatchWrapper86_copies1216 a h).2.1
  have c38914 := (publicBatchWrapper86_copies1216 a h).2.2.1
  have c38915 := (publicBatchWrapper86_copies1216 a h).2.2.2.1
  have c38916 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.1
  have c38917 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.1
  have c38918 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_616_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_16
  simp only [← c38913, ← c38914, ← c38915] at e_616_16
  have e_616_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_17
  simp only [← c38916, ← c38917, k2, ← c38918] at e_616_17
  have hr := e_616_17
  simp only [e_616_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 616 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3527)) := by
  have c38919 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.1
  have c38920 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.1
  have c38921 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.1
  have c38922 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38923 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38924 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_616_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_18
  simp only [← c38919, ← c38920, ← c38921] at e_616_18
  have e_616_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_616_19
  simp only [← c38922, ← c38923, k2, ← c38924] at e_616_19
  have hr := e_616_19
  simp only [e_616_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 617 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3528)) := by
  have c38925 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38926 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38927 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38928 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38929 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38930 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_617_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_0
  simp only [← c38925, ← c38926, ← c38927] at e_617_0
  have e_617_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_1
  simp only [← c38928, ← c38929, k2, ← c38930] at e_617_1
  have hr := e_617_1
  simp only [e_617_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 617 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3529)) := by
  have c38931 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38932 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38933 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38934 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38935 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38936 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_617_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_2
  simp only [← c38931, ← c38932, ← c38933] at e_617_2
  have e_617_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_3
  simp only [← c38934, ← c38935, k2, ← c38936] at e_617_3
  have hr := e_617_3
  simp only [e_617_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 617 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3530)) := by
  have c38937 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38938 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38939 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38940 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38941 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38942 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_617_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_4
  simp only [← c38937, ← c38938, ← c38939] at e_617_4
  have e_617_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_5
  simp only [← c38940, ← c38941, k2, ← c38942] at e_617_5
  have hr := e_617_5
  simp only [e_617_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 617 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3531)) := by
  have c38943 := (publicBatchWrapper86_copies1216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38944 := (publicBatchWrapper86_copies1217 a h).1
  have c38945 := (publicBatchWrapper86_copies1217 a h).2.1
  have c38946 := (publicBatchWrapper86_copies1217 a h).2.2.1
  have c38947 := (publicBatchWrapper86_copies1217 a h).2.2.2.1
  have c38948 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_617_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_6
  simp only [← c38943, ← c38944, ← c38945] at e_617_6
  have e_617_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_7
  simp only [← c38946, ← c38947, k2, ← c38948] at e_617_7
  have hr := e_617_7
  simp only [e_617_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 617 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3532)) := by
  have c38949 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.1
  have c38950 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.1
  have c38951 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.1
  have c38952 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.1
  have c38953 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.1
  have c38954 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_617_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_8
  simp only [← c38949, ← c38950, ← c38951] at e_617_8
  have e_617_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_9
  simp only [← c38952, ← c38953, k2, ← c38954] at e_617_9
  have hr := e_617_9
  simp only [e_617_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 617 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3533)) := by
  have c38955 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38956 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38957 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38958 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38959 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38960 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_617_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_10
  simp only [← c38955, ← c38956, ← c38957] at e_617_10
  have e_617_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_11
  simp only [← c38958, ← c38959, k2, ← c38960] at e_617_11
  have hr := e_617_11
  simp only [e_617_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 617 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3534)) := by
  have c38961 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38962 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38963 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38964 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38965 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38966 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_617_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_12
  simp only [← c38961, ← c38962, ← c38963] at e_617_12
  have e_617_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_13
  simp only [← c38964, ← c38965, k2, ← c38966] at e_617_13
  have hr := e_617_13
  simp only [e_617_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 617 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3535)) := by
  have c38967 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38968 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38969 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38970 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38971 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38972 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_617_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_14
  simp only [← c38967, ← c38968, ← c38969] at e_617_14
  have e_617_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_15
  simp only [← c38970, ← c38971, k2, ← c38972] at e_617_15
  have hr := e_617_15
  simp only [e_617_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 617 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3536)) := by
  have c38973 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38974 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38975 := (publicBatchWrapper86_copies1217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c38976 := (publicBatchWrapper86_copies1218 a h).1
  have c38977 := (publicBatchWrapper86_copies1218 a h).2.1
  have c38978 := (publicBatchWrapper86_copies1218 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_617_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_16
  simp only [← c38973, ← c38974, ← c38975] at e_617_16
  have e_617_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_17
  simp only [← c38976, ← c38977, k2, ← c38978] at e_617_17
  have hr := e_617_17
  simp only [e_617_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 617 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3537)) := by
  have c38979 := (publicBatchWrapper86_copies1218 a h).2.2.2.1
  have c38980 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.1
  have c38981 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.1
  have c38982 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.1
  have c38983 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.1
  have c38984 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_617_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_18
  simp only [← c38979, ← c38980, ← c38981] at e_617_18
  have e_617_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_617_19
  simp only [← c38982, ← c38983, k2, ← c38984] at e_617_19
  have hr := e_617_19
  simp only [e_617_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 618 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3538)) := by
  have c38985 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.1
  have c38986 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.1
  have c38987 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c38988 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38989 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38990 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_618_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_0
  simp only [← c38985, ← c38986, ← c38987] at e_618_0
  have e_618_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_1
  simp only [← c38988, ← c38989, k2, ← c38990] at e_618_1
  have hr := e_618_1
  simp only [e_618_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 618 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3539)) := by
  have c38991 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38992 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38993 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38994 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38995 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38996 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_618_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_2
  simp only [← c38991, ← c38992, ← c38993] at e_618_2
  have e_618_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_3
  simp only [← c38994, ← c38995, k2, ← c38996] at e_618_3
  have hr := e_618_3
  simp only [e_618_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 618 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3540)) := by
  have c38997 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38998 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c38999 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39000 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39001 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39002 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_618_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_4
  simp only [← c38997, ← c38998, ← c38999] at e_618_4
  have e_618_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_5
  simp only [← c39000, ← c39001, k2, ← c39002] at e_618_5
  have hr := e_618_5
  simp only [e_618_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 618 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3541)) := by
  have c39003 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39004 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39005 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39006 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39007 := (publicBatchWrapper86_copies1218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39008 := (publicBatchWrapper86_copies1219 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_618_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_6
  simp only [← c39003, ← c39004, ← c39005] at e_618_6
  have e_618_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_7
  simp only [← c39006, ← c39007, k2, ← c39008] at e_618_7
  have hr := e_618_7
  simp only [e_618_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 618 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3542)) := by
  have c39009 := (publicBatchWrapper86_copies1219 a h).2.1
  have c39010 := (publicBatchWrapper86_copies1219 a h).2.2.1
  have c39011 := (publicBatchWrapper86_copies1219 a h).2.2.2.1
  have c39012 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.1
  have c39013 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.1
  have c39014 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_618_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_8
  simp only [← c39009, ← c39010, ← c39011] at e_618_8
  have e_618_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_9
  simp only [← c39012, ← c39013, k2, ← c39014] at e_618_9
  have hr := e_618_9
  simp only [e_618_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 618 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3543)) := by
  have c39015 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.1
  have c39016 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.1
  have c39017 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.1
  have c39018 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39019 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39020 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_618_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_10
  simp only [← c39015, ← c39016, ← c39017] at e_618_10
  have e_618_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_11
  simp only [← c39018, ← c39019, k2, ← c39020] at e_618_11
  have hr := e_618_11
  simp only [e_618_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 618 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3544)) := by
  have c39021 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39022 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39023 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39024 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39025 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39026 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_618_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_12
  simp only [← c39021, ← c39022, ← c39023] at e_618_12
  have e_618_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_13
  simp only [← c39024, ← c39025, k2, ← c39026] at e_618_13
  have hr := e_618_13
  simp only [e_618_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 618 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3545)) := by
  have c39027 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39028 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39029 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39030 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39031 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39032 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_618_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_14
  simp only [← c39027, ← c39028, ← c39029] at e_618_14
  have e_618_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_15
  simp only [← c39030, ← c39031, k2, ← c39032] at e_618_15
  have hr := e_618_15
  simp only [e_618_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 618 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3546)) := by
  have c39033 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39034 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39035 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39036 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39037 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39038 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_618_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_16
  simp only [← c39033, ← c39034, ← c39035] at e_618_16
  have e_618_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_17
  simp only [← c39036, ← c39037, k2, ← c39038] at e_618_17
  have hr := e_618_17
  simp only [e_618_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5781 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 618 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3547)) := by
  have c39039 := (publicBatchWrapper86_copies1219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39040 := (publicBatchWrapper86_copies1220 a h).1
  have c39041 := (publicBatchWrapper86_copies1220 a h).2.1
  have c39042 := (publicBatchWrapper86_copies1220 a h).2.2.1
  have c39043 := (publicBatchWrapper86_copies1220 a h).2.2.2.1
  have c39044 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_618_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_18
  simp only [← c39039, ← c39040, ← c39041] at e_618_18
  have e_618_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_618_19
  simp only [← c39042, ← c39043, k2, ← c39044] at e_618_19
  have hr := e_618_19
  simp only [e_618_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5782 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 619 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3548)) := by
  have c39045 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.1
  have c39046 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.1
  have c39047 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.1
  have c39048 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.1
  have c39049 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.1
  have c39050 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_619_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_0
  simp only [← c39045, ← c39046, ← c39047] at e_619_0
  have e_619_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_1
  simp only [← c39048, ← c39049, k2, ← c39050] at e_619_1
  have hr := e_619_1
  simp only [e_619_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5783 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 619 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3549)) := by
  have c39051 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39052 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39053 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39054 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39055 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39056 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_619_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_2
  simp only [← c39051, ← c39052, ← c39053] at e_619_2
  have e_619_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_3
  simp only [← c39054, ← c39055, k2, ← c39056] at e_619_3
  have hr := e_619_3
  simp only [e_619_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5784 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 619 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3550)) := by
  have c39057 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39058 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39059 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39060 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39061 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39062 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_619_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_4
  simp only [← c39057, ← c39058, ← c39059] at e_619_4
  have e_619_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_5
  simp only [← c39060, ← c39061, k2, ← c39062] at e_619_5
  have hr := e_619_5
  simp only [e_619_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5785 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 619 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3551)) := by
  have c39063 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39064 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39065 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39066 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39067 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39068 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_619_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_6
  simp only [← c39063, ← c39064, ← c39065] at e_619_6
  have e_619_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_7
  simp only [← c39066, ← c39067, k2, ← c39068] at e_619_7
  have hr := e_619_7
  simp only [e_619_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5786 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 619 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3552)) := by
  have c39069 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39070 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39071 := (publicBatchWrapper86_copies1220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39072 := (publicBatchWrapper86_copies1221 a h).1
  have c39073 := (publicBatchWrapper86_copies1221 a h).2.1
  have c39074 := (publicBatchWrapper86_copies1221 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_619_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_8
  simp only [← c39069, ← c39070, ← c39071] at e_619_8
  have e_619_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_9
  simp only [← c39072, ← c39073, k2, ← c39074] at e_619_9
  have hr := e_619_9
  simp only [e_619_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5787 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 619 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3553)) := by
  have c39075 := (publicBatchWrapper86_copies1221 a h).2.2.2.1
  have c39076 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.1
  have c39077 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.1
  have c39078 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.1
  have c39079 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.1
  have c39080 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_619_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_10
  simp only [← c39075, ← c39076, ← c39077] at e_619_10
  have e_619_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_11
  simp only [← c39078, ← c39079, k2, ← c39080] at e_619_11
  have hr := e_619_11
  simp only [e_619_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5788 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 619 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3554)) := by
  have c39081 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.1
  have c39082 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39083 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39084 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39085 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39086 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_619_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_12
  simp only [← c39081, ← c39082, ← c39083] at e_619_12
  have e_619_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_13
  simp only [← c39084, ← c39085, k2, ← c39086] at e_619_13
  have hr := e_619_13
  simp only [e_619_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5789 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 619 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3555)) := by
  have c39087 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39088 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39089 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39090 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39091 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39092 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_619_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_14
  simp only [← c39087, ← c39088, ← c39089] at e_619_14
  have e_619_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_15
  simp only [← c39090, ← c39091, k2, ← c39092] at e_619_15
  have hr := e_619_15
  simp only [e_619_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5790 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 619 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3556)) := by
  have c39093 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39094 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39095 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39096 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39097 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39098 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_619_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_16
  simp only [← c39093, ← c39094, ← c39095] at e_619_16
  have e_619_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_17
  simp only [← c39096, ← c39097, k2, ← c39098] at e_619_17
  have hr := e_619_17
  simp only [e_619_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5791 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 619 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3557)) := by
  have c39099 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39100 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39101 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39102 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39103 := (publicBatchWrapper86_copies1221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39104 := (publicBatchWrapper86_copies1222 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_619_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_18
  simp only [← c39099, ← c39100, ← c39101] at e_619_18
  have e_619_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_619_19
  simp only [← c39102, ← c39103, k2, ← c39104] at e_619_19
  have hr := e_619_19
  simp only [e_619_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5792 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 620 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3558)) := by
  have c39105 := (publicBatchWrapper86_copies1222 a h).2.1
  have c39106 := (publicBatchWrapper86_copies1222 a h).2.2.1
  have c39107 := (publicBatchWrapper86_copies1222 a h).2.2.2.1
  have c39108 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.1
  have c39109 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.1
  have c39110 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_620_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_0
  simp only [← c39105, ← c39106, ← c39107] at e_620_0
  have e_620_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_1
  simp only [← c39108, ← c39109, k2, ← c39110] at e_620_1
  have hr := e_620_1
  simp only [e_620_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5793 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 620 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3559)) := by
  have c39111 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.1
  have c39112 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.1
  have c39113 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.1
  have c39114 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39115 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39116 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_620_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_2
  simp only [← c39111, ← c39112, ← c39113] at e_620_2
  have e_620_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_3
  simp only [← c39114, ← c39115, k2, ← c39116] at e_620_3
  have hr := e_620_3
  simp only [e_620_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5794 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 620 23) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3560)) := by
  have c39117 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39118 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39119 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39120 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39121 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39122 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_620_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_4
  simp only [← c39117, ← c39118, ← c39119] at e_620_4
  have e_620_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_5
  simp only [← c39120, ← c39121, k2, ← c39122] at e_620_5
  have hr := e_620_5
  simp only [e_620_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5795 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 620 31) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3561)) := by
  have c39123 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39124 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39125 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39126 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39127 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39128 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_620_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_6
  simp only [← c39123, ← c39124, ← c39125] at e_620_6
  have e_620_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_7
  simp only [← c39126, ← c39127, k2, ← c39128] at e_620_7
  have hr := e_620_7
  simp only [e_620_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5796 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 620 39) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3562)) := by
  have c39129 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39130 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39131 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39132 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39133 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39134 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_620_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_8
  simp only [← c39129, ← c39130, ← c39131] at e_620_8
  have e_620_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_9
  simp only [← c39132, ← c39133, k2, ← c39134] at e_620_9
  have hr := e_620_9
  simp only [e_620_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5797 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 620 47) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3563)) := by
  have c39135 := (publicBatchWrapper86_copies1222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39136 := (publicBatchWrapper86_copies1223 a h).1
  have c39137 := (publicBatchWrapper86_copies1223 a h).2.1
  have c39138 := (publicBatchWrapper86_copies1223 a h).2.2.1
  have c39139 := (publicBatchWrapper86_copies1223 a h).2.2.2.1
  have c39140 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_620_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_10
  simp only [← c39135, ← c39136, ← c39137] at e_620_10
  have e_620_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_11
  simp only [← c39138, ← c39139, k2, ← c39140] at e_620_11
  have hr := e_620_11
  simp only [e_620_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5798 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 620 55) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3564)) := by
  have c39141 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.1
  have c39142 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.1
  have c39143 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.1
  have c39144 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.1
  have c39145 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.1
  have c39146 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_620_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_12
  simp only [← c39141, ← c39142, ← c39143] at e_620_12
  have e_620_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_13
  simp only [← c39144, ← c39145, k2, ← c39146] at e_620_13
  have hr := e_620_13
  simp only [e_620_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5799 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 620 63) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3565)) := by
  have c39147 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39148 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39149 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39150 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39151 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39152 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_620_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_14
  simp only [← c39147, ← c39148, ← c39149] at e_620_14
  have e_620_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_15
  simp only [← c39150, ← c39151, k2, ← c39152] at e_620_15
  have hr := e_620_15
  simp only [e_620_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 620 71) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3566)) := by
  have c39153 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39154 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39155 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39156 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39157 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39158 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_620_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_16
  simp only [← c39153, ← c39154, ← c39155] at e_620_16
  have e_620_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_17
  simp only [← c39156, ← c39157, k2, ← c39158] at e_620_17
  have hr := e_620_17
  simp only [e_620_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 620 79) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3567)) := by
  have c39159 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39160 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39161 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39162 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39163 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39164 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_620_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_18
  simp only [← c39159, ← c39160, ← c39161] at e_620_18
  have e_620_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_620_19
  simp only [← c39162, ← c39163, k2, ← c39164] at e_620_19
  have hr := e_620_19
  simp only [e_620_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 621 7) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3568)) := by
  have c39165 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39166 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39167 := (publicBatchWrapper86_copies1223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39168 := (publicBatchWrapper86_copies1224 a h).1
  have c39169 := (publicBatchWrapper86_copies1224 a h).2.1
  have c39170 := (publicBatchWrapper86_copies1224 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_621_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_0
  simp only [← c39165, ← c39166, ← c39167] at e_621_0
  have e_621_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_1
  simp only [← c39168, ← c39169, k2, ← c39170] at e_621_1
  have hr := e_621_1
  simp only [e_621_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 621 15) = bselect (a (.wire 18 55)) (a (.virt 200)) (a (.virt 3569)) := by
  have c39171 := (publicBatchWrapper86_copies1224 a h).2.2.2.1
  have c39172 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.1
  have c39173 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.1
  have c39174 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.1
  have c39175 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.1
  have c39176 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_621_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_2
  simp only [← c39171, ← c39172, ← c39173] at e_621_2
  have e_621_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_3
  simp only [← c39174, ← c39175, k2, ← c39176] at e_621_3
  have hr := e_621_3
  simp only [e_621_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 621 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3610)) := by
  have c39177 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.1
  have c39178 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39179 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39180 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39181 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39182 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_621_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_4
  simp only [← c39177, ← c39178, ← c39179] at e_621_4
  have e_621_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_5
  simp only [← c39180, ← c39181, k2, ← c39182] at e_621_5
  have hr := e_621_5
  simp only [e_621_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 621 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3611)) := by
  have c39183 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39184 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39185 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39186 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39187 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39188 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_621_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_6
  simp only [← c39183, ← c39184, ← c39185] at e_621_6
  have e_621_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_7
  simp only [← c39186, ← c39187, k2, ← c39188] at e_621_7
  have hr := e_621_7
  simp only [e_621_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 621 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3612)) := by
  have c39189 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39190 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39191 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39192 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39193 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39194 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_621_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_8
  simp only [← c39189, ← c39190, ← c39191] at e_621_8
  have e_621_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_9
  simp only [← c39192, ← c39193, k2, ← c39194] at e_621_9
  have hr := e_621_9
  simp only [e_621_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 621 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3613)) := by
  have c39195 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39196 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39197 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39198 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39199 := (publicBatchWrapper86_copies1224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39200 := (publicBatchWrapper86_copies1225 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_621_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_10
  simp only [← c39195, ← c39196, ← c39197] at e_621_10
  have e_621_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_11
  simp only [← c39198, ← c39199, k2, ← c39200] at e_621_11
  have hr := e_621_11
  simp only [e_621_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 621 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3614)) := by
  have c39201 := (publicBatchWrapper86_copies1225 a h).2.1
  have c39202 := (publicBatchWrapper86_copies1225 a h).2.2.1
  have c39203 := (publicBatchWrapper86_copies1225 a h).2.2.2.1
  have c39204 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.1
  have c39205 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.1
  have c39206 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_621_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_12
  simp only [← c39201, ← c39202, ← c39203] at e_621_12
  have e_621_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_13
  simp only [← c39204, ← c39205, k2, ← c39206] at e_621_13
  have hr := e_621_13
  simp only [e_621_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 621 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3615)) := by
  have c39207 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.1
  have c39208 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.1
  have c39209 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.1
  have c39210 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39211 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39212 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_621_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_14
  simp only [← c39207, ← c39208, ← c39209] at e_621_14
  have e_621_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_15
  simp only [← c39210, ← c39211, k2, ← c39212] at e_621_15
  have hr := e_621_15
  simp only [e_621_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 621 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3616)) := by
  have c39213 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39214 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39215 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39216 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39217 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39218 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_621_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_16
  simp only [← c39213, ← c39214, ← c39215] at e_621_16
  have e_621_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_17
  simp only [← c39216, ← c39217, k2, ← c39218] at e_621_17
  have hr := e_621_17
  simp only [e_621_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 621 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3617)) := by
  have c39219 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39220 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39221 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39222 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39223 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39224 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_621_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_18
  simp only [← c39219, ← c39220, ← c39221] at e_621_18
  have e_621_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_621_19
  simp only [← c39222, ← c39223, k2, ← c39224] at e_621_19
  have hr := e_621_19
  simp only [e_621_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 622 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3618)) := by
  have c39225 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39226 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39227 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39228 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39229 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39230 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_622_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_0
  simp only [← c39225, ← c39226, ← c39227] at e_622_0
  have e_622_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_1
  simp only [← c39228, ← c39229, k2, ← c39230] at e_622_1
  have hr := e_622_1
  simp only [e_622_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 622 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3619)) := by
  have c39231 := (publicBatchWrapper86_copies1225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39232 := (publicBatchWrapper86_copies1226 a h).1
  have c39233 := (publicBatchWrapper86_copies1226 a h).2.1
  have c39234 := (publicBatchWrapper86_copies1226 a h).2.2.1
  have c39235 := (publicBatchWrapper86_copies1226 a h).2.2.2.1
  have c39236 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_622_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_2
  simp only [← c39231, ← c39232, ← c39233] at e_622_2
  have e_622_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_3
  simp only [← c39234, ← c39235, k2, ← c39236] at e_622_3
  have hr := e_622_3
  simp only [e_622_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 622 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3620)) := by
  have c39237 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.1
  have c39238 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.1
  have c39239 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.1
  have c39240 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.1
  have c39241 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.1
  have c39242 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_622_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_4
  simp only [← c39237, ← c39238, ← c39239] at e_622_4
  have e_622_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_5
  simp only [← c39240, ← c39241, k2, ← c39242] at e_622_5
  have hr := e_622_5
  simp only [e_622_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 622 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3621)) := by
  have c39243 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39244 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39245 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39246 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39247 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39248 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_622_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_6
  simp only [← c39243, ← c39244, ← c39245] at e_622_6
  have e_622_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_7
  simp only [← c39246, ← c39247, k2, ← c39248] at e_622_7
  have hr := e_622_7
  simp only [e_622_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 622 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3622)) := by
  have c39249 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39250 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39251 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39252 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39253 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39254 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_622_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_8
  simp only [← c39249, ← c39250, ← c39251] at e_622_8
  have e_622_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_9
  simp only [← c39252, ← c39253, k2, ← c39254] at e_622_9
  have hr := e_622_9
  simp only [e_622_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 622 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3623)) := by
  have c39255 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39256 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39257 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39258 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39259 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39260 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_622_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_10
  simp only [← c39255, ← c39256, ← c39257] at e_622_10
  have e_622_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_11
  simp only [← c39258, ← c39259, k2, ← c39260] at e_622_11
  have hr := e_622_11
  simp only [e_622_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 622 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3624)) := by
  have c39261 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39262 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39263 := (publicBatchWrapper86_copies1226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39264 := (publicBatchWrapper86_copies1227 a h).1
  have c39265 := (publicBatchWrapper86_copies1227 a h).2.1
  have c39266 := (publicBatchWrapper86_copies1227 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_622_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_12
  simp only [← c39261, ← c39262, ← c39263] at e_622_12
  have e_622_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_13
  simp only [← c39264, ← c39265, k2, ← c39266] at e_622_13
  have hr := e_622_13
  simp only [e_622_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 622 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3625)) := by
  have c39267 := (publicBatchWrapper86_copies1227 a h).2.2.2.1
  have c39268 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.1
  have c39269 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.1
  have c39270 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.1
  have c39271 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.1
  have c39272 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_622_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_14
  simp only [← c39267, ← c39268, ← c39269] at e_622_14
  have e_622_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_15
  simp only [← c39270, ← c39271, k2, ← c39272] at e_622_15
  have hr := e_622_15
  simp only [e_622_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 622 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3626)) := by
  have c39273 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.1
  have c39274 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39275 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39276 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39277 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39278 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_622_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_16
  simp only [← c39273, ← c39274, ← c39275] at e_622_16
  have e_622_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_17
  simp only [← c39276, ← c39277, k2, ← c39278] at e_622_17
  have hr := e_622_17
  simp only [e_622_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 622 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3627)) := by
  have c39279 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39280 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39281 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39282 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39283 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39284 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_622_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_18
  simp only [← c39279, ← c39280, ← c39281] at e_622_18
  have e_622_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_622_19
  simp only [← c39282, ← c39283, k2, ← c39284] at e_622_19
  have hr := e_622_19
  simp only [e_622_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 623 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3628)) := by
  have c39285 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39286 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39287 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39288 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39289 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39290 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_623_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_0
  simp only [← c39285, ← c39286, ← c39287] at e_623_0
  have e_623_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_1
  simp only [← c39288, ← c39289, k2, ← c39290] at e_623_1
  have hr := e_623_1
  simp only [e_623_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 623 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3629)) := by
  have c39291 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39292 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39293 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39294 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39295 := (publicBatchWrapper86_copies1227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39296 := (publicBatchWrapper86_copies1228 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_623_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_2
  simp only [← c39291, ← c39292, ← c39293] at e_623_2
  have e_623_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_3
  simp only [← c39294, ← c39295, k2, ← c39296] at e_623_3
  have hr := e_623_3
  simp only [e_623_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 623 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3630)) := by
  have c39297 := (publicBatchWrapper86_copies1228 a h).2.1
  have c39298 := (publicBatchWrapper86_copies1228 a h).2.2.1
  have c39299 := (publicBatchWrapper86_copies1228 a h).2.2.2.1
  have c39300 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.1
  have c39301 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.1
  have c39302 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_623_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_4
  simp only [← c39297, ← c39298, ← c39299] at e_623_4
  have e_623_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_5
  simp only [← c39300, ← c39301, k2, ← c39302] at e_623_5
  have hr := e_623_5
  simp only [e_623_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 623 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3631)) := by
  have c39303 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.1
  have c39304 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.1
  have c39305 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.1
  have c39306 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39307 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39308 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_623_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_6
  simp only [← c39303, ← c39304, ← c39305] at e_623_6
  have e_623_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_7
  simp only [← c39306, ← c39307, k2, ← c39308] at e_623_7
  have hr := e_623_7
  simp only [e_623_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5826 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 623 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3632)) := by
  have c39309 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39310 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39311 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39312 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39313 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39314 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_623_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_8
  simp only [← c39309, ← c39310, ← c39311] at e_623_8
  have e_623_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_9
  simp only [← c39312, ← c39313, k2, ← c39314] at e_623_9
  have hr := e_623_9
  simp only [e_623_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5827 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 623 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3633)) := by
  have c39315 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39316 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39317 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39318 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39319 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39320 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_623_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_10
  simp only [← c39315, ← c39316, ← c39317] at e_623_10
  have e_623_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_11
  simp only [← c39318, ← c39319, k2, ← c39320] at e_623_11
  have hr := e_623_11
  simp only [e_623_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5828 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 623 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3634)) := by
  have c39321 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39322 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39323 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39324 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39325 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39326 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_623_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_12
  simp only [← c39321, ← c39322, ← c39323] at e_623_12
  have e_623_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_13
  simp only [← c39324, ← c39325, k2, ← c39326] at e_623_13
  have hr := e_623_13
  simp only [e_623_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5829 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 623 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3635)) := by
  have c39327 := (publicBatchWrapper86_copies1228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39328 := (publicBatchWrapper86_copies1229 a h).1
  have c39329 := (publicBatchWrapper86_copies1229 a h).2.1
  have c39330 := (publicBatchWrapper86_copies1229 a h).2.2.1
  have c39331 := (publicBatchWrapper86_copies1229 a h).2.2.2.1
  have c39332 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_623_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_14
  simp only [← c39327, ← c39328, ← c39329] at e_623_14
  have e_623_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_15
  simp only [← c39330, ← c39331, k2, ← c39332] at e_623_15
  have hr := e_623_15
  simp only [e_623_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5830 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 623 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3636)) := by
  have c39333 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.1
  have c39334 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.1
  have c39335 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.1
  have c39336 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.1
  have c39337 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.1
  have c39338 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_623_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_16
  simp only [← c39333, ← c39334, ← c39335] at e_623_16
  have e_623_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_17
  simp only [← c39336, ← c39337, k2, ← c39338] at e_623_17
  have hr := e_623_17
  simp only [e_623_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5831 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 623 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3637)) := by
  have c39339 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39340 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39341 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39342 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39343 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39344 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_623_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_18
  simp only [← c39339, ← c39340, ← c39341] at e_623_18
  have e_623_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_623_19
  simp only [← c39342, ← c39343, k2, ← c39344] at e_623_19
  have hr := e_623_19
  simp only [e_623_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5832 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 624 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3638)) := by
  have c39345 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39346 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39347 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39348 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39349 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39350 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_624_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_0
  simp only [← c39345, ← c39346, ← c39347] at e_624_0
  have e_624_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_1
  simp only [← c39348, ← c39349, k2, ← c39350] at e_624_1
  have hr := e_624_1
  simp only [e_624_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5833 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 624 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3639)) := by
  have c39351 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39352 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39353 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39354 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39355 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39356 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_624_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_2
  simp only [← c39351, ← c39352, ← c39353] at e_624_2
  have e_624_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_3
  simp only [← c39354, ← c39355, k2, ← c39356] at e_624_3
  have hr := e_624_3
  simp only [e_624_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5834 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 624 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3640)) := by
  have c39357 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39358 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39359 := (publicBatchWrapper86_copies1229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39360 := (publicBatchWrapper86_copies1230 a h).1
  have c39361 := (publicBatchWrapper86_copies1230 a h).2.1
  have c39362 := (publicBatchWrapper86_copies1230 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_624_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_4
  simp only [← c39357, ← c39358, ← c39359] at e_624_4
  have e_624_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_5
  simp only [← c39360, ← c39361, k2, ← c39362] at e_624_5
  have hr := e_624_5
  simp only [e_624_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5835 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 624 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3641)) := by
  have c39363 := (publicBatchWrapper86_copies1230 a h).2.2.2.1
  have c39364 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.1
  have c39365 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.1
  have c39366 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.1
  have c39367 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.1
  have c39368 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_624_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_6
  simp only [← c39363, ← c39364, ← c39365] at e_624_6
  have e_624_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_7
  simp only [← c39366, ← c39367, k2, ← c39368] at e_624_7
  have hr := e_624_7
  simp only [e_624_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5836 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 624 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3642)) := by
  have c39369 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.1
  have c39370 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39371 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39372 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39373 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39374 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_624_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_8
  simp only [← c39369, ← c39370, ← c39371] at e_624_8
  have e_624_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_9
  simp only [← c39372, ← c39373, k2, ← c39374] at e_624_9
  have hr := e_624_9
  simp only [e_624_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5837 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 624 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3643)) := by
  have c39375 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39376 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39377 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39378 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39379 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39380 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_624_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_10
  simp only [← c39375, ← c39376, ← c39377] at e_624_10
  have e_624_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_11
  simp only [← c39378, ← c39379, k2, ← c39380] at e_624_11
  have hr := e_624_11
  simp only [e_624_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5838 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 624 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3644)) := by
  have c39381 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39382 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39383 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39384 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39385 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39386 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_624_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_12
  simp only [← c39381, ← c39382, ← c39383] at e_624_12
  have e_624_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_13
  simp only [← c39384, ← c39385, k2, ← c39386] at e_624_13
  have hr := e_624_13
  simp only [e_624_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5839 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 624 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3645)) := by
  have c39387 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39388 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39389 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39390 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39391 := (publicBatchWrapper86_copies1230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39392 := (publicBatchWrapper86_copies1231 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_624_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_14
  simp only [← c39387, ← c39388, ← c39389] at e_624_14
  have e_624_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_15
  simp only [← c39390, ← c39391, k2, ← c39392] at e_624_15
  have hr := e_624_15
  simp only [e_624_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5840 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 624 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3646)) := by
  have c39393 := (publicBatchWrapper86_copies1231 a h).2.1
  have c39394 := (publicBatchWrapper86_copies1231 a h).2.2.1
  have c39395 := (publicBatchWrapper86_copies1231 a h).2.2.2.1
  have c39396 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.1
  have c39397 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.1
  have c39398 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_624_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_16
  simp only [← c39393, ← c39394, ← c39395] at e_624_16
  have e_624_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_17
  simp only [← c39396, ← c39397, k2, ← c39398] at e_624_17
  have hr := e_624_17
  simp only [e_624_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5841 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 624 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3647)) := by
  have c39399 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.1
  have c39400 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.1
  have c39401 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.1
  have c39402 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39403 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39404 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_624_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_18
  simp only [← c39399, ← c39400, ← c39401] at e_624_18
  have e_624_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_624_19
  simp only [← c39402, ← c39403, k2, ← c39404] at e_624_19
  have hr := e_624_19
  simp only [e_624_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5842 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 625 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3648)) := by
  have c39405 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39406 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39407 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39408 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39409 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39410 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_625_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_0
  simp only [← c39405, ← c39406, ← c39407] at e_625_0
  have e_625_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_1
  simp only [← c39408, ← c39409, k2, ← c39410] at e_625_1
  have hr := e_625_1
  simp only [e_625_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5843 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 625 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3649)) := by
  have c39411 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39412 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39413 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39414 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39415 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39416 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_625_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_2
  simp only [← c39411, ← c39412, ← c39413] at e_625_2
  have e_625_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_3
  simp only [← c39414, ← c39415, k2, ← c39416] at e_625_3
  have hr := e_625_3
  simp only [e_625_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5844 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 625 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3650)) := by
  have c39417 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39418 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39419 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39420 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39421 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39422 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_625_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_4
  simp only [← c39417, ← c39418, ← c39419] at e_625_4
  have e_625_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_5
  simp only [← c39420, ← c39421, k2, ← c39422] at e_625_5
  have hr := e_625_5
  simp only [e_625_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5845 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 625 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3651)) := by
  have c39423 := (publicBatchWrapper86_copies1231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39424 := (publicBatchWrapper86_copies1232 a h).1
  have c39425 := (publicBatchWrapper86_copies1232 a h).2.1
  have c39426 := (publicBatchWrapper86_copies1232 a h).2.2.1
  have c39427 := (publicBatchWrapper86_copies1232 a h).2.2.2.1
  have c39428 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_625_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_6
  simp only [← c39423, ← c39424, ← c39425] at e_625_6
  have e_625_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_7
  simp only [← c39426, ← c39427, k2, ← c39428] at e_625_7
  have hr := e_625_7
  simp only [e_625_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5846 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 625 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3652)) := by
  have c39429 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.1
  have c39430 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.1
  have c39431 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.1
  have c39432 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.1
  have c39433 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.1
  have c39434 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_625_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_8
  simp only [← c39429, ← c39430, ← c39431] at e_625_8
  have e_625_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_9
  simp only [← c39432, ← c39433, k2, ← c39434] at e_625_9
  have hr := e_625_9
  simp only [e_625_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5847 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 625 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3653)) := by
  have c39435 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39436 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39437 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39438 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39439 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39440 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_625_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_10
  simp only [← c39435, ← c39436, ← c39437] at e_625_10
  have e_625_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_11
  simp only [← c39438, ← c39439, k2, ← c39440] at e_625_11
  have hr := e_625_11
  simp only [e_625_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5848 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 625 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3654)) := by
  have c39441 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39442 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39443 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39444 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39445 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39446 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_625_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_12
  simp only [← c39441, ← c39442, ← c39443] at e_625_12
  have e_625_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_13
  simp only [← c39444, ← c39445, k2, ← c39446] at e_625_13
  have hr := e_625_13
  simp only [e_625_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5849 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 625 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3655)) := by
  have c39447 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39448 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39449 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39450 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39451 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39452 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_625_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_14
  simp only [← c39447, ← c39448, ← c39449] at e_625_14
  have e_625_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_15
  simp only [← c39450, ← c39451, k2, ← c39452] at e_625_15
  have hr := e_625_15
  simp only [e_625_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5850 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 625 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3656)) := by
  have c39453 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39454 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39455 := (publicBatchWrapper86_copies1232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39456 := (publicBatchWrapper86_copies1233 a h).1
  have c39457 := (publicBatchWrapper86_copies1233 a h).2.1
  have c39458 := (publicBatchWrapper86_copies1233 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_625_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_16
  simp only [← c39453, ← c39454, ← c39455] at e_625_16
  have e_625_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_17
  simp only [← c39456, ← c39457, k2, ← c39458] at e_625_17
  have hr := e_625_17
  simp only [e_625_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5851 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 625 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3657)) := by
  have c39459 := (publicBatchWrapper86_copies1233 a h).2.2.2.1
  have c39460 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.1
  have c39461 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.1
  have c39462 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.1
  have c39463 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.1
  have c39464 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_625_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_18
  simp only [← c39459, ← c39460, ← c39461] at e_625_18
  have e_625_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_625_19
  simp only [← c39462, ← c39463, k2, ← c39464] at e_625_19
  have hr := e_625_19
  simp only [e_625_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5852 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 626 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3658)) := by
  have c39465 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.1
  have c39466 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39467 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39468 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39469 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39470 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_626_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_0
  simp only [← c39465, ← c39466, ← c39467] at e_626_0
  have e_626_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_1
  simp only [← c39468, ← c39469, k2, ← c39470] at e_626_1
  have hr := e_626_1
  simp only [e_626_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5853 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 626 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3659)) := by
  have c39471 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39472 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39473 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39474 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39475 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39476 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_626_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_2
  simp only [← c39471, ← c39472, ← c39473] at e_626_2
  have e_626_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_3
  simp only [← c39474, ← c39475, k2, ← c39476] at e_626_3
  have hr := e_626_3
  simp only [e_626_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5854 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 626 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3660)) := by
  have c39477 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39478 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39479 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39480 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39481 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39482 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_626_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_4
  simp only [← c39477, ← c39478, ← c39479] at e_626_4
  have e_626_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_5
  simp only [← c39480, ← c39481, k2, ← c39482] at e_626_5
  have hr := e_626_5
  simp only [e_626_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5855 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 626 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3661)) := by
  have c39483 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39484 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39485 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39486 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39487 := (publicBatchWrapper86_copies1233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39488 := (publicBatchWrapper86_copies1234 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_626_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_6
  simp only [← c39483, ← c39484, ← c39485] at e_626_6
  have e_626_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_7
  simp only [← c39486, ← c39487, k2, ← c39488] at e_626_7
  have hr := e_626_7
  simp only [e_626_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5856 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 626 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3662)) := by
  have c39489 := (publicBatchWrapper86_copies1234 a h).2.1
  have c39490 := (publicBatchWrapper86_copies1234 a h).2.2.1
  have c39491 := (publicBatchWrapper86_copies1234 a h).2.2.2.1
  have c39492 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.1
  have c39493 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.1
  have c39494 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_626_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_8
  simp only [← c39489, ← c39490, ← c39491] at e_626_8
  have e_626_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_9
  simp only [← c39492, ← c39493, k2, ← c39494] at e_626_9
  have hr := e_626_9
  simp only [e_626_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5857 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 626 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3663)) := by
  have c39495 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.1
  have c39496 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.1
  have c39497 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.1
  have c39498 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39499 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39500 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_626_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_10
  simp only [← c39495, ← c39496, ← c39497] at e_626_10
  have e_626_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_11
  simp only [← c39498, ← c39499, k2, ← c39500] at e_626_11
  have hr := e_626_11
  simp only [e_626_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5858 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 626 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3664)) := by
  have c39501 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39502 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39503 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39504 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39505 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39506 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_626_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_12
  simp only [← c39501, ← c39502, ← c39503] at e_626_12
  have e_626_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_13
  simp only [← c39504, ← c39505, k2, ← c39506] at e_626_13
  have hr := e_626_13
  simp only [e_626_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5859 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 626 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3665)) := by
  have c39507 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39508 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39509 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39510 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39511 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39512 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_626_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_14
  simp only [← c39507, ← c39508, ← c39509] at e_626_14
  have e_626_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_15
  simp only [← c39510, ← c39511, k2, ← c39512] at e_626_15
  have hr := e_626_15
  simp only [e_626_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5860 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 626 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3666)) := by
  have c39513 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39514 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39515 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39516 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39517 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39518 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_626_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_16
  simp only [← c39513, ← c39514, ← c39515] at e_626_16
  have e_626_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_17
  simp only [← c39516, ← c39517, k2, ← c39518] at e_626_17
  have hr := e_626_17
  simp only [e_626_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5861 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 626 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3667)) := by
  have c39519 := (publicBatchWrapper86_copies1234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39520 := (publicBatchWrapper86_copies1235 a h).1
  have c39521 := (publicBatchWrapper86_copies1235 a h).2.1
  have c39522 := (publicBatchWrapper86_copies1235 a h).2.2.1
  have c39523 := (publicBatchWrapper86_copies1235 a h).2.2.2.1
  have c39524 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_626_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_18
  simp only [← c39519, ← c39520, ← c39521] at e_626_18
  have e_626_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_626_19
  simp only [← c39522, ← c39523, k2, ← c39524] at e_626_19
  have hr := e_626_19
  simp only [e_626_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5862 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 627 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3668)) := by
  have c39525 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.1
  have c39526 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.1
  have c39527 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.1
  have c39528 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.1
  have c39529 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.1
  have c39530 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_627_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_0
  simp only [← c39525, ← c39526, ← c39527] at e_627_0
  have e_627_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_1
  simp only [← c39528, ← c39529, k2, ← c39530] at e_627_1
  have hr := e_627_1
  simp only [e_627_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5863 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 627 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3669)) := by
  have c39531 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39532 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39533 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39534 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39535 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39536 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_627_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_2
  simp only [← c39531, ← c39532, ← c39533] at e_627_2
  have e_627_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_3
  simp only [← c39534, ← c39535, k2, ← c39536] at e_627_3
  have hr := e_627_3
  simp only [e_627_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5864 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 627 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3670)) := by
  have c39537 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39538 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39539 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39540 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39541 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39542 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_627_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_4
  simp only [← c39537, ← c39538, ← c39539] at e_627_4
  have e_627_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_5
  simp only [← c39540, ← c39541, k2, ← c39542] at e_627_5
  have hr := e_627_5
  simp only [e_627_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5865 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 627 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3671)) := by
  have c39543 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39544 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39545 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39546 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39547 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39548 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_627_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_6
  simp only [← c39543, ← c39544, ← c39545] at e_627_6
  have e_627_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_7
  simp only [← c39546, ← c39547, k2, ← c39548] at e_627_7
  have hr := e_627_7
  simp only [e_627_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5866 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 627 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3672)) := by
  have c39549 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39550 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39551 := (publicBatchWrapper86_copies1235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39552 := (publicBatchWrapper86_copies1236 a h).1
  have c39553 := (publicBatchWrapper86_copies1236 a h).2.1
  have c39554 := (publicBatchWrapper86_copies1236 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_627_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_8
  simp only [← c39549, ← c39550, ← c39551] at e_627_8
  have e_627_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_9
  simp only [← c39552, ← c39553, k2, ← c39554] at e_627_9
  have hr := e_627_9
  simp only [e_627_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5867 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 627 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3673)) := by
  have c39555 := (publicBatchWrapper86_copies1236 a h).2.2.2.1
  have c39556 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.1
  have c39557 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.1
  have c39558 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.1
  have c39559 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.1
  have c39560 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_627_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_10
  simp only [← c39555, ← c39556, ← c39557] at e_627_10
  have e_627_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_11
  simp only [← c39558, ← c39559, k2, ← c39560] at e_627_11
  have hr := e_627_11
  simp only [e_627_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5868 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 627 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3674)) := by
  have c39561 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.1
  have c39562 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39563 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39564 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39565 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39566 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_627_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_12
  simp only [← c39561, ← c39562, ← c39563] at e_627_12
  have e_627_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_13
  simp only [← c39564, ← c39565, k2, ← c39566] at e_627_13
  have hr := e_627_13
  simp only [e_627_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5869 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 627 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3675)) := by
  have c39567 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39568 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39569 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39570 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39571 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39572 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_627_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_14
  simp only [← c39567, ← c39568, ← c39569] at e_627_14
  have e_627_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_15
  simp only [← c39570, ← c39571, k2, ← c39572] at e_627_15
  have hr := e_627_15
  simp only [e_627_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 627 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3676)) := by
  have c39573 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39574 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39575 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39576 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39577 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39578 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_627_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_16
  simp only [← c39573, ← c39574, ← c39575] at e_627_16
  have e_627_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_17
  simp only [← c39576, ← c39577, k2, ← c39578] at e_627_17
  have hr := e_627_17
  simp only [e_627_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 627 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3677)) := by
  have c39579 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39580 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39581 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39582 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39583 := (publicBatchWrapper86_copies1236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39584 := (publicBatchWrapper86_copies1237 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_627_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_18
  simp only [← c39579, ← c39580, ← c39581] at e_627_18
  have e_627_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_627_19
  simp only [← c39582, ← c39583, k2, ← c39584] at e_627_19
  have hr := e_627_19
  simp only [e_627_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 628 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3678)) := by
  have c39585 := (publicBatchWrapper86_copies1237 a h).2.1
  have c39586 := (publicBatchWrapper86_copies1237 a h).2.2.1
  have c39587 := (publicBatchWrapper86_copies1237 a h).2.2.2.1
  have c39588 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.1
  have c39589 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.1
  have c39590 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_628_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_0
  simp only [← c39585, ← c39586, ← c39587] at e_628_0
  have e_628_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_1
  simp only [← c39588, ← c39589, k2, ← c39590] at e_628_1
  have hr := e_628_1
  simp only [e_628_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 628 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3679)) := by
  have c39591 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.1
  have c39592 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.1
  have c39593 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.1
  have c39594 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39595 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39596 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_628_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_2
  simp only [← c39591, ← c39592, ← c39593] at e_628_2
  have e_628_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_3
  simp only [← c39594, ← c39595, k2, ← c39596] at e_628_3
  have hr := e_628_3
  simp only [e_628_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 628 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3680)) := by
  have c39597 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39598 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39599 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39600 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39601 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39602 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_628_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_4
  simp only [← c39597, ← c39598, ← c39599] at e_628_4
  have e_628_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_5
  simp only [← c39600, ← c39601, k2, ← c39602] at e_628_5
  have hr := e_628_5
  simp only [e_628_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 628 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3681)) := by
  have c39603 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39604 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39605 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39606 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39607 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39608 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_628_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_6
  simp only [← c39603, ← c39604, ← c39605] at e_628_6
  have e_628_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_7
  simp only [← c39606, ← c39607, k2, ← c39608] at e_628_7
  have hr := e_628_7
  simp only [e_628_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 628 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3682)) := by
  have c39609 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39610 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39611 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39612 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39613 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39614 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_628_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_8
  simp only [← c39609, ← c39610, ← c39611] at e_628_8
  have e_628_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_9
  simp only [← c39612, ← c39613, k2, ← c39614] at e_628_9
  have hr := e_628_9
  simp only [e_628_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5877 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 628 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3683)) := by
  have c39615 := (publicBatchWrapper86_copies1237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39616 := (publicBatchWrapper86_copies1238 a h).1
  have c39617 := (publicBatchWrapper86_copies1238 a h).2.1
  have c39618 := (publicBatchWrapper86_copies1238 a h).2.2.1
  have c39619 := (publicBatchWrapper86_copies1238 a h).2.2.2.1
  have c39620 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_628_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_10
  simp only [← c39615, ← c39616, ← c39617] at e_628_10
  have e_628_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_11
  simp only [← c39618, ← c39619, k2, ← c39620] at e_628_11
  have hr := e_628_11
  simp only [e_628_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5878 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 628 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3684)) := by
  have c39621 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.1
  have c39622 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.1
  have c39623 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.1
  have c39624 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.1
  have c39625 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.1
  have c39626 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_628_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_12
  simp only [← c39621, ← c39622, ← c39623] at e_628_12
  have e_628_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_13
  simp only [← c39624, ← c39625, k2, ← c39626] at e_628_13
  have hr := e_628_13
  simp only [e_628_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5879 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 628 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3685)) := by
  have c39627 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39628 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39629 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39630 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39631 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39632 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_628_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_14
  simp only [← c39627, ← c39628, ← c39629] at e_628_14
  have e_628_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_15
  simp only [← c39630, ← c39631, k2, ← c39632] at e_628_15
  have hr := e_628_15
  simp only [e_628_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5880 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 628 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3686)) := by
  have c39633 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39634 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39635 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39636 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39637 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39638 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_628_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_16
  simp only [← c39633, ← c39634, ← c39635] at e_628_16
  have e_628_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_17
  simp only [← c39636, ← c39637, k2, ← c39638] at e_628_17
  have hr := e_628_17
  simp only [e_628_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5881 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 628 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3687)) := by
  have c39639 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39640 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39641 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39642 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39643 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39644 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_628_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_18
  simp only [← c39639, ← c39640, ← c39641] at e_628_18
  have e_628_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_628_19
  simp only [← c39642, ← c39643, k2, ← c39644] at e_628_19
  have hr := e_628_19
  simp only [e_628_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5882 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 629 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3688)) := by
  have c39645 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39646 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39647 := (publicBatchWrapper86_copies1238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39648 := (publicBatchWrapper86_copies1239 a h).1
  have c39649 := (publicBatchWrapper86_copies1239 a h).2.1
  have c39650 := (publicBatchWrapper86_copies1239 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_629_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_0
  simp only [← c39645, ← c39646, ← c39647] at e_629_0
  have e_629_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_1
  simp only [← c39648, ← c39649, k2, ← c39650] at e_629_1
  have hr := e_629_1
  simp only [e_629_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5883 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 629 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3689)) := by
  have c39651 := (publicBatchWrapper86_copies1239 a h).2.2.2.1
  have c39652 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.1
  have c39653 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.1
  have c39654 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.1
  have c39655 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.1
  have c39656 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_629_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_2
  simp only [← c39651, ← c39652, ← c39653] at e_629_2
  have e_629_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_3
  simp only [← c39654, ← c39655, k2, ← c39656] at e_629_3
  have hr := e_629_3
  simp only [e_629_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5884 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 629 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3690)) := by
  have c39657 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.1
  have c39658 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39659 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39660 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39661 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39662 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_629_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_4
  simp only [← c39657, ← c39658, ← c39659] at e_629_4
  have e_629_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_5
  simp only [← c39660, ← c39661, k2, ← c39662] at e_629_5
  have hr := e_629_5
  simp only [e_629_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5885 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 629 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3691)) := by
  have c39663 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39664 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39665 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39666 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39667 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39668 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_629_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_6
  simp only [← c39663, ← c39664, ← c39665] at e_629_6
  have e_629_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_7
  simp only [← c39666, ← c39667, k2, ← c39668] at e_629_7
  have hr := e_629_7
  simp only [e_629_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5886 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 629 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3692)) := by
  have c39669 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39670 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39671 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39672 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39673 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39674 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_629_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_8
  simp only [← c39669, ← c39670, ← c39671] at e_629_8
  have e_629_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_9
  simp only [← c39672, ← c39673, k2, ← c39674] at e_629_9
  have hr := e_629_9
  simp only [e_629_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5887 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 629 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3693)) := by
  have c39675 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39676 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39677 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39678 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39679 := (publicBatchWrapper86_copies1239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39680 := (publicBatchWrapper86_copies1240 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_629_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_10
  simp only [← c39675, ← c39676, ← c39677] at e_629_10
  have e_629_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_11
  simp only [← c39678, ← c39679, k2, ← c39680] at e_629_11
  have hr := e_629_11
  simp only [e_629_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5888 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 629 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3694)) := by
  have c39681 := (publicBatchWrapper86_copies1240 a h).2.1
  have c39682 := (publicBatchWrapper86_copies1240 a h).2.2.1
  have c39683 := (publicBatchWrapper86_copies1240 a h).2.2.2.1
  have c39684 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.1
  have c39685 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.1
  have c39686 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_629_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_12
  simp only [← c39681, ← c39682, ← c39683] at e_629_12
  have e_629_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_13
  simp only [← c39684, ← c39685, k2, ← c39686] at e_629_13
  have hr := e_629_13
  simp only [e_629_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5889 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 629 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3695)) := by
  have c39687 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.1
  have c39688 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.1
  have c39689 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.1
  have c39690 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39691 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39692 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_629_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_14
  simp only [← c39687, ← c39688, ← c39689] at e_629_14
  have e_629_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_15
  simp only [← c39690, ← c39691, k2, ← c39692] at e_629_15
  have hr := e_629_15
  simp only [e_629_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5890 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 629 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3696)) := by
  have c39693 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39694 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39695 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39696 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39697 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39698 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_629_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_16
  simp only [← c39693, ← c39694, ← c39695] at e_629_16
  have e_629_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_17
  simp only [← c39696, ← c39697, k2, ← c39698] at e_629_17
  have hr := e_629_17
  simp only [e_629_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5891 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 629 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3697)) := by
  have c39699 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39700 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39701 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39702 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39703 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39704 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_629_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_18
  simp only [← c39699, ← c39700, ← c39701] at e_629_18
  have e_629_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_629_19
  simp only [← c39702, ← c39703, k2, ← c39704] at e_629_19
  have hr := e_629_19
  simp only [e_629_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5892 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 630 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3698)) := by
  have c39705 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39706 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39707 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39708 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39709 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39710 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_630_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_0
  simp only [← c39705, ← c39706, ← c39707] at e_630_0
  have e_630_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_1
  simp only [← c39708, ← c39709, k2, ← c39710] at e_630_1
  have hr := e_630_1
  simp only [e_630_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5893 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 630 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3699)) := by
  have c39711 := (publicBatchWrapper86_copies1240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39712 := (publicBatchWrapper86_copies1241 a h).1
  have c39713 := (publicBatchWrapper86_copies1241 a h).2.1
  have c39714 := (publicBatchWrapper86_copies1241 a h).2.2.1
  have c39715 := (publicBatchWrapper86_copies1241 a h).2.2.2.1
  have c39716 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_630_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_2
  simp only [← c39711, ← c39712, ← c39713] at e_630_2
  have e_630_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_3
  simp only [← c39714, ← c39715, k2, ← c39716] at e_630_3
  have hr := e_630_3
  simp only [e_630_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5894 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 630 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3700)) := by
  have c39717 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.1
  have c39718 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.1
  have c39719 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.1
  have c39720 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.1
  have c39721 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.1
  have c39722 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_630_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_4
  simp only [← c39717, ← c39718, ← c39719] at e_630_4
  have e_630_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_5
  simp only [← c39720, ← c39721, k2, ← c39722] at e_630_5
  have hr := e_630_5
  simp only [e_630_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5895 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 630 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3701)) := by
  have c39723 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39724 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39725 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39726 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39727 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39728 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_630_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_6
  simp only [← c39723, ← c39724, ← c39725] at e_630_6
  have e_630_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_7
  simp only [← c39726, ← c39727, k2, ← c39728] at e_630_7
  have hr := e_630_7
  simp only [e_630_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5896 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 630 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3702)) := by
  have c39729 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39730 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39731 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39732 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39733 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39734 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_630_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_8
  simp only [← c39729, ← c39730, ← c39731] at e_630_8
  have e_630_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_9
  simp only [← c39732, ← c39733, k2, ← c39734] at e_630_9
  have hr := e_630_9
  simp only [e_630_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5897 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 630 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3703)) := by
  have c39735 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39736 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39737 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39738 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39739 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39740 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_630_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_10
  simp only [← c39735, ← c39736, ← c39737] at e_630_10
  have e_630_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_11
  simp only [← c39738, ← c39739, k2, ← c39740] at e_630_11
  have hr := e_630_11
  simp only [e_630_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5898 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 630 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3704)) := by
  have c39741 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39742 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39743 := (publicBatchWrapper86_copies1241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39744 := (publicBatchWrapper86_copies1242 a h).1
  have c39745 := (publicBatchWrapper86_copies1242 a h).2.1
  have c39746 := (publicBatchWrapper86_copies1242 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_630_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_12
  simp only [← c39741, ← c39742, ← c39743] at e_630_12
  have e_630_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_13
  simp only [← c39744, ← c39745, k2, ← c39746] at e_630_13
  have hr := e_630_13
  simp only [e_630_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5899 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 630 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3705)) := by
  have c39747 := (publicBatchWrapper86_copies1242 a h).2.2.2.1
  have c39748 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.1
  have c39749 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.1
  have c39750 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.1
  have c39751 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.1
  have c39752 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_630_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_14
  simp only [← c39747, ← c39748, ← c39749] at e_630_14
  have e_630_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_15
  simp only [← c39750, ← c39751, k2, ← c39752] at e_630_15
  have hr := e_630_15
  simp only [e_630_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5900 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 630 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3706)) := by
  have c39753 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.1
  have c39754 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39755 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39756 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39757 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39758 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_630_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_16
  simp only [← c39753, ← c39754, ← c39755] at e_630_16
  have e_630_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_17
  simp only [← c39756, ← c39757, k2, ← c39758] at e_630_17
  have hr := e_630_17
  simp only [e_630_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5901 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 630 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3707)) := by
  have c39759 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39760 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39761 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39762 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39763 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39764 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_630_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_18
  simp only [← c39759, ← c39760, ← c39761] at e_630_18
  have e_630_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_630_19
  simp only [← c39762, ← c39763, k2, ← c39764] at e_630_19
  have hr := e_630_19
  simp only [e_630_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5902 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 631 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3708)) := by
  have c39765 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39766 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39767 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39768 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39769 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39770 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_631_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_0
  simp only [← c39765, ← c39766, ← c39767] at e_631_0
  have e_631_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_1
  simp only [← c39768, ← c39769, k2, ← c39770] at e_631_1
  have hr := e_631_1
  simp only [e_631_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5903 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 631 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3709)) := by
  have c39771 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39772 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39773 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39774 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39775 := (publicBatchWrapper86_copies1242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39776 := (publicBatchWrapper86_copies1243 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_631_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_2
  simp only [← c39771, ← c39772, ← c39773] at e_631_2
  have e_631_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_3
  simp only [← c39774, ← c39775, k2, ← c39776] at e_631_3
  have hr := e_631_3
  simp only [e_631_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5904 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 631 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3710)) := by
  have c39777 := (publicBatchWrapper86_copies1243 a h).2.1
  have c39778 := (publicBatchWrapper86_copies1243 a h).2.2.1
  have c39779 := (publicBatchWrapper86_copies1243 a h).2.2.2.1
  have c39780 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.1
  have c39781 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.1
  have c39782 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_631_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_4
  simp only [← c39777, ← c39778, ← c39779] at e_631_4
  have e_631_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_5
  simp only [← c39780, ← c39781, k2, ← c39782] at e_631_5
  have hr := e_631_5
  simp only [e_631_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5905 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 631 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3711)) := by
  have c39783 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.1
  have c39784 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.1
  have c39785 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.1
  have c39786 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39787 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39788 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_631_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_6
  simp only [← c39783, ← c39784, ← c39785] at e_631_6
  have e_631_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_7
  simp only [← c39786, ← c39787, k2, ← c39788] at e_631_7
  have hr := e_631_7
  simp only [e_631_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5906 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 631 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3712)) := by
  have c39789 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39790 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39791 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39792 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39793 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39794 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_631_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_8
  simp only [← c39789, ← c39790, ← c39791] at e_631_8
  have e_631_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_9
  simp only [← c39792, ← c39793, k2, ← c39794] at e_631_9
  have hr := e_631_9
  simp only [e_631_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5907 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 631 47) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3713)) := by
  have c39795 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39796 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39797 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39798 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39799 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39800 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_631_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_10
  simp only [← c39795, ← c39796, ← c39797] at e_631_10
  have e_631_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_11
  simp only [← c39798, ← c39799, k2, ← c39800] at e_631_11
  have hr := e_631_11
  simp only [e_631_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5908 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 631 55) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3714)) := by
  have c39801 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39802 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39803 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39804 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39805 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39806 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_631_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_12
  simp only [← c39801, ← c39802, ← c39803] at e_631_12
  have e_631_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_13
  simp only [← c39804, ← c39805, k2, ← c39806] at e_631_13
  have hr := e_631_13
  simp only [e_631_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5909 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 631 63) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3715)) := by
  have c39807 := (publicBatchWrapper86_copies1243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39808 := (publicBatchWrapper86_copies1244 a h).1
  have c39809 := (publicBatchWrapper86_copies1244 a h).2.1
  have c39810 := (publicBatchWrapper86_copies1244 a h).2.2.1
  have c39811 := (publicBatchWrapper86_copies1244 a h).2.2.2.1
  have c39812 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_631_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_14
  simp only [← c39807, ← c39808, ← c39809] at e_631_14
  have e_631_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_15
  simp only [← c39810, ← c39811, k2, ← c39812] at e_631_15
  have hr := e_631_15
  simp only [e_631_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5910 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 631 71) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3716)) := by
  have c39813 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.1
  have c39814 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.1
  have c39815 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.1
  have c39816 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.1
  have c39817 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.1
  have c39818 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_631_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_16
  simp only [← c39813, ← c39814, ← c39815] at e_631_16
  have e_631_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_17
  simp only [← c39816, ← c39817, k2, ← c39818] at e_631_17
  have hr := e_631_17
  simp only [e_631_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5911 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 631 79) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3717)) := by
  have c39819 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39820 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39821 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39822 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39823 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39824 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_631_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_18
  simp only [← c39819, ← c39820, ← c39821] at e_631_18
  have e_631_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_631_19
  simp only [← c39822, ← c39823, k2, ← c39824] at e_631_19
  have hr := e_631_19
  simp only [e_631_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5912 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 632 7) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3718)) := by
  have c39825 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39826 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39827 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39828 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39829 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39830 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_632_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_0
  simp only [← c39825, ← c39826, ← c39827] at e_632_0
  have e_632_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_1
  simp only [← c39828, ← c39829, k2, ← c39830] at e_632_1
  have hr := e_632_1
  simp only [e_632_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5913 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 632 15) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3719)) := by
  have c39831 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39832 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39833 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39834 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39835 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39836 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_632_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_2
  simp only [← c39831, ← c39832, ← c39833] at e_632_2
  have e_632_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_3
  simp only [← c39834, ← c39835, k2, ← c39836] at e_632_3
  have hr := e_632_3
  simp only [e_632_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5914 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 632 23) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3720)) := by
  have c39837 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39838 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39839 := (publicBatchWrapper86_copies1244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c39840 := (publicBatchWrapper86_copies1245 a h).1
  have c39841 := (publicBatchWrapper86_copies1245 a h).2.1
  have c39842 := (publicBatchWrapper86_copies1245 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_632_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_4
  simp only [← c39837, ← c39838, ← c39839] at e_632_4
  have e_632_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_5
  simp only [← c39840, ← c39841, k2, ← c39842] at e_632_5
  have hr := e_632_5
  simp only [e_632_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5915 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 632 31) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3721)) := by
  have c39843 := (publicBatchWrapper86_copies1245 a h).2.2.2.1
  have c39844 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.1
  have c39845 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.1
  have c39846 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.1
  have c39847 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.1
  have c39848 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_632_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_6
  simp only [← c39843, ← c39844, ← c39845] at e_632_6
  have e_632_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_7
  simp only [← c39846, ← c39847, k2, ← c39848] at e_632_7
  have hr := e_632_7
  simp only [e_632_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5916 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 632 39) = bselect (a (.wire 20 19)) (a (.virt 200)) (a (.virt 3722)) := by
  have c39849 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.1
  have c39850 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.1
  have c39851 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c39852 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39853 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c39854 := (publicBatchWrapper86_copies1245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_632_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_8
  simp only [← c39849, ← c39850, ← c39851] at e_632_8
  have e_632_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_632_9
  simp only [← c39852, ← c39853, k2, ← c39854] at e_632_9
  have hr := e_632_9
  simp only [e_632_8] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
