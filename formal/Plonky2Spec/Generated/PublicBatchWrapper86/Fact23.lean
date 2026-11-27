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

theorem publicBatchWrapper86_f17478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1788 55) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4996)) := by
  have c109221 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.1
  have c109222 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.1
  have c109223 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.1
  have c109224 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.1
  have c109225 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.1
  have c109226 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1788_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_12
  simp only [← c109221, ← c109222, ← c109223] at e_1788_12
  have e_1788_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_13
  simp only [← c109224, ← c109225, k2, ← c109226] at e_1788_13
  have hr := e_1788_13
  simp only [e_1788_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1788 63) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4997)) := by
  have c109227 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109228 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109229 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109230 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109231 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109232 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1788_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_14
  simp only [← c109227, ← c109228, ← c109229] at e_1788_14
  have e_1788_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_15
  simp only [← c109230, ← c109231, k2, ← c109232] at e_1788_15
  have hr := e_1788_15
  simp only [e_1788_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1788 71) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4998)) := by
  have c109233 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109234 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109235 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109236 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109237 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109238 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1788_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_16
  simp only [← c109233, ← c109234, ← c109235] at e_1788_16
  have e_1788_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_17
  simp only [← c109236, ← c109237, k2, ← c109238] at e_1788_17
  have hr := e_1788_17
  simp only [e_1788_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1788 79) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 4999)) := by
  have c109239 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109240 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109241 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109242 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109243 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109244 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1788_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_18
  simp only [← c109239, ← c109240, ← c109241] at e_1788_18
  have e_1788_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1788 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_19
  simp only [← c109242, ← c109243, k2, ← c109244] at e_1788_19
  have hr := e_1788_19
  simp only [e_1788_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 7) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 5000)) := by
  have c109245 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109246 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109247 := (publicBatchWrapper86_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109248 := (publicBatchWrapper86_copies3414 a h).1
  have c109249 := (publicBatchWrapper86_copies3414 a h).2.1
  have c109250 := (publicBatchWrapper86_copies3414 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_0
  simp only [← c109245, ← c109246, ← c109247] at e_1789_0
  have e_1789_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_1
  simp only [← c109248, ← c109249, k2, ← c109250] at e_1789_1
  have hr := e_1789_1
  simp only [e_1789_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 15) = bselect (a (.wire 25 43)) (a (.virt 200)) (a (.virt 5001)) := by
  have c109251 := (publicBatchWrapper86_copies3414 a h).2.2.2.1
  have c109252 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.1
  have c109253 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.1
  have c109254 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.1
  have c109255 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.1
  have c109256 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_2
  simp only [← c109251, ← c109252, ← c109253] at e_1789_2
  have e_1789_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_3
  simp only [← c109254, ← c109255, k2, ← c109256] at e_1789_3
  have hr := e_1789_3
  simp only [e_1789_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 23) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5170)) := by
  have c109257 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.1
  have c109258 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109259 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109260 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109261 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109262 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_4
  simp only [← c109257, ← c109258, ← c109259] at e_1789_4
  have e_1789_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_5
  simp only [← c109260, ← c109261, k2, ← c109262] at e_1789_5
  have hr := e_1789_5
  simp only [e_1789_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 31) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5171)) := by
  have c109263 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109264 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109265 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109266 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109267 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109268 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_6
  simp only [← c109263, ← c109264, ← c109265] at e_1789_6
  have e_1789_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_7
  simp only [← c109266, ← c109267, k2, ← c109268] at e_1789_7
  have hr := e_1789_7
  simp only [e_1789_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 39) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5172)) := by
  have c109269 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109270 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109271 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109272 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109273 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109274 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_8
  simp only [← c109269, ← c109270, ← c109271] at e_1789_8
  have e_1789_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_9
  simp only [← c109272, ← c109273, k2, ← c109274] at e_1789_9
  have hr := e_1789_9
  simp only [e_1789_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 47) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5173)) := by
  have c109275 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109276 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109277 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109278 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109279 := (publicBatchWrapper86_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109280 := (publicBatchWrapper86_copies3415 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_10
  simp only [← c109275, ← c109276, ← c109277] at e_1789_10
  have e_1789_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_11
  simp only [← c109278, ← c109279, k2, ← c109280] at e_1789_11
  have hr := e_1789_11
  simp only [e_1789_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 55) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5174)) := by
  have c109281 := (publicBatchWrapper86_copies3415 a h).2.1
  have c109282 := (publicBatchWrapper86_copies3415 a h).2.2.1
  have c109283 := (publicBatchWrapper86_copies3415 a h).2.2.2.1
  have c109284 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.1
  have c109285 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.1
  have c109286 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_12
  simp only [← c109281, ← c109282, ← c109283] at e_1789_12
  have e_1789_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_13
  simp only [← c109284, ← c109285, k2, ← c109286] at e_1789_13
  have hr := e_1789_13
  simp only [e_1789_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 63) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5175)) := by
  have c109287 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.1
  have c109288 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.1
  have c109289 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.1
  have c109290 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109291 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109292 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_14
  simp only [← c109287, ← c109288, ← c109289] at e_1789_14
  have e_1789_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_15
  simp only [← c109290, ← c109291, k2, ← c109292] at e_1789_15
  have hr := e_1789_15
  simp only [e_1789_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 71) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5176)) := by
  have c109293 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109294 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109295 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109296 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109297 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109298 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_16
  simp only [← c109293, ← c109294, ← c109295] at e_1789_16
  have e_1789_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_17
  simp only [← c109296, ← c109297, k2, ← c109298] at e_1789_17
  have hr := e_1789_17
  simp only [e_1789_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1789 79) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5177)) := by
  have c109299 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109300 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109301 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109302 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109303 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109304 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1789_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_18
  simp only [← c109299, ← c109300, ← c109301] at e_1789_18
  have e_1789_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1789 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_19
  simp only [← c109302, ← c109303, k2, ← c109304] at e_1789_19
  have hr := e_1789_19
  simp only [e_1789_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 7) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5178)) := by
  have c109305 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109306 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109307 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109308 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109309 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109310 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_0
  simp only [← c109305, ← c109306, ← c109307] at e_1790_0
  have e_1790_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_1
  simp only [← c109308, ← c109309, k2, ← c109310] at e_1790_1
  have hr := e_1790_1
  simp only [e_1790_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 15) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5179)) := by
  have c109311 := (publicBatchWrapper86_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109312 := (publicBatchWrapper86_copies3416 a h).1
  have c109313 := (publicBatchWrapper86_copies3416 a h).2.1
  have c109314 := (publicBatchWrapper86_copies3416 a h).2.2.1
  have c109315 := (publicBatchWrapper86_copies3416 a h).2.2.2.1
  have c109316 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_2
  simp only [← c109311, ← c109312, ← c109313] at e_1790_2
  have e_1790_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_3
  simp only [← c109314, ← c109315, k2, ← c109316] at e_1790_3
  have hr := e_1790_3
  simp only [e_1790_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 23) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5180)) := by
  have c109317 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.1
  have c109318 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.1
  have c109319 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.1
  have c109320 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.1
  have c109321 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.1
  have c109322 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_4
  simp only [← c109317, ← c109318, ← c109319] at e_1790_4
  have e_1790_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_5
  simp only [← c109320, ← c109321, k2, ← c109322] at e_1790_5
  have hr := e_1790_5
  simp only [e_1790_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 31) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5181)) := by
  have c109323 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109324 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109325 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109326 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109327 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109328 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_6
  simp only [← c109323, ← c109324, ← c109325] at e_1790_6
  have e_1790_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_7
  simp only [← c109326, ← c109327, k2, ← c109328] at e_1790_7
  have hr := e_1790_7
  simp only [e_1790_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 39) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5182)) := by
  have c109329 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109330 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109331 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109332 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109333 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109334 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_8
  simp only [← c109329, ← c109330, ← c109331] at e_1790_8
  have e_1790_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_9
  simp only [← c109332, ← c109333, k2, ← c109334] at e_1790_9
  have hr := e_1790_9
  simp only [e_1790_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 47) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5183)) := by
  have c109335 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109336 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109337 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109338 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109339 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109340 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_10
  simp only [← c109335, ← c109336, ← c109337] at e_1790_10
  have e_1790_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_11
  simp only [← c109338, ← c109339, k2, ← c109340] at e_1790_11
  have hr := e_1790_11
  simp only [e_1790_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 55) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5184)) := by
  have c109341 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109342 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109343 := (publicBatchWrapper86_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109344 := (publicBatchWrapper86_copies3417 a h).1
  have c109345 := (publicBatchWrapper86_copies3417 a h).2.1
  have c109346 := (publicBatchWrapper86_copies3417 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_12
  simp only [← c109341, ← c109342, ← c109343] at e_1790_12
  have e_1790_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_13
  simp only [← c109344, ← c109345, k2, ← c109346] at e_1790_13
  have hr := e_1790_13
  simp only [e_1790_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 63) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5185)) := by
  have c109347 := (publicBatchWrapper86_copies3417 a h).2.2.2.1
  have c109348 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.1
  have c109349 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.1
  have c109350 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.1
  have c109351 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.1
  have c109352 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_14
  simp only [← c109347, ← c109348, ← c109349] at e_1790_14
  have e_1790_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_15
  simp only [← c109350, ← c109351, k2, ← c109352] at e_1790_15
  have hr := e_1790_15
  simp only [e_1790_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 71) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5186)) := by
  have c109353 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.1
  have c109354 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109355 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109356 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109357 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109358 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_16
  simp only [← c109353, ← c109354, ← c109355] at e_1790_16
  have e_1790_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_17
  simp only [← c109356, ← c109357, k2, ← c109358] at e_1790_17
  have hr := e_1790_17
  simp only [e_1790_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1790 79) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5187)) := by
  have c109359 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109360 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109361 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109362 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109363 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109364 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1790_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_18
  simp only [← c109359, ← c109360, ← c109361] at e_1790_18
  have e_1790_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1790 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_19
  simp only [← c109362, ← c109363, k2, ← c109364] at e_1790_19
  have hr := e_1790_19
  simp only [e_1790_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 7) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5188)) := by
  have c109365 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109366 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109367 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109368 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109369 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109370 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_0
  simp only [← c109365, ← c109366, ← c109367] at e_1791_0
  have e_1791_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_1
  simp only [← c109368, ← c109369, k2, ← c109370] at e_1791_1
  have hr := e_1791_1
  simp only [e_1791_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 15) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5189)) := by
  have c109371 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109372 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109373 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109374 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109375 := (publicBatchWrapper86_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109376 := (publicBatchWrapper86_copies3418 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_2
  simp only [← c109371, ← c109372, ← c109373] at e_1791_2
  have e_1791_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_3
  simp only [← c109374, ← c109375, k2, ← c109376] at e_1791_3
  have hr := e_1791_3
  simp only [e_1791_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 23) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5190)) := by
  have c109377 := (publicBatchWrapper86_copies3418 a h).2.1
  have c109378 := (publicBatchWrapper86_copies3418 a h).2.2.1
  have c109379 := (publicBatchWrapper86_copies3418 a h).2.2.2.1
  have c109380 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.1
  have c109381 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.1
  have c109382 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_4
  simp only [← c109377, ← c109378, ← c109379] at e_1791_4
  have e_1791_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_5
  simp only [← c109380, ← c109381, k2, ← c109382] at e_1791_5
  have hr := e_1791_5
  simp only [e_1791_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 31) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5191)) := by
  have c109383 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.1
  have c109384 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.1
  have c109385 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.1
  have c109386 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109387 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109388 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_6
  simp only [← c109383, ← c109384, ← c109385] at e_1791_6
  have e_1791_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_7
  simp only [← c109386, ← c109387, k2, ← c109388] at e_1791_7
  have hr := e_1791_7
  simp only [e_1791_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 39) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5192)) := by
  have c109389 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109390 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109391 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109392 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109393 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109394 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_8
  simp only [← c109389, ← c109390, ← c109391] at e_1791_8
  have e_1791_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_9
  simp only [← c109392, ← c109393, k2, ← c109394] at e_1791_9
  have hr := e_1791_9
  simp only [e_1791_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 47) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5193)) := by
  have c109395 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109396 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109397 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109398 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109399 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109400 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_10
  simp only [← c109395, ← c109396, ← c109397] at e_1791_10
  have e_1791_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_11
  simp only [← c109398, ← c109399, k2, ← c109400] at e_1791_11
  have hr := e_1791_11
  simp only [e_1791_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 55) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5194)) := by
  have c109401 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109402 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109403 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109404 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109405 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109406 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_12
  simp only [← c109401, ← c109402, ← c109403] at e_1791_12
  have e_1791_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_13
  simp only [← c109404, ← c109405, k2, ← c109406] at e_1791_13
  have hr := e_1791_13
  simp only [e_1791_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 63) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5195)) := by
  have c109407 := (publicBatchWrapper86_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109408 := (publicBatchWrapper86_copies3419 a h).1
  have c109409 := (publicBatchWrapper86_copies3419 a h).2.1
  have c109410 := (publicBatchWrapper86_copies3419 a h).2.2.1
  have c109411 := (publicBatchWrapper86_copies3419 a h).2.2.2.1
  have c109412 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_14
  simp only [← c109407, ← c109408, ← c109409] at e_1791_14
  have e_1791_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_15
  simp only [← c109410, ← c109411, k2, ← c109412] at e_1791_15
  have hr := e_1791_15
  simp only [e_1791_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 71) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5196)) := by
  have c109413 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.1
  have c109414 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.1
  have c109415 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.1
  have c109416 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.1
  have c109417 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.1
  have c109418 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_16
  simp only [← c109413, ← c109414, ← c109415] at e_1791_16
  have e_1791_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_17
  simp only [← c109416, ← c109417, k2, ← c109418] at e_1791_17
  have hr := e_1791_17
  simp only [e_1791_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1791 79) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5197)) := by
  have c109419 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109420 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109421 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109422 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109423 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109424 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1791_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_18
  simp only [← c109419, ← c109420, ← c109421] at e_1791_18
  have e_1791_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1791 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_19
  simp only [← c109422, ← c109423, k2, ← c109424] at e_1791_19
  have hr := e_1791_19
  simp only [e_1791_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 7) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5198)) := by
  have c109425 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109426 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109427 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109428 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109429 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109430 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_0
  simp only [← c109425, ← c109426, ← c109427] at e_1792_0
  have e_1792_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_1
  simp only [← c109428, ← c109429, k2, ← c109430] at e_1792_1
  have hr := e_1792_1
  simp only [e_1792_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 15) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5199)) := by
  have c109431 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109432 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109433 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109434 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109435 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109436 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_2
  simp only [← c109431, ← c109432, ← c109433] at e_1792_2
  have e_1792_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_3
  simp only [← c109434, ← c109435, k2, ← c109436] at e_1792_3
  have hr := e_1792_3
  simp only [e_1792_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 23) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5200)) := by
  have c109437 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109438 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109439 := (publicBatchWrapper86_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109440 := (publicBatchWrapper86_copies3420 a h).1
  have c109441 := (publicBatchWrapper86_copies3420 a h).2.1
  have c109442 := (publicBatchWrapper86_copies3420 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_4
  simp only [← c109437, ← c109438, ← c109439] at e_1792_4
  have e_1792_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_5
  simp only [← c109440, ← c109441, k2, ← c109442] at e_1792_5
  have hr := e_1792_5
  simp only [e_1792_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 31) = bselect (a (.wire 27 7)) (a (.virt 200)) (a (.virt 5201)) := by
  have c109443 := (publicBatchWrapper86_copies3420 a h).2.2.2.1
  have c109444 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.1
  have c109445 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.1
  have c109446 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.1
  have c109447 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.1
  have c109448 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_6
  simp only [← c109443, ← c109444, ← c109445] at e_1792_6
  have e_1792_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_7
  simp only [← c109446, ← c109447, k2, ← c109448] at e_1792_7
  have hr := e_1792_7
  simp only [e_1792_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 39) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5370)) := by
  have c109449 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.1
  have c109450 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109451 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109452 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109453 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109454 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_8
  simp only [← c109449, ← c109450, ← c109451] at e_1792_8
  have e_1792_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_9
  simp only [← c109452, ← c109453, k2, ← c109454] at e_1792_9
  have hr := e_1792_9
  simp only [e_1792_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 47) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5371)) := by
  have c109455 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109456 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109457 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109458 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109459 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109460 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_10
  simp only [← c109455, ← c109456, ← c109457] at e_1792_10
  have e_1792_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_11
  simp only [← c109458, ← c109459, k2, ← c109460] at e_1792_11
  have hr := e_1792_11
  simp only [e_1792_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 55) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5372)) := by
  have c109461 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109462 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109463 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109464 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109465 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109466 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_12
  simp only [← c109461, ← c109462, ← c109463] at e_1792_12
  have e_1792_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_13
  simp only [← c109464, ← c109465, k2, ← c109466] at e_1792_13
  have hr := e_1792_13
  simp only [e_1792_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 63) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5373)) := by
  have c109467 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109468 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109469 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109470 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109471 := (publicBatchWrapper86_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109472 := (publicBatchWrapper86_copies3421 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_14
  simp only [← c109467, ← c109468, ← c109469] at e_1792_14
  have e_1792_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_15
  simp only [← c109470, ← c109471, k2, ← c109472] at e_1792_15
  have hr := e_1792_15
  simp only [e_1792_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 71) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5374)) := by
  have c109473 := (publicBatchWrapper86_copies3421 a h).2.1
  have c109474 := (publicBatchWrapper86_copies3421 a h).2.2.1
  have c109475 := (publicBatchWrapper86_copies3421 a h).2.2.2.1
  have c109476 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.1
  have c109477 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.1
  have c109478 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_16
  simp only [← c109473, ← c109474, ← c109475] at e_1792_16
  have e_1792_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_17
  simp only [← c109476, ← c109477, k2, ← c109478] at e_1792_17
  have hr := e_1792_17
  simp only [e_1792_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1792 79) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5375)) := by
  have c109479 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.1
  have c109480 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.1
  have c109481 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.1
  have c109482 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109483 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109484 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1792_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_18
  simp only [← c109479, ← c109480, ← c109481] at e_1792_18
  have e_1792_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1792 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_19
  simp only [← c109482, ← c109483, k2, ← c109484] at e_1792_19
  have hr := e_1792_19
  simp only [e_1792_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 7) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5376)) := by
  have c109485 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109486 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109487 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109488 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109489 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109490 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_0
  simp only [← c109485, ← c109486, ← c109487] at e_1793_0
  have e_1793_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_1
  simp only [← c109488, ← c109489, k2, ← c109490] at e_1793_1
  have hr := e_1793_1
  simp only [e_1793_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 15) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5377)) := by
  have c109491 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109492 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109493 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109494 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109495 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109496 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_2
  simp only [← c109491, ← c109492, ← c109493] at e_1793_2
  have e_1793_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_3
  simp only [← c109494, ← c109495, k2, ← c109496] at e_1793_3
  have hr := e_1793_3
  simp only [e_1793_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 23) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5378)) := by
  have c109497 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109498 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109499 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109500 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109501 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109502 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_4
  simp only [← c109497, ← c109498, ← c109499] at e_1793_4
  have e_1793_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_5
  simp only [← c109500, ← c109501, k2, ← c109502] at e_1793_5
  have hr := e_1793_5
  simp only [e_1793_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 31) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5379)) := by
  have c109503 := (publicBatchWrapper86_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109504 := (publicBatchWrapper86_copies3422 a h).1
  have c109505 := (publicBatchWrapper86_copies3422 a h).2.1
  have c109506 := (publicBatchWrapper86_copies3422 a h).2.2.1
  have c109507 := (publicBatchWrapper86_copies3422 a h).2.2.2.1
  have c109508 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_6
  simp only [← c109503, ← c109504, ← c109505] at e_1793_6
  have e_1793_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_7
  simp only [← c109506, ← c109507, k2, ← c109508] at e_1793_7
  have hr := e_1793_7
  simp only [e_1793_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 39) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5380)) := by
  have c109509 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.1
  have c109510 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.1
  have c109511 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.1
  have c109512 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.1
  have c109513 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.1
  have c109514 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_8
  simp only [← c109509, ← c109510, ← c109511] at e_1793_8
  have e_1793_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_9
  simp only [← c109512, ← c109513, k2, ← c109514] at e_1793_9
  have hr := e_1793_9
  simp only [e_1793_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 47) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5381)) := by
  have c109515 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109516 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109517 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109518 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109519 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109520 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_10
  simp only [← c109515, ← c109516, ← c109517] at e_1793_10
  have e_1793_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_11
  simp only [← c109518, ← c109519, k2, ← c109520] at e_1793_11
  have hr := e_1793_11
  simp only [e_1793_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 55) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5382)) := by
  have c109521 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109522 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109523 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109524 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109525 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109526 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_12
  simp only [← c109521, ← c109522, ← c109523] at e_1793_12
  have e_1793_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_13
  simp only [← c109524, ← c109525, k2, ← c109526] at e_1793_13
  have hr := e_1793_13
  simp only [e_1793_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 63) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5383)) := by
  have c109527 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109528 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109529 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109530 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109531 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109532 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_14
  simp only [← c109527, ← c109528, ← c109529] at e_1793_14
  have e_1793_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_15
  simp only [← c109530, ← c109531, k2, ← c109532] at e_1793_15
  have hr := e_1793_15
  simp only [e_1793_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 71) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5384)) := by
  have c109533 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109534 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109535 := (publicBatchWrapper86_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109536 := (publicBatchWrapper86_copies3423 a h).1
  have c109537 := (publicBatchWrapper86_copies3423 a h).2.1
  have c109538 := (publicBatchWrapper86_copies3423 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_16
  simp only [← c109533, ← c109534, ← c109535] at e_1793_16
  have e_1793_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_17
  simp only [← c109536, ← c109537, k2, ← c109538] at e_1793_17
  have hr := e_1793_17
  simp only [e_1793_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1793 79) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5385)) := by
  have c109539 := (publicBatchWrapper86_copies3423 a h).2.2.2.1
  have c109540 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.1
  have c109541 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.1
  have c109542 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.1
  have c109543 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.1
  have c109544 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1793_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_18
  simp only [← c109539, ← c109540, ← c109541] at e_1793_18
  have e_1793_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1793 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_19
  simp only [← c109542, ← c109543, k2, ← c109544] at e_1793_19
  have hr := e_1793_19
  simp only [e_1793_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 7) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5386)) := by
  have c109545 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.1
  have c109546 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109547 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109548 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109549 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109550 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_0
  simp only [← c109545, ← c109546, ← c109547] at e_1794_0
  have e_1794_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_1
  simp only [← c109548, ← c109549, k2, ← c109550] at e_1794_1
  have hr := e_1794_1
  simp only [e_1794_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 15) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5387)) := by
  have c109551 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109552 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109553 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109554 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109555 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109556 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_2
  simp only [← c109551, ← c109552, ← c109553] at e_1794_2
  have e_1794_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_3
  simp only [← c109554, ← c109555, k2, ← c109556] at e_1794_3
  have hr := e_1794_3
  simp only [e_1794_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 23) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5388)) := by
  have c109557 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109558 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109559 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109560 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109561 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109562 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_4
  simp only [← c109557, ← c109558, ← c109559] at e_1794_4
  have e_1794_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_5
  simp only [← c109560, ← c109561, k2, ← c109562] at e_1794_5
  have hr := e_1794_5
  simp only [e_1794_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 31) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5389)) := by
  have c109563 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109564 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109565 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109566 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109567 := (publicBatchWrapper86_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109568 := (publicBatchWrapper86_copies3424 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_6
  simp only [← c109563, ← c109564, ← c109565] at e_1794_6
  have e_1794_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_7
  simp only [← c109566, ← c109567, k2, ← c109568] at e_1794_7
  have hr := e_1794_7
  simp only [e_1794_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 39) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5390)) := by
  have c109569 := (publicBatchWrapper86_copies3424 a h).2.1
  have c109570 := (publicBatchWrapper86_copies3424 a h).2.2.1
  have c109571 := (publicBatchWrapper86_copies3424 a h).2.2.2.1
  have c109572 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.1
  have c109573 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.1
  have c109574 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_8
  simp only [← c109569, ← c109570, ← c109571] at e_1794_8
  have e_1794_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_9
  simp only [← c109572, ← c109573, k2, ← c109574] at e_1794_9
  have hr := e_1794_9
  simp only [e_1794_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 47) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5391)) := by
  have c109575 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.1
  have c109576 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.1
  have c109577 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.1
  have c109578 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109579 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109580 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_10
  simp only [← c109575, ← c109576, ← c109577] at e_1794_10
  have e_1794_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_11
  simp only [← c109578, ← c109579, k2, ← c109580] at e_1794_11
  have hr := e_1794_11
  simp only [e_1794_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 55) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5392)) := by
  have c109581 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109582 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109583 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109584 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109585 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109586 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_12
  simp only [← c109581, ← c109582, ← c109583] at e_1794_12
  have e_1794_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_13
  simp only [← c109584, ← c109585, k2, ← c109586] at e_1794_13
  have hr := e_1794_13
  simp only [e_1794_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 63) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5393)) := by
  have c109587 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109588 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109589 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109590 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109591 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109592 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_14
  simp only [← c109587, ← c109588, ← c109589] at e_1794_14
  have e_1794_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_15
  simp only [← c109590, ← c109591, k2, ← c109592] at e_1794_15
  have hr := e_1794_15
  simp only [e_1794_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 71) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5394)) := by
  have c109593 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109594 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109595 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109596 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109597 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109598 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_16
  simp only [← c109593, ← c109594, ← c109595] at e_1794_16
  have e_1794_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_17
  simp only [← c109596, ← c109597, k2, ← c109598] at e_1794_17
  have hr := e_1794_17
  simp only [e_1794_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1794 79) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5395)) := by
  have c109599 := (publicBatchWrapper86_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109600 := (publicBatchWrapper86_copies3425 a h).1
  have c109601 := (publicBatchWrapper86_copies3425 a h).2.1
  have c109602 := (publicBatchWrapper86_copies3425 a h).2.2.1
  have c109603 := (publicBatchWrapper86_copies3425 a h).2.2.2.1
  have c109604 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1794_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_18
  simp only [← c109599, ← c109600, ← c109601] at e_1794_18
  have e_1794_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1794 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_19
  simp only [← c109602, ← c109603, k2, ← c109604] at e_1794_19
  have hr := e_1794_19
  simp only [e_1794_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 7) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5396)) := by
  have c109605 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.1
  have c109606 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.1
  have c109607 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.1
  have c109608 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.1
  have c109609 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.1
  have c109610 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_0
  simp only [← c109605, ← c109606, ← c109607] at e_1795_0
  have e_1795_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_1
  simp only [← c109608, ← c109609, k2, ← c109610] at e_1795_1
  have hr := e_1795_1
  simp only [e_1795_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 15) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5397)) := by
  have c109611 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109612 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109613 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109614 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109615 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109616 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_2
  simp only [← c109611, ← c109612, ← c109613] at e_1795_2
  have e_1795_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_3
  simp only [← c109614, ← c109615, k2, ← c109616] at e_1795_3
  have hr := e_1795_3
  simp only [e_1795_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 23) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5398)) := by
  have c109617 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109618 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109619 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109620 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109621 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109622 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_4
  simp only [← c109617, ← c109618, ← c109619] at e_1795_4
  have e_1795_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_5
  simp only [← c109620, ← c109621, k2, ← c109622] at e_1795_5
  have hr := e_1795_5
  simp only [e_1795_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 31) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5399)) := by
  have c109623 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109624 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109625 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109626 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109627 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109628 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_6
  simp only [← c109623, ← c109624, ← c109625] at e_1795_6
  have e_1795_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_7
  simp only [← c109626, ← c109627, k2, ← c109628] at e_1795_7
  have hr := e_1795_7
  simp only [e_1795_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 39) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5400)) := by
  have c109629 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109630 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109631 := (publicBatchWrapper86_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109632 := (publicBatchWrapper86_copies3426 a h).1
  have c109633 := (publicBatchWrapper86_copies3426 a h).2.1
  have c109634 := (publicBatchWrapper86_copies3426 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_8
  simp only [← c109629, ← c109630, ← c109631] at e_1795_8
  have e_1795_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_9
  simp only [← c109632, ← c109633, k2, ← c109634] at e_1795_9
  have hr := e_1795_9
  simp only [e_1795_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 47) = bselect (a (.wire 27 51)) (a (.virt 200)) (a (.virt 5401)) := by
  have c109635 := (publicBatchWrapper86_copies3426 a h).2.2.2.1
  have c109636 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.1
  have c109637 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.1
  have c109638 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.1
  have c109639 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.1
  have c109640 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_10
  simp only [← c109635, ← c109636, ← c109637] at e_1795_10
  have e_1795_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_11
  simp only [← c109638, ← c109639, k2, ← c109640] at e_1795_11
  have hr := e_1795_11
  simp only [e_1795_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 55) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5570)) := by
  have c109641 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.1
  have c109642 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109643 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109644 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109645 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109646 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_12
  simp only [← c109641, ← c109642, ← c109643] at e_1795_12
  have e_1795_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_13
  simp only [← c109644, ← c109645, k2, ← c109646] at e_1795_13
  have hr := e_1795_13
  simp only [e_1795_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 63) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5571)) := by
  have c109647 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109648 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109649 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109650 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109651 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109652 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_14
  simp only [← c109647, ← c109648, ← c109649] at e_1795_14
  have e_1795_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_15
  simp only [← c109650, ← c109651, k2, ← c109652] at e_1795_15
  have hr := e_1795_15
  simp only [e_1795_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 71) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5572)) := by
  have c109653 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109654 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109655 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109656 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109657 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109658 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_16
  simp only [← c109653, ← c109654, ← c109655] at e_1795_16
  have e_1795_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_17
  simp only [← c109656, ← c109657, k2, ← c109658] at e_1795_17
  have hr := e_1795_17
  simp only [e_1795_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1795 79) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5573)) := by
  have c109659 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109660 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109661 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109662 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109663 := (publicBatchWrapper86_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109664 := (publicBatchWrapper86_copies3427 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1795_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_18
  simp only [← c109659, ← c109660, ← c109661] at e_1795_18
  have e_1795_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1795 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_19
  simp only [← c109662, ← c109663, k2, ← c109664] at e_1795_19
  have hr := e_1795_19
  simp only [e_1795_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 7) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5574)) := by
  have c109665 := (publicBatchWrapper86_copies3427 a h).2.1
  have c109666 := (publicBatchWrapper86_copies3427 a h).2.2.1
  have c109667 := (publicBatchWrapper86_copies3427 a h).2.2.2.1
  have c109668 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.1
  have c109669 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.1
  have c109670 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_0
  simp only [← c109665, ← c109666, ← c109667] at e_1796_0
  have e_1796_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_1
  simp only [← c109668, ← c109669, k2, ← c109670] at e_1796_1
  have hr := e_1796_1
  simp only [e_1796_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 15) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5575)) := by
  have c109671 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.1
  have c109672 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.1
  have c109673 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.1
  have c109674 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109675 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109676 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_2
  simp only [← c109671, ← c109672, ← c109673] at e_1796_2
  have e_1796_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_3
  simp only [← c109674, ← c109675, k2, ← c109676] at e_1796_3
  have hr := e_1796_3
  simp only [e_1796_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 23) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5576)) := by
  have c109677 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109678 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109679 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109680 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109681 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109682 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_4
  simp only [← c109677, ← c109678, ← c109679] at e_1796_4
  have e_1796_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_5
  simp only [← c109680, ← c109681, k2, ← c109682] at e_1796_5
  have hr := e_1796_5
  simp only [e_1796_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 31) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5577)) := by
  have c109683 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109684 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109685 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109686 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109687 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109688 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_6
  simp only [← c109683, ← c109684, ← c109685] at e_1796_6
  have e_1796_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_7
  simp only [← c109686, ← c109687, k2, ← c109688] at e_1796_7
  have hr := e_1796_7
  simp only [e_1796_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 39) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5578)) := by
  have c109689 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109690 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109691 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109692 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109693 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109694 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_8
  simp only [← c109689, ← c109690, ← c109691] at e_1796_8
  have e_1796_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_9
  simp only [← c109692, ← c109693, k2, ← c109694] at e_1796_9
  have hr := e_1796_9
  simp only [e_1796_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 47) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5579)) := by
  have c109695 := (publicBatchWrapper86_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109696 := (publicBatchWrapper86_copies3428 a h).1
  have c109697 := (publicBatchWrapper86_copies3428 a h).2.1
  have c109698 := (publicBatchWrapper86_copies3428 a h).2.2.1
  have c109699 := (publicBatchWrapper86_copies3428 a h).2.2.2.1
  have c109700 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_10
  simp only [← c109695, ← c109696, ← c109697] at e_1796_10
  have e_1796_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_11
  simp only [← c109698, ← c109699, k2, ← c109700] at e_1796_11
  have hr := e_1796_11
  simp only [e_1796_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 55) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5580)) := by
  have c109701 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.1
  have c109702 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.1
  have c109703 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.1
  have c109704 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.1
  have c109705 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.1
  have c109706 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_12
  simp only [← c109701, ← c109702, ← c109703] at e_1796_12
  have e_1796_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_13
  simp only [← c109704, ← c109705, k2, ← c109706] at e_1796_13
  have hr := e_1796_13
  simp only [e_1796_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 63) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5581)) := by
  have c109707 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109708 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109709 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109710 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109711 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109712 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_14
  simp only [← c109707, ← c109708, ← c109709] at e_1796_14
  have e_1796_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_15
  simp only [← c109710, ← c109711, k2, ← c109712] at e_1796_15
  have hr := e_1796_15
  simp only [e_1796_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 71) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5582)) := by
  have c109713 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109714 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109715 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109716 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109717 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109718 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_16
  simp only [← c109713, ← c109714, ← c109715] at e_1796_16
  have e_1796_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_17
  simp only [← c109716, ← c109717, k2, ← c109718] at e_1796_17
  have hr := e_1796_17
  simp only [e_1796_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1796 79) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5583)) := by
  have c109719 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109720 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109721 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109722 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109723 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109724 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1796_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_18
  simp only [← c109719, ← c109720, ← c109721] at e_1796_18
  have e_1796_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1796 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_19
  simp only [← c109722, ← c109723, k2, ← c109724] at e_1796_19
  have hr := e_1796_19
  simp only [e_1796_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 7) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5584)) := by
  have c109725 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109726 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109727 := (publicBatchWrapper86_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109728 := (publicBatchWrapper86_copies3429 a h).1
  have c109729 := (publicBatchWrapper86_copies3429 a h).2.1
  have c109730 := (publicBatchWrapper86_copies3429 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_0
  simp only [← c109725, ← c109726, ← c109727] at e_1797_0
  have e_1797_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_1
  simp only [← c109728, ← c109729, k2, ← c109730] at e_1797_1
  have hr := e_1797_1
  simp only [e_1797_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 15) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5585)) := by
  have c109731 := (publicBatchWrapper86_copies3429 a h).2.2.2.1
  have c109732 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.1
  have c109733 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.1
  have c109734 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.1
  have c109735 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.1
  have c109736 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_2
  simp only [← c109731, ← c109732, ← c109733] at e_1797_2
  have e_1797_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_3
  simp only [← c109734, ← c109735, k2, ← c109736] at e_1797_3
  have hr := e_1797_3
  simp only [e_1797_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 23) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5586)) := by
  have c109737 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.1
  have c109738 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109739 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109740 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109741 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109742 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_4
  simp only [← c109737, ← c109738, ← c109739] at e_1797_4
  have e_1797_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_5
  simp only [← c109740, ← c109741, k2, ← c109742] at e_1797_5
  have hr := e_1797_5
  simp only [e_1797_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 31) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5587)) := by
  have c109743 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109744 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109745 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109746 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109747 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109748 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_6
  simp only [← c109743, ← c109744, ← c109745] at e_1797_6
  have e_1797_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_7
  simp only [← c109746, ← c109747, k2, ← c109748] at e_1797_7
  have hr := e_1797_7
  simp only [e_1797_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 39) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5588)) := by
  have c109749 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109750 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109751 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109752 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109753 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109754 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_8
  simp only [← c109749, ← c109750, ← c109751] at e_1797_8
  have e_1797_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_9
  simp only [← c109752, ← c109753, k2, ← c109754] at e_1797_9
  have hr := e_1797_9
  simp only [e_1797_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 47) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5589)) := by
  have c109755 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109756 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109757 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109758 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109759 := (publicBatchWrapper86_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109760 := (publicBatchWrapper86_copies3430 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_10
  simp only [← c109755, ← c109756, ← c109757] at e_1797_10
  have e_1797_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_11
  simp only [← c109758, ← c109759, k2, ← c109760] at e_1797_11
  have hr := e_1797_11
  simp only [e_1797_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 55) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5590)) := by
  have c109761 := (publicBatchWrapper86_copies3430 a h).2.1
  have c109762 := (publicBatchWrapper86_copies3430 a h).2.2.1
  have c109763 := (publicBatchWrapper86_copies3430 a h).2.2.2.1
  have c109764 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.1
  have c109765 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.1
  have c109766 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_12
  simp only [← c109761, ← c109762, ← c109763] at e_1797_12
  have e_1797_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_13
  simp only [← c109764, ← c109765, k2, ← c109766] at e_1797_13
  have hr := e_1797_13
  simp only [e_1797_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 63) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5591)) := by
  have c109767 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.1
  have c109768 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.1
  have c109769 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.1
  have c109770 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109771 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109772 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_14
  simp only [← c109767, ← c109768, ← c109769] at e_1797_14
  have e_1797_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_15
  simp only [← c109770, ← c109771, k2, ← c109772] at e_1797_15
  have hr := e_1797_15
  simp only [e_1797_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 71) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5592)) := by
  have c109773 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109774 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109775 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109776 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109777 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109778 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_16
  simp only [← c109773, ← c109774, ← c109775] at e_1797_16
  have e_1797_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_17
  simp only [← c109776, ← c109777, k2, ← c109778] at e_1797_17
  have hr := e_1797_17
  simp only [e_1797_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1797 79) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5593)) := by
  have c109779 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109780 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109781 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109782 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109783 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109784 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1797_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_18
  simp only [← c109779, ← c109780, ← c109781] at e_1797_18
  have e_1797_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1797 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_19
  simp only [← c109782, ← c109783, k2, ← c109784] at e_1797_19
  have hr := e_1797_19
  simp only [e_1797_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 7) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5594)) := by
  have c109785 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109786 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109787 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109788 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109789 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109790 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_0
  simp only [← c109785, ← c109786, ← c109787] at e_1798_0
  have e_1798_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_1
  simp only [← c109788, ← c109789, k2, ← c109790] at e_1798_1
  have hr := e_1798_1
  simp only [e_1798_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 15) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5595)) := by
  have c109791 := (publicBatchWrapper86_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109792 := (publicBatchWrapper86_copies3431 a h).1
  have c109793 := (publicBatchWrapper86_copies3431 a h).2.1
  have c109794 := (publicBatchWrapper86_copies3431 a h).2.2.1
  have c109795 := (publicBatchWrapper86_copies3431 a h).2.2.2.1
  have c109796 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_2
  simp only [← c109791, ← c109792, ← c109793] at e_1798_2
  have e_1798_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_3
  simp only [← c109794, ← c109795, k2, ← c109796] at e_1798_3
  have hr := e_1798_3
  simp only [e_1798_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 23) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5596)) := by
  have c109797 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.1
  have c109798 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.1
  have c109799 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.1
  have c109800 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.1
  have c109801 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.1
  have c109802 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_4
  simp only [← c109797, ← c109798, ← c109799] at e_1798_4
  have e_1798_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_5
  simp only [← c109800, ← c109801, k2, ← c109802] at e_1798_5
  have hr := e_1798_5
  simp only [e_1798_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 31) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5597)) := by
  have c109803 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109804 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109805 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109806 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109807 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109808 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_6
  simp only [← c109803, ← c109804, ← c109805] at e_1798_6
  have e_1798_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_7
  simp only [← c109806, ← c109807, k2, ← c109808] at e_1798_7
  have hr := e_1798_7
  simp only [e_1798_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 39) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5598)) := by
  have c109809 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109810 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109811 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109812 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109813 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109814 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_8
  simp only [← c109809, ← c109810, ← c109811] at e_1798_8
  have e_1798_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_9
  simp only [← c109812, ← c109813, k2, ← c109814] at e_1798_9
  have hr := e_1798_9
  simp only [e_1798_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 47) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5599)) := by
  have c109815 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109816 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109817 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109818 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109819 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109820 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_10
  simp only [← c109815, ← c109816, ← c109817] at e_1798_10
  have e_1798_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_11
  simp only [← c109818, ← c109819, k2, ← c109820] at e_1798_11
  have hr := e_1798_11
  simp only [e_1798_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 55) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5600)) := by
  have c109821 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109822 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109823 := (publicBatchWrapper86_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109824 := (publicBatchWrapper86_copies3432 a h).1
  have c109825 := (publicBatchWrapper86_copies3432 a h).2.1
  have c109826 := (publicBatchWrapper86_copies3432 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_12
  simp only [← c109821, ← c109822, ← c109823] at e_1798_12
  have e_1798_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_13
  simp only [← c109824, ← c109825, k2, ← c109826] at e_1798_13
  have hr := e_1798_13
  simp only [e_1798_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 63) = bselect (a (.wire 29 15)) (a (.virt 200)) (a (.virt 5601)) := by
  have c109827 := (publicBatchWrapper86_copies3432 a h).2.2.2.1
  have c109828 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.1
  have c109829 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.1
  have c109830 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.1
  have c109831 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.1
  have c109832 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_14
  simp only [← c109827, ← c109828, ← c109829] at e_1798_14
  have e_1798_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_15
  simp only [← c109830, ← c109831, k2, ← c109832] at e_1798_15
  have hr := e_1798_15
  simp only [e_1798_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 71) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5770)) := by
  have c109833 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.1
  have c109834 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109835 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109836 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109837 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109838 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_16
  simp only [← c109833, ← c109834, ← c109835] at e_1798_16
  have e_1798_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_17
  simp only [← c109836, ← c109837, k2, ← c109838] at e_1798_17
  have hr := e_1798_17
  simp only [e_1798_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1798 79) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5771)) := by
  have c109839 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109840 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109841 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109842 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109843 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109844 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1798_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_18
  simp only [← c109839, ← c109840, ← c109841] at e_1798_18
  have e_1798_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1798 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_19
  simp only [← c109842, ← c109843, k2, ← c109844] at e_1798_19
  have hr := e_1798_19
  simp only [e_1798_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 7) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5772)) := by
  have c109845 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109846 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109847 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109848 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109849 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109850 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_0
  simp only [← c109845, ← c109846, ← c109847] at e_1799_0
  have e_1799_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_1
  simp only [← c109848, ← c109849, k2, ← c109850] at e_1799_1
  have hr := e_1799_1
  simp only [e_1799_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 15) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5773)) := by
  have c109851 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109852 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109853 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109854 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109855 := (publicBatchWrapper86_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109856 := (publicBatchWrapper86_copies3433 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_2
  simp only [← c109851, ← c109852, ← c109853] at e_1799_2
  have e_1799_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_3
  simp only [← c109854, ← c109855, k2, ← c109856] at e_1799_3
  have hr := e_1799_3
  simp only [e_1799_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 23) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5774)) := by
  have c109857 := (publicBatchWrapper86_copies3433 a h).2.1
  have c109858 := (publicBatchWrapper86_copies3433 a h).2.2.1
  have c109859 := (publicBatchWrapper86_copies3433 a h).2.2.2.1
  have c109860 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.1
  have c109861 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.1
  have c109862 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_4
  simp only [← c109857, ← c109858, ← c109859] at e_1799_4
  have e_1799_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_5
  simp only [← c109860, ← c109861, k2, ← c109862] at e_1799_5
  have hr := e_1799_5
  simp only [e_1799_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 31) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5775)) := by
  have c109863 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.1
  have c109864 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.1
  have c109865 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.1
  have c109866 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109867 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109868 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_6
  simp only [← c109863, ← c109864, ← c109865] at e_1799_6
  have e_1799_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_7
  simp only [← c109866, ← c109867, k2, ← c109868] at e_1799_7
  have hr := e_1799_7
  simp only [e_1799_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 39) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5776)) := by
  have c109869 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109870 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109871 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109872 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109873 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109874 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_8
  simp only [← c109869, ← c109870, ← c109871] at e_1799_8
  have e_1799_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_9
  simp only [← c109872, ← c109873, k2, ← c109874] at e_1799_9
  have hr := e_1799_9
  simp only [e_1799_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 47) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5777)) := by
  have c109875 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109876 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109877 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109878 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109879 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109880 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_10
  simp only [← c109875, ← c109876, ← c109877] at e_1799_10
  have e_1799_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_11
  simp only [← c109878, ← c109879, k2, ← c109880] at e_1799_11
  have hr := e_1799_11
  simp only [e_1799_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 55) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5778)) := by
  have c109881 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109882 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109883 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109884 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109885 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109886 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_12
  simp only [← c109881, ← c109882, ← c109883] at e_1799_12
  have e_1799_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_13
  simp only [← c109884, ← c109885, k2, ← c109886] at e_1799_13
  have hr := e_1799_13
  simp only [e_1799_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 63) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5779)) := by
  have c109887 := (publicBatchWrapper86_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109888 := (publicBatchWrapper86_copies3434 a h).1
  have c109889 := (publicBatchWrapper86_copies3434 a h).2.1
  have c109890 := (publicBatchWrapper86_copies3434 a h).2.2.1
  have c109891 := (publicBatchWrapper86_copies3434 a h).2.2.2.1
  have c109892 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_14
  simp only [← c109887, ← c109888, ← c109889] at e_1799_14
  have e_1799_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_15
  simp only [← c109890, ← c109891, k2, ← c109892] at e_1799_15
  have hr := e_1799_15
  simp only [e_1799_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 71) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5780)) := by
  have c109893 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.1
  have c109894 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.1
  have c109895 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.1
  have c109896 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.1
  have c109897 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.1
  have c109898 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_16
  simp only [← c109893, ← c109894, ← c109895] at e_1799_16
  have e_1799_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_17
  simp only [← c109896, ← c109897, k2, ← c109898] at e_1799_17
  have hr := e_1799_17
  simp only [e_1799_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1799 79) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5781)) := by
  have c109899 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109900 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109901 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109902 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109903 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109904 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1799_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_18
  simp only [← c109899, ← c109900, ← c109901] at e_1799_18
  have e_1799_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1799 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_19
  simp only [← c109902, ← c109903, k2, ← c109904] at e_1799_19
  have hr := e_1799_19
  simp only [e_1799_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 7) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5782)) := by
  have c109905 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109906 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109907 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109908 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109909 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109910 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_0
  simp only [← c109905, ← c109906, ← c109907] at e_1800_0
  have e_1800_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_1
  simp only [← c109908, ← c109909, k2, ← c109910] at e_1800_1
  have hr := e_1800_1
  simp only [e_1800_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 15) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5783)) := by
  have c109911 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109912 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109913 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109914 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109915 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109916 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_2
  simp only [← c109911, ← c109912, ← c109913] at e_1800_2
  have e_1800_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_3
  simp only [← c109914, ← c109915, k2, ← c109916] at e_1800_3
  have hr := e_1800_3
  simp only [e_1800_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 23) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5784)) := by
  have c109917 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109918 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109919 := (publicBatchWrapper86_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109920 := (publicBatchWrapper86_copies3435 a h).1
  have c109921 := (publicBatchWrapper86_copies3435 a h).2.1
  have c109922 := (publicBatchWrapper86_copies3435 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_4
  simp only [← c109917, ← c109918, ← c109919] at e_1800_4
  have e_1800_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_5
  simp only [← c109920, ← c109921, k2, ← c109922] at e_1800_5
  have hr := e_1800_5
  simp only [e_1800_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 31) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5785)) := by
  have c109923 := (publicBatchWrapper86_copies3435 a h).2.2.2.1
  have c109924 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.1
  have c109925 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.1
  have c109926 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.1
  have c109927 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.1
  have c109928 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_6
  simp only [← c109923, ← c109924, ← c109925] at e_1800_6
  have e_1800_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_7
  simp only [← c109926, ← c109927, k2, ← c109928] at e_1800_7
  have hr := e_1800_7
  simp only [e_1800_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 39) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5786)) := by
  have c109929 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.1
  have c109930 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109931 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109932 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109933 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109934 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_8
  simp only [← c109929, ← c109930, ← c109931] at e_1800_8
  have e_1800_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_9
  simp only [← c109932, ← c109933, k2, ← c109934] at e_1800_9
  have hr := e_1800_9
  simp only [e_1800_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 47) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5787)) := by
  have c109935 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109936 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109937 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109938 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109939 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109940 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_10
  simp only [← c109935, ← c109936, ← c109937] at e_1800_10
  have e_1800_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_11
  simp only [← c109938, ← c109939, k2, ← c109940] at e_1800_11
  have hr := e_1800_11
  simp only [e_1800_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 55) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5788)) := by
  have c109941 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109942 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109943 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109944 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109945 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109946 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_12
  simp only [← c109941, ← c109942, ← c109943] at e_1800_12
  have e_1800_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_13
  simp only [← c109944, ← c109945, k2, ← c109946] at e_1800_13
  have hr := e_1800_13
  simp only [e_1800_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 63) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5789)) := by
  have c109947 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109948 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109949 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109950 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109951 := (publicBatchWrapper86_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109952 := (publicBatchWrapper86_copies3436 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_14
  simp only [← c109947, ← c109948, ← c109949] at e_1800_14
  have e_1800_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_15
  simp only [← c109950, ← c109951, k2, ← c109952] at e_1800_15
  have hr := e_1800_15
  simp only [e_1800_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 71) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5790)) := by
  have c109953 := (publicBatchWrapper86_copies3436 a h).2.1
  have c109954 := (publicBatchWrapper86_copies3436 a h).2.2.1
  have c109955 := (publicBatchWrapper86_copies3436 a h).2.2.2.1
  have c109956 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.1
  have c109957 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.1
  have c109958 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_16
  simp only [← c109953, ← c109954, ← c109955] at e_1800_16
  have e_1800_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_17
  simp only [← c109956, ← c109957, k2, ← c109958] at e_1800_17
  have hr := e_1800_17
  simp only [e_1800_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1800 79) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5791)) := by
  have c109959 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.1
  have c109960 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.1
  have c109961 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.1
  have c109962 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109963 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109964 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1800_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_18
  simp only [← c109959, ← c109960, ← c109961] at e_1800_18
  have e_1800_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1800 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_19
  simp only [← c109962, ← c109963, k2, ← c109964] at e_1800_19
  have hr := e_1800_19
  simp only [e_1800_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 7) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5792)) := by
  have c109965 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109966 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109967 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109968 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109969 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109970 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_0
  simp only [← c109965, ← c109966, ← c109967] at e_1801_0
  have e_1801_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_1
  simp only [← c109968, ← c109969, k2, ← c109970] at e_1801_1
  have hr := e_1801_1
  simp only [e_1801_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 15) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5793)) := by
  have c109971 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109972 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109973 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109974 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109975 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109976 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_2
  simp only [← c109971, ← c109972, ← c109973] at e_1801_2
  have e_1801_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_3
  simp only [← c109974, ← c109975, k2, ← c109976] at e_1801_3
  have hr := e_1801_3
  simp only [e_1801_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 23) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5794)) := by
  have c109977 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109978 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109979 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109980 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109981 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109982 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_4
  simp only [← c109977, ← c109978, ← c109979] at e_1801_4
  have e_1801_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_5
  simp only [← c109980, ← c109981, k2, ← c109982] at e_1801_5
  have hr := e_1801_5
  simp only [e_1801_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 31) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5795)) := by
  have c109983 := (publicBatchWrapper86_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109984 := (publicBatchWrapper86_copies3437 a h).1
  have c109985 := (publicBatchWrapper86_copies3437 a h).2.1
  have c109986 := (publicBatchWrapper86_copies3437 a h).2.2.1
  have c109987 := (publicBatchWrapper86_copies3437 a h).2.2.2.1
  have c109988 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_6
  simp only [← c109983, ← c109984, ← c109985] at e_1801_6
  have e_1801_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_7
  simp only [← c109986, ← c109987, k2, ← c109988] at e_1801_7
  have hr := e_1801_7
  simp only [e_1801_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 39) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5796)) := by
  have c109989 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.1
  have c109990 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.1
  have c109991 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.1
  have c109992 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.1
  have c109993 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.1
  have c109994 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_8
  simp only [← c109989, ← c109990, ← c109991] at e_1801_8
  have e_1801_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_9
  simp only [← c109992, ← c109993, k2, ← c109994] at e_1801_9
  have hr := e_1801_9
  simp only [e_1801_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 47) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5797)) := by
  have c109995 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109996 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109997 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109998 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109999 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110000 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_10
  simp only [← c109995, ← c109996, ← c109997] at e_1801_10
  have e_1801_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_11
  simp only [← c109998, ← c109999, k2, ← c110000] at e_1801_11
  have hr := e_1801_11
  simp only [e_1801_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 55) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5798)) := by
  have c110001 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110002 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110003 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110004 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110005 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110006 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_12
  simp only [← c110001, ← c110002, ← c110003] at e_1801_12
  have e_1801_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_13
  simp only [← c110004, ← c110005, k2, ← c110006] at e_1801_13
  have hr := e_1801_13
  simp only [e_1801_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 63) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5799)) := by
  have c110007 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110008 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110009 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110010 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110011 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110012 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_14
  simp only [← c110007, ← c110008, ← c110009] at e_1801_14
  have e_1801_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_15
  simp only [← c110010, ← c110011, k2, ← c110012] at e_1801_15
  have hr := e_1801_15
  simp only [e_1801_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 71) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5800)) := by
  have c110013 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110014 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110015 := (publicBatchWrapper86_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110016 := (publicBatchWrapper86_copies3438 a h).1
  have c110017 := (publicBatchWrapper86_copies3438 a h).2.1
  have c110018 := (publicBatchWrapper86_copies3438 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_16
  simp only [← c110013, ← c110014, ← c110015] at e_1801_16
  have e_1801_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_17
  simp only [← c110016, ← c110017, k2, ← c110018] at e_1801_17
  have hr := e_1801_17
  simp only [e_1801_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1801 79) = bselect (a (.wire 29 59)) (a (.virt 200)) (a (.virt 5801)) := by
  have c110019 := (publicBatchWrapper86_copies3438 a h).2.2.2.1
  have c110020 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.1
  have c110021 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.1
  have c110022 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.1
  have c110023 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.1
  have c110024 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1801_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_18
  simp only [← c110019, ← c110020, ← c110021] at e_1801_18
  have e_1801_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1801 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_19
  simp only [← c110022, ← c110023, k2, ← c110024] at e_1801_19
  have hr := e_1801_19
  simp only [e_1801_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 7) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5970)) := by
  have c110025 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.1
  have c110026 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110027 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110028 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110029 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110030 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_0
  simp only [← c110025, ← c110026, ← c110027] at e_1802_0
  have e_1802_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_1
  simp only [← c110028, ← c110029, k2, ← c110030] at e_1802_1
  have hr := e_1802_1
  simp only [e_1802_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 15) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5971)) := by
  have c110031 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110032 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110033 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110034 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110035 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110036 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_2
  simp only [← c110031, ← c110032, ← c110033] at e_1802_2
  have e_1802_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_3
  simp only [← c110034, ← c110035, k2, ← c110036] at e_1802_3
  have hr := e_1802_3
  simp only [e_1802_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 23) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5972)) := by
  have c110037 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110038 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110039 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110040 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110041 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110042 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_4
  simp only [← c110037, ← c110038, ← c110039] at e_1802_4
  have e_1802_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_5
  simp only [← c110040, ← c110041, k2, ← c110042] at e_1802_5
  have hr := e_1802_5
  simp only [e_1802_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 31) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5973)) := by
  have c110043 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110044 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110045 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110046 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110047 := (publicBatchWrapper86_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110048 := (publicBatchWrapper86_copies3439 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_6
  simp only [← c110043, ← c110044, ← c110045] at e_1802_6
  have e_1802_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_7
  simp only [← c110046, ← c110047, k2, ← c110048] at e_1802_7
  have hr := e_1802_7
  simp only [e_1802_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 39) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5974)) := by
  have c110049 := (publicBatchWrapper86_copies3439 a h).2.1
  have c110050 := (publicBatchWrapper86_copies3439 a h).2.2.1
  have c110051 := (publicBatchWrapper86_copies3439 a h).2.2.2.1
  have c110052 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.1
  have c110053 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.1
  have c110054 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_8
  simp only [← c110049, ← c110050, ← c110051] at e_1802_8
  have e_1802_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_9
  simp only [← c110052, ← c110053, k2, ← c110054] at e_1802_9
  have hr := e_1802_9
  simp only [e_1802_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 47) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5975)) := by
  have c110055 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.1
  have c110056 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.1
  have c110057 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.1
  have c110058 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110059 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110060 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_10
  simp only [← c110055, ← c110056, ← c110057] at e_1802_10
  have e_1802_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_11
  simp only [← c110058, ← c110059, k2, ← c110060] at e_1802_11
  have hr := e_1802_11
  simp only [e_1802_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 55) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5976)) := by
  have c110061 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110062 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110063 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110064 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110065 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110066 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_12
  simp only [← c110061, ← c110062, ← c110063] at e_1802_12
  have e_1802_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_13
  simp only [← c110064, ← c110065, k2, ← c110066] at e_1802_13
  have hr := e_1802_13
  simp only [e_1802_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 63) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5977)) := by
  have c110067 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110068 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110069 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110070 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110071 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110072 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_14
  simp only [← c110067, ← c110068, ← c110069] at e_1802_14
  have e_1802_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_15
  simp only [← c110070, ← c110071, k2, ← c110072] at e_1802_15
  have hr := e_1802_15
  simp only [e_1802_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 71) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5978)) := by
  have c110073 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110074 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110075 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110076 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110077 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110078 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_16
  simp only [← c110073, ← c110074, ← c110075] at e_1802_16
  have e_1802_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_17
  simp only [← c110076, ← c110077, k2, ← c110078] at e_1802_17
  have hr := e_1802_17
  simp only [e_1802_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1802 79) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5979)) := by
  have c110079 := (publicBatchWrapper86_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110080 := (publicBatchWrapper86_copies3440 a h).1
  have c110081 := (publicBatchWrapper86_copies3440 a h).2.1
  have c110082 := (publicBatchWrapper86_copies3440 a h).2.2.1
  have c110083 := (publicBatchWrapper86_copies3440 a h).2.2.2.1
  have c110084 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1802_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_18
  simp only [← c110079, ← c110080, ← c110081] at e_1802_18
  have e_1802_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1802 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_19
  simp only [← c110082, ← c110083, k2, ← c110084] at e_1802_19
  have hr := e_1802_19
  simp only [e_1802_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 7) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5980)) := by
  have c110085 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.1
  have c110086 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.1
  have c110087 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.1
  have c110088 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.1
  have c110089 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.1
  have c110090 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_0
  simp only [← c110085, ← c110086, ← c110087] at e_1803_0
  have e_1803_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_1
  simp only [← c110088, ← c110089, k2, ← c110090] at e_1803_1
  have hr := e_1803_1
  simp only [e_1803_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 15) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5981)) := by
  have c110091 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110092 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110093 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110094 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110095 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110096 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_2
  simp only [← c110091, ← c110092, ← c110093] at e_1803_2
  have e_1803_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_3
  simp only [← c110094, ← c110095, k2, ← c110096] at e_1803_3
  have hr := e_1803_3
  simp only [e_1803_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 23) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5982)) := by
  have c110097 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110098 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110099 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110100 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110101 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110102 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_4
  simp only [← c110097, ← c110098, ← c110099] at e_1803_4
  have e_1803_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_5
  simp only [← c110100, ← c110101, k2, ← c110102] at e_1803_5
  have hr := e_1803_5
  simp only [e_1803_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 31) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5983)) := by
  have c110103 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110104 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110105 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110106 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110107 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110108 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_6
  simp only [← c110103, ← c110104, ← c110105] at e_1803_6
  have e_1803_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_7
  simp only [← c110106, ← c110107, k2, ← c110108] at e_1803_7
  have hr := e_1803_7
  simp only [e_1803_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 39) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5984)) := by
  have c110109 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110110 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110111 := (publicBatchWrapper86_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110112 := (publicBatchWrapper86_copies3441 a h).1
  have c110113 := (publicBatchWrapper86_copies3441 a h).2.1
  have c110114 := (publicBatchWrapper86_copies3441 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_8
  simp only [← c110109, ← c110110, ← c110111] at e_1803_8
  have e_1803_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_9
  simp only [← c110112, ← c110113, k2, ← c110114] at e_1803_9
  have hr := e_1803_9
  simp only [e_1803_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 47) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5985)) := by
  have c110115 := (publicBatchWrapper86_copies3441 a h).2.2.2.1
  have c110116 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.1
  have c110117 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.1
  have c110118 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.1
  have c110119 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.1
  have c110120 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_10
  simp only [← c110115, ← c110116, ← c110117] at e_1803_10
  have e_1803_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_11
  simp only [← c110118, ← c110119, k2, ← c110120] at e_1803_11
  have hr := e_1803_11
  simp only [e_1803_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 55) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5986)) := by
  have c110121 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.1
  have c110122 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110123 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110124 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110125 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110126 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_12
  simp only [← c110121, ← c110122, ← c110123] at e_1803_12
  have e_1803_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_13
  simp only [← c110124, ← c110125, k2, ← c110126] at e_1803_13
  have hr := e_1803_13
  simp only [e_1803_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 63) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5987)) := by
  have c110127 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110128 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110129 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110130 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110131 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110132 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_14
  simp only [← c110127, ← c110128, ← c110129] at e_1803_14
  have e_1803_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_15
  simp only [← c110130, ← c110131, k2, ← c110132] at e_1803_15
  have hr := e_1803_15
  simp only [e_1803_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 71) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5988)) := by
  have c110133 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110134 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110135 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110136 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110137 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110138 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_16
  simp only [← c110133, ← c110134, ← c110135] at e_1803_16
  have e_1803_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_17
  simp only [← c110136, ← c110137, k2, ← c110138] at e_1803_17
  have hr := e_1803_17
  simp only [e_1803_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1803 79) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5989)) := by
  have c110139 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110140 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110141 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110142 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110143 := (publicBatchWrapper86_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110144 := (publicBatchWrapper86_copies3442 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1803_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_18
  simp only [← c110139, ← c110140, ← c110141] at e_1803_18
  have e_1803_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1803 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_19
  simp only [← c110142, ← c110143, k2, ← c110144] at e_1803_19
  have hr := e_1803_19
  simp only [e_1803_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 7) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5990)) := by
  have c110145 := (publicBatchWrapper86_copies3442 a h).2.1
  have c110146 := (publicBatchWrapper86_copies3442 a h).2.2.1
  have c110147 := (publicBatchWrapper86_copies3442 a h).2.2.2.1
  have c110148 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.1
  have c110149 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.1
  have c110150 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_0
  simp only [← c110145, ← c110146, ← c110147] at e_1804_0
  have e_1804_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_1
  simp only [← c110148, ← c110149, k2, ← c110150] at e_1804_1
  have hr := e_1804_1
  simp only [e_1804_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 15) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5991)) := by
  have c110151 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.1
  have c110152 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.1
  have c110153 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.1
  have c110154 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110155 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110156 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_2
  simp only [← c110151, ← c110152, ← c110153] at e_1804_2
  have e_1804_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_3
  simp only [← c110154, ← c110155, k2, ← c110156] at e_1804_3
  have hr := e_1804_3
  simp only [e_1804_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 23) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5992)) := by
  have c110157 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110158 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110159 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110160 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110161 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110162 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_4
  simp only [← c110157, ← c110158, ← c110159] at e_1804_4
  have e_1804_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_5
  simp only [← c110160, ← c110161, k2, ← c110162] at e_1804_5
  have hr := e_1804_5
  simp only [e_1804_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 31) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5993)) := by
  have c110163 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110164 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110165 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110166 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110167 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110168 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_6
  simp only [← c110163, ← c110164, ← c110165] at e_1804_6
  have e_1804_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_7
  simp only [← c110166, ← c110167, k2, ← c110168] at e_1804_7
  have hr := e_1804_7
  simp only [e_1804_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 39) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5994)) := by
  have c110169 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110170 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110171 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110172 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110173 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110174 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_8
  simp only [← c110169, ← c110170, ← c110171] at e_1804_8
  have e_1804_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_9
  simp only [← c110172, ← c110173, k2, ← c110174] at e_1804_9
  have hr := e_1804_9
  simp only [e_1804_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 47) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5995)) := by
  have c110175 := (publicBatchWrapper86_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110176 := (publicBatchWrapper86_copies3443 a h).1
  have c110177 := (publicBatchWrapper86_copies3443 a h).2.1
  have c110178 := (publicBatchWrapper86_copies3443 a h).2.2.1
  have c110179 := (publicBatchWrapper86_copies3443 a h).2.2.2.1
  have c110180 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_10
  simp only [← c110175, ← c110176, ← c110177] at e_1804_10
  have e_1804_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_11
  simp only [← c110178, ← c110179, k2, ← c110180] at e_1804_11
  have hr := e_1804_11
  simp only [e_1804_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 55) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5996)) := by
  have c110181 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.1
  have c110182 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.1
  have c110183 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.1
  have c110184 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.1
  have c110185 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.1
  have c110186 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_12
  simp only [← c110181, ← c110182, ← c110183] at e_1804_12
  have e_1804_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_13
  simp only [← c110184, ← c110185, k2, ← c110186] at e_1804_13
  have hr := e_1804_13
  simp only [e_1804_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 63) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5997)) := by
  have c110187 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110188 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110189 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110190 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110191 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110192 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_14
  simp only [← c110187, ← c110188, ← c110189] at e_1804_14
  have e_1804_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_15
  simp only [← c110190, ← c110191, k2, ← c110192] at e_1804_15
  have hr := e_1804_15
  simp only [e_1804_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 71) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5998)) := by
  have c110193 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110194 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110195 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110196 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110197 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110198 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_16
  simp only [← c110193, ← c110194, ← c110195] at e_1804_16
  have e_1804_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_17
  simp only [← c110196, ← c110197, k2, ← c110198] at e_1804_17
  have hr := e_1804_17
  simp only [e_1804_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1804 79) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 5999)) := by
  have c110199 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110200 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110201 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110202 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110203 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110204 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1804_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_18
  simp only [← c110199, ← c110200, ← c110201] at e_1804_18
  have e_1804_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1804 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_19
  simp only [← c110202, ← c110203, k2, ← c110204] at e_1804_19
  have hr := e_1804_19
  simp only [e_1804_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 7) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 6000)) := by
  have c110205 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110206 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110207 := (publicBatchWrapper86_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110208 := (publicBatchWrapper86_copies3444 a h).1
  have c110209 := (publicBatchWrapper86_copies3444 a h).2.1
  have c110210 := (publicBatchWrapper86_copies3444 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_0
  simp only [← c110205, ← c110206, ← c110207] at e_1805_0
  have e_1805_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_1
  simp only [← c110208, ← c110209, k2, ← c110210] at e_1805_1
  have hr := e_1805_1
  simp only [e_1805_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 15) = bselect (a (.wire 30 23)) (a (.virt 200)) (a (.virt 6001)) := by
  have c110211 := (publicBatchWrapper86_copies3444 a h).2.2.2.1
  have c110212 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.1
  have c110213 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.1
  have c110214 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.1
  have c110215 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.1
  have c110216 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_2
  simp only [← c110211, ← c110212, ← c110213] at e_1805_2
  have e_1805_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_3
  simp only [← c110214, ← c110215, k2, ← c110216] at e_1805_3
  have hr := e_1805_3
  simp only [e_1805_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 23) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6170)) := by
  have c110217 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.1
  have c110218 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110219 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110220 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110221 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110222 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_4
  simp only [← c110217, ← c110218, ← c110219] at e_1805_4
  have e_1805_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_5
  simp only [← c110220, ← c110221, k2, ← c110222] at e_1805_5
  have hr := e_1805_5
  simp only [e_1805_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 31) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6171)) := by
  have c110223 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110224 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110225 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110226 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110227 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110228 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_6
  simp only [← c110223, ← c110224, ← c110225] at e_1805_6
  have e_1805_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_7
  simp only [← c110226, ← c110227, k2, ← c110228] at e_1805_7
  have hr := e_1805_7
  simp only [e_1805_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 39) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6172)) := by
  have c110229 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110230 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110231 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110232 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110233 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110234 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_8
  simp only [← c110229, ← c110230, ← c110231] at e_1805_8
  have e_1805_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_9
  simp only [← c110232, ← c110233, k2, ← c110234] at e_1805_9
  have hr := e_1805_9
  simp only [e_1805_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 47) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6173)) := by
  have c110235 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110236 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110237 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110238 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110239 := (publicBatchWrapper86_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110240 := (publicBatchWrapper86_copies3445 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_10
  simp only [← c110235, ← c110236, ← c110237] at e_1805_10
  have e_1805_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_11
  simp only [← c110238, ← c110239, k2, ← c110240] at e_1805_11
  have hr := e_1805_11
  simp only [e_1805_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 55) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6174)) := by
  have c110241 := (publicBatchWrapper86_copies3445 a h).2.1
  have c110242 := (publicBatchWrapper86_copies3445 a h).2.2.1
  have c110243 := (publicBatchWrapper86_copies3445 a h).2.2.2.1
  have c110244 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.1
  have c110245 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.1
  have c110246 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_12
  simp only [← c110241, ← c110242, ← c110243] at e_1805_12
  have e_1805_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_13
  simp only [← c110244, ← c110245, k2, ← c110246] at e_1805_13
  have hr := e_1805_13
  simp only [e_1805_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 63) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6175)) := by
  have c110247 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.1
  have c110248 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.1
  have c110249 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.1
  have c110250 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110251 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110252 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_14
  simp only [← c110247, ← c110248, ← c110249] at e_1805_14
  have e_1805_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_15
  simp only [← c110250, ← c110251, k2, ← c110252] at e_1805_15
  have hr := e_1805_15
  simp only [e_1805_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 71) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6176)) := by
  have c110253 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110254 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110255 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110256 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110257 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110258 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_16
  simp only [← c110253, ← c110254, ← c110255] at e_1805_16
  have e_1805_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_17
  simp only [← c110256, ← c110257, k2, ← c110258] at e_1805_17
  have hr := e_1805_17
  simp only [e_1805_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1805 79) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6177)) := by
  have c110259 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110260 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110261 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110262 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110263 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110264 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1805_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_18
  simp only [← c110259, ← c110260, ← c110261] at e_1805_18
  have e_1805_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1805 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_19
  simp only [← c110262, ← c110263, k2, ← c110264] at e_1805_19
  have hr := e_1805_19
  simp only [e_1805_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 7) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6178)) := by
  have c110265 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110266 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110267 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110268 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110269 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110270 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_0
  simp only [← c110265, ← c110266, ← c110267] at e_1806_0
  have e_1806_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_1
  simp only [← c110268, ← c110269, k2, ← c110270] at e_1806_1
  have hr := e_1806_1
  simp only [e_1806_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 15) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6179)) := by
  have c110271 := (publicBatchWrapper86_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110272 := (publicBatchWrapper86_copies3446 a h).1
  have c110273 := (publicBatchWrapper86_copies3446 a h).2.1
  have c110274 := (publicBatchWrapper86_copies3446 a h).2.2.1
  have c110275 := (publicBatchWrapper86_copies3446 a h).2.2.2.1
  have c110276 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_2
  simp only [← c110271, ← c110272, ← c110273] at e_1806_2
  have e_1806_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_3
  simp only [← c110274, ← c110275, k2, ← c110276] at e_1806_3
  have hr := e_1806_3
  simp only [e_1806_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 23) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6180)) := by
  have c110277 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.1
  have c110278 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.1
  have c110279 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.1
  have c110280 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.1
  have c110281 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.1
  have c110282 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_4
  simp only [← c110277, ← c110278, ← c110279] at e_1806_4
  have e_1806_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_5
  simp only [← c110280, ← c110281, k2, ← c110282] at e_1806_5
  have hr := e_1806_5
  simp only [e_1806_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 31) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6181)) := by
  have c110283 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110284 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110285 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110286 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110287 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110288 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_6
  simp only [← c110283, ← c110284, ← c110285] at e_1806_6
  have e_1806_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_7
  simp only [← c110286, ← c110287, k2, ← c110288] at e_1806_7
  have hr := e_1806_7
  simp only [e_1806_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 39) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6182)) := by
  have c110289 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110290 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110291 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110292 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110293 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110294 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_8
  simp only [← c110289, ← c110290, ← c110291] at e_1806_8
  have e_1806_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_9
  simp only [← c110292, ← c110293, k2, ← c110294] at e_1806_9
  have hr := e_1806_9
  simp only [e_1806_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 47) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6183)) := by
  have c110295 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110296 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110297 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110298 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110299 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110300 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_10
  simp only [← c110295, ← c110296, ← c110297] at e_1806_10
  have e_1806_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_11
  simp only [← c110298, ← c110299, k2, ← c110300] at e_1806_11
  have hr := e_1806_11
  simp only [e_1806_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 55) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6184)) := by
  have c110301 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110302 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110303 := (publicBatchWrapper86_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110304 := (publicBatchWrapper86_copies3447 a h).1
  have c110305 := (publicBatchWrapper86_copies3447 a h).2.1
  have c110306 := (publicBatchWrapper86_copies3447 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_12
  simp only [← c110301, ← c110302, ← c110303] at e_1806_12
  have e_1806_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_13
  simp only [← c110304, ← c110305, k2, ← c110306] at e_1806_13
  have hr := e_1806_13
  simp only [e_1806_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 63) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6185)) := by
  have c110307 := (publicBatchWrapper86_copies3447 a h).2.2.2.1
  have c110308 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.1
  have c110309 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.1
  have c110310 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.1
  have c110311 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.1
  have c110312 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_14
  simp only [← c110307, ← c110308, ← c110309] at e_1806_14
  have e_1806_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_15
  simp only [← c110310, ← c110311, k2, ← c110312] at e_1806_15
  have hr := e_1806_15
  simp only [e_1806_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 71) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6186)) := by
  have c110313 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.1
  have c110314 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110315 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110316 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110317 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110318 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_16
  simp only [← c110313, ← c110314, ← c110315] at e_1806_16
  have e_1806_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_17
  simp only [← c110316, ← c110317, k2, ← c110318] at e_1806_17
  have hr := e_1806_17
  simp only [e_1806_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1806 79) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6187)) := by
  have c110319 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110320 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110321 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110322 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110323 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110324 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1806_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_18
  simp only [← c110319, ← c110320, ← c110321] at e_1806_18
  have e_1806_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1806 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_19
  simp only [← c110322, ← c110323, k2, ← c110324] at e_1806_19
  have hr := e_1806_19
  simp only [e_1806_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 7) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6188)) := by
  have c110325 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110326 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110327 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110328 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110329 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110330 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_0
  simp only [← c110325, ← c110326, ← c110327] at e_1807_0
  have e_1807_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_1
  simp only [← c110328, ← c110329, k2, ← c110330] at e_1807_1
  have hr := e_1807_1
  simp only [e_1807_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 15) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6189)) := by
  have c110331 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110332 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110333 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110334 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110335 := (publicBatchWrapper86_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110336 := (publicBatchWrapper86_copies3448 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_2
  simp only [← c110331, ← c110332, ← c110333] at e_1807_2
  have e_1807_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_3
  simp only [← c110334, ← c110335, k2, ← c110336] at e_1807_3
  have hr := e_1807_3
  simp only [e_1807_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 23) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6190)) := by
  have c110337 := (publicBatchWrapper86_copies3448 a h).2.1
  have c110338 := (publicBatchWrapper86_copies3448 a h).2.2.1
  have c110339 := (publicBatchWrapper86_copies3448 a h).2.2.2.1
  have c110340 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.1
  have c110341 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.1
  have c110342 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_4
  simp only [← c110337, ← c110338, ← c110339] at e_1807_4
  have e_1807_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_5
  simp only [← c110340, ← c110341, k2, ← c110342] at e_1807_5
  have hr := e_1807_5
  simp only [e_1807_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 31) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6191)) := by
  have c110343 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.1
  have c110344 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.1
  have c110345 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.1
  have c110346 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110347 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110348 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_6
  simp only [← c110343, ← c110344, ← c110345] at e_1807_6
  have e_1807_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_7
  simp only [← c110346, ← c110347, k2, ← c110348] at e_1807_7
  have hr := e_1807_7
  simp only [e_1807_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 39) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6192)) := by
  have c110349 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110350 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110351 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110352 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110353 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110354 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_8
  simp only [← c110349, ← c110350, ← c110351] at e_1807_8
  have e_1807_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_9
  simp only [← c110352, ← c110353, k2, ← c110354] at e_1807_9
  have hr := e_1807_9
  simp only [e_1807_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 47) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6193)) := by
  have c110355 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110356 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110357 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110358 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110359 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110360 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_10
  simp only [← c110355, ← c110356, ← c110357] at e_1807_10
  have e_1807_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_11
  simp only [← c110358, ← c110359, k2, ← c110360] at e_1807_11
  have hr := e_1807_11
  simp only [e_1807_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 55) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6194)) := by
  have c110361 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110362 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110363 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110364 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110365 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110366 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_12
  simp only [← c110361, ← c110362, ← c110363] at e_1807_12
  have e_1807_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_13
  simp only [← c110364, ← c110365, k2, ← c110366] at e_1807_13
  have hr := e_1807_13
  simp only [e_1807_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 63) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6195)) := by
  have c110367 := (publicBatchWrapper86_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110368 := (publicBatchWrapper86_copies3449 a h).1
  have c110369 := (publicBatchWrapper86_copies3449 a h).2.1
  have c110370 := (publicBatchWrapper86_copies3449 a h).2.2.1
  have c110371 := (publicBatchWrapper86_copies3449 a h).2.2.2.1
  have c110372 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_14
  simp only [← c110367, ← c110368, ← c110369] at e_1807_14
  have e_1807_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_15
  simp only [← c110370, ← c110371, k2, ← c110372] at e_1807_15
  have hr := e_1807_15
  simp only [e_1807_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 71) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6196)) := by
  have c110373 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.1
  have c110374 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.1
  have c110375 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.1
  have c110376 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.1
  have c110377 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.1
  have c110378 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_16
  simp only [← c110373, ← c110374, ← c110375] at e_1807_16
  have e_1807_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_17
  simp only [← c110376, ← c110377, k2, ← c110378] at e_1807_17
  have hr := e_1807_17
  simp only [e_1807_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1807 79) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6197)) := by
  have c110379 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110380 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110381 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110382 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110383 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110384 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1807_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_18
  simp only [← c110379, ← c110380, ← c110381] at e_1807_18
  have e_1807_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1807 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_19
  simp only [← c110382, ← c110383, k2, ← c110384] at e_1807_19
  have hr := e_1807_19
  simp only [e_1807_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 7) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6198)) := by
  have c110385 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110386 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110387 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110388 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110389 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110390 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_0
  simp only [← c110385, ← c110386, ← c110387] at e_1808_0
  have e_1808_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_1
  simp only [← c110388, ← c110389, k2, ← c110390] at e_1808_1
  have hr := e_1808_1
  simp only [e_1808_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 15) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6199)) := by
  have c110391 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110392 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110393 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110394 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110395 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110396 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_2
  simp only [← c110391, ← c110392, ← c110393] at e_1808_2
  have e_1808_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_3
  simp only [← c110394, ← c110395, k2, ← c110396] at e_1808_3
  have hr := e_1808_3
  simp only [e_1808_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 23) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6200)) := by
  have c110397 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110398 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110399 := (publicBatchWrapper86_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110400 := (publicBatchWrapper86_copies3450 a h).1
  have c110401 := (publicBatchWrapper86_copies3450 a h).2.1
  have c110402 := (publicBatchWrapper86_copies3450 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_4
  simp only [← c110397, ← c110398, ← c110399] at e_1808_4
  have e_1808_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_5
  simp only [← c110400, ← c110401, k2, ← c110402] at e_1808_5
  have hr := e_1808_5
  simp only [e_1808_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 31) = bselect (a (.wire 30 67)) (a (.virt 200)) (a (.virt 6201)) := by
  have c110403 := (publicBatchWrapper86_copies3450 a h).2.2.2.1
  have c110404 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.1
  have c110405 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.1
  have c110406 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.1
  have c110407 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.1
  have c110408 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_6
  simp only [← c110403, ← c110404, ← c110405] at e_1808_6
  have e_1808_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_7
  simp only [← c110406, ← c110407, k2, ← c110408] at e_1808_7
  have hr := e_1808_7
  simp only [e_1808_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 39) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6370)) := by
  have c110409 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.1
  have c110410 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110411 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110412 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110413 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110414 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_8
  simp only [← c110409, ← c110410, ← c110411] at e_1808_8
  have e_1808_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_9
  simp only [← c110412, ← c110413, k2, ← c110414] at e_1808_9
  have hr := e_1808_9
  simp only [e_1808_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 47) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6371)) := by
  have c110415 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110416 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110417 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110418 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110419 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110420 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_10
  simp only [← c110415, ← c110416, ← c110417] at e_1808_10
  have e_1808_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_11
  simp only [← c110418, ← c110419, k2, ← c110420] at e_1808_11
  have hr := e_1808_11
  simp only [e_1808_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 55) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6372)) := by
  have c110421 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110422 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110423 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110424 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110425 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110426 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_12
  simp only [← c110421, ← c110422, ← c110423] at e_1808_12
  have e_1808_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_13
  simp only [← c110424, ← c110425, k2, ← c110426] at e_1808_13
  have hr := e_1808_13
  simp only [e_1808_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 63) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6373)) := by
  have c110427 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110428 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110429 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110430 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110431 := (publicBatchWrapper86_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110432 := (publicBatchWrapper86_copies3451 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_14
  simp only [← c110427, ← c110428, ← c110429] at e_1808_14
  have e_1808_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_15
  simp only [← c110430, ← c110431, k2, ← c110432] at e_1808_15
  have hr := e_1808_15
  simp only [e_1808_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 71) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6374)) := by
  have c110433 := (publicBatchWrapper86_copies3451 a h).2.1
  have c110434 := (publicBatchWrapper86_copies3451 a h).2.2.1
  have c110435 := (publicBatchWrapper86_copies3451 a h).2.2.2.1
  have c110436 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.1
  have c110437 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.1
  have c110438 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_16
  simp only [← c110433, ← c110434, ← c110435] at e_1808_16
  have e_1808_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_17
  simp only [← c110436, ← c110437, k2, ← c110438] at e_1808_17
  have hr := e_1808_17
  simp only [e_1808_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1808 79) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6375)) := by
  have c110439 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.1
  have c110440 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.1
  have c110441 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.1
  have c110442 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110443 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110444 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1808_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_18
  simp only [← c110439, ← c110440, ← c110441] at e_1808_18
  have e_1808_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1808 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_19
  simp only [← c110442, ← c110443, k2, ← c110444] at e_1808_19
  have hr := e_1808_19
  simp only [e_1808_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 7) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6376)) := by
  have c110445 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110446 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110447 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110448 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110449 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110450 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_0
  simp only [← c110445, ← c110446, ← c110447] at e_1809_0
  have e_1809_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_1
  simp only [← c110448, ← c110449, k2, ← c110450] at e_1809_1
  have hr := e_1809_1
  simp only [e_1809_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 15) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6377)) := by
  have c110451 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110452 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110453 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110454 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110455 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110456 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_2
  simp only [← c110451, ← c110452, ← c110453] at e_1809_2
  have e_1809_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_3
  simp only [← c110454, ← c110455, k2, ← c110456] at e_1809_3
  have hr := e_1809_3
  simp only [e_1809_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 23) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6378)) := by
  have c110457 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110458 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110459 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110460 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110461 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110462 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_4
  simp only [← c110457, ← c110458, ← c110459] at e_1809_4
  have e_1809_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_5
  simp only [← c110460, ← c110461, k2, ← c110462] at e_1809_5
  have hr := e_1809_5
  simp only [e_1809_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 31) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6379)) := by
  have c110463 := (publicBatchWrapper86_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110464 := (publicBatchWrapper86_copies3452 a h).1
  have c110465 := (publicBatchWrapper86_copies3452 a h).2.1
  have c110466 := (publicBatchWrapper86_copies3452 a h).2.2.1
  have c110467 := (publicBatchWrapper86_copies3452 a h).2.2.2.1
  have c110468 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_6
  simp only [← c110463, ← c110464, ← c110465] at e_1809_6
  have e_1809_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_7
  simp only [← c110466, ← c110467, k2, ← c110468] at e_1809_7
  have hr := e_1809_7
  simp only [e_1809_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 39) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6380)) := by
  have c110469 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.1
  have c110470 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.1
  have c110471 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.1
  have c110472 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.1
  have c110473 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.1
  have c110474 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_8
  simp only [← c110469, ← c110470, ← c110471] at e_1809_8
  have e_1809_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_9
  simp only [← c110472, ← c110473, k2, ← c110474] at e_1809_9
  have hr := e_1809_9
  simp only [e_1809_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 47) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6381)) := by
  have c110475 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110476 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110477 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110478 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110479 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110480 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_10
  simp only [← c110475, ← c110476, ← c110477] at e_1809_10
  have e_1809_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_11
  simp only [← c110478, ← c110479, k2, ← c110480] at e_1809_11
  have hr := e_1809_11
  simp only [e_1809_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 55) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6382)) := by
  have c110481 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110482 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110483 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110484 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110485 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110486 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_12
  simp only [← c110481, ← c110482, ← c110483] at e_1809_12
  have e_1809_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_13
  simp only [← c110484, ← c110485, k2, ← c110486] at e_1809_13
  have hr := e_1809_13
  simp only [e_1809_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 63) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6383)) := by
  have c110487 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110488 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110489 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110490 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110491 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110492 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_14
  simp only [← c110487, ← c110488, ← c110489] at e_1809_14
  have e_1809_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_15
  simp only [← c110490, ← c110491, k2, ← c110492] at e_1809_15
  have hr := e_1809_15
  simp only [e_1809_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 71) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6384)) := by
  have c110493 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110494 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110495 := (publicBatchWrapper86_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110496 := (publicBatchWrapper86_copies3453 a h).1
  have c110497 := (publicBatchWrapper86_copies3453 a h).2.1
  have c110498 := (publicBatchWrapper86_copies3453 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_16
  simp only [← c110493, ← c110494, ← c110495] at e_1809_16
  have e_1809_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_17
  simp only [← c110496, ← c110497, k2, ← c110498] at e_1809_17
  have hr := e_1809_17
  simp only [e_1809_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1809 79) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6385)) := by
  have c110499 := (publicBatchWrapper86_copies3453 a h).2.2.2.1
  have c110500 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.1
  have c110501 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.1
  have c110502 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.1
  have c110503 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.1
  have c110504 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1809_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_18
  simp only [← c110499, ← c110500, ← c110501] at e_1809_18
  have e_1809_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1809 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_19
  simp only [← c110502, ← c110503, k2, ← c110504] at e_1809_19
  have hr := e_1809_19
  simp only [e_1809_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 7) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6386)) := by
  have c110505 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.1
  have c110506 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110507 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110508 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110509 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110510 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_0
  simp only [← c110505, ← c110506, ← c110507] at e_1810_0
  have e_1810_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_1
  simp only [← c110508, ← c110509, k2, ← c110510] at e_1810_1
  have hr := e_1810_1
  simp only [e_1810_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 15) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6387)) := by
  have c110511 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110512 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110513 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110514 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110515 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110516 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_2
  simp only [← c110511, ← c110512, ← c110513] at e_1810_2
  have e_1810_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_3
  simp only [← c110514, ← c110515, k2, ← c110516] at e_1810_3
  have hr := e_1810_3
  simp only [e_1810_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 23) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6388)) := by
  have c110517 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110518 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110519 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110520 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110521 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110522 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_4
  simp only [← c110517, ← c110518, ← c110519] at e_1810_4
  have e_1810_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_5
  simp only [← c110520, ← c110521, k2, ← c110522] at e_1810_5
  have hr := e_1810_5
  simp only [e_1810_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 31) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6389)) := by
  have c110523 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110524 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110525 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110526 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110527 := (publicBatchWrapper86_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110528 := (publicBatchWrapper86_copies3454 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_6
  simp only [← c110523, ← c110524, ← c110525] at e_1810_6
  have e_1810_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_7
  simp only [← c110526, ← c110527, k2, ← c110528] at e_1810_7
  have hr := e_1810_7
  simp only [e_1810_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 39) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6390)) := by
  have c110529 := (publicBatchWrapper86_copies3454 a h).2.1
  have c110530 := (publicBatchWrapper86_copies3454 a h).2.2.1
  have c110531 := (publicBatchWrapper86_copies3454 a h).2.2.2.1
  have c110532 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.1
  have c110533 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.1
  have c110534 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_8
  simp only [← c110529, ← c110530, ← c110531] at e_1810_8
  have e_1810_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_9
  simp only [← c110532, ← c110533, k2, ← c110534] at e_1810_9
  have hr := e_1810_9
  simp only [e_1810_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 47) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6391)) := by
  have c110535 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.1
  have c110536 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.1
  have c110537 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.1
  have c110538 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110539 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110540 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_10
  simp only [← c110535, ← c110536, ← c110537] at e_1810_10
  have e_1810_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_11
  simp only [← c110538, ← c110539, k2, ← c110540] at e_1810_11
  have hr := e_1810_11
  simp only [e_1810_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 55) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6392)) := by
  have c110541 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110542 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110543 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110544 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110545 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110546 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_12
  simp only [← c110541, ← c110542, ← c110543] at e_1810_12
  have e_1810_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_13
  simp only [← c110544, ← c110545, k2, ← c110546] at e_1810_13
  have hr := e_1810_13
  simp only [e_1810_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 63) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6393)) := by
  have c110547 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110548 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110549 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110550 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110551 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110552 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_14
  simp only [← c110547, ← c110548, ← c110549] at e_1810_14
  have e_1810_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_15
  simp only [← c110550, ← c110551, k2, ← c110552] at e_1810_15
  have hr := e_1810_15
  simp only [e_1810_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 71) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6394)) := by
  have c110553 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110554 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110555 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110556 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110557 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110558 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_16
  simp only [← c110553, ← c110554, ← c110555] at e_1810_16
  have e_1810_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_17
  simp only [← c110556, ← c110557, k2, ← c110558] at e_1810_17
  have hr := e_1810_17
  simp only [e_1810_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1810 79) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6395)) := by
  have c110559 := (publicBatchWrapper86_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110560 := (publicBatchWrapper86_copies3455 a h).1
  have c110561 := (publicBatchWrapper86_copies3455 a h).2.1
  have c110562 := (publicBatchWrapper86_copies3455 a h).2.2.1
  have c110563 := (publicBatchWrapper86_copies3455 a h).2.2.2.1
  have c110564 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1810_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_18
  simp only [← c110559, ← c110560, ← c110561] at e_1810_18
  have e_1810_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1810 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_19
  simp only [← c110562, ← c110563, k2, ← c110564] at e_1810_19
  have hr := e_1810_19
  simp only [e_1810_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 7) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6396)) := by
  have c110565 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.1
  have c110566 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.1
  have c110567 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.1
  have c110568 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.1
  have c110569 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.1
  have c110570 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_0
  simp only [← c110565, ← c110566, ← c110567] at e_1811_0
  have e_1811_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_1
  simp only [← c110568, ← c110569, k2, ← c110570] at e_1811_1
  have hr := e_1811_1
  simp only [e_1811_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 15) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6397)) := by
  have c110571 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110572 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110573 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110574 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110575 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110576 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_2
  simp only [← c110571, ← c110572, ← c110573] at e_1811_2
  have e_1811_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_3
  simp only [← c110574, ← c110575, k2, ← c110576] at e_1811_3
  have hr := e_1811_3
  simp only [e_1811_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 23) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6398)) := by
  have c110577 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110578 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110579 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110580 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110581 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110582 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_4
  simp only [← c110577, ← c110578, ← c110579] at e_1811_4
  have e_1811_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_5
  simp only [← c110580, ← c110581, k2, ← c110582] at e_1811_5
  have hr := e_1811_5
  simp only [e_1811_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 31) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6399)) := by
  have c110583 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110584 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110585 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110586 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110587 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110588 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_6
  simp only [← c110583, ← c110584, ← c110585] at e_1811_6
  have e_1811_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_7
  simp only [← c110586, ← c110587, k2, ← c110588] at e_1811_7
  have hr := e_1811_7
  simp only [e_1811_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 39) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6400)) := by
  have c110589 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110590 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110591 := (publicBatchWrapper86_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110592 := (publicBatchWrapper86_copies3456 a h).1
  have c110593 := (publicBatchWrapper86_copies3456 a h).2.1
  have c110594 := (publicBatchWrapper86_copies3456 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_8
  simp only [← c110589, ← c110590, ← c110591] at e_1811_8
  have e_1811_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_9
  simp only [← c110592, ← c110593, k2, ← c110594] at e_1811_9
  have hr := e_1811_9
  simp only [e_1811_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 47) = bselect (a (.wire 32 31)) (a (.virt 200)) (a (.virt 6401)) := by
  have c110595 := (publicBatchWrapper86_copies3456 a h).2.2.2.1
  have c110596 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.1
  have c110597 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.1
  have c110598 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.1
  have c110599 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.1
  have c110600 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_10
  simp only [← c110595, ← c110596, ← c110597] at e_1811_10
  have e_1811_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_11
  simp only [← c110598, ← c110599, k2, ← c110600] at e_1811_11
  have hr := e_1811_11
  simp only [e_1811_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 55) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6570)) := by
  have c110601 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.1
  have c110602 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110603 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110604 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110605 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110606 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_12
  simp only [← c110601, ← c110602, ← c110603] at e_1811_12
  have e_1811_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_13
  simp only [← c110604, ← c110605, k2, ← c110606] at e_1811_13
  have hr := e_1811_13
  simp only [e_1811_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 63) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6571)) := by
  have c110607 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110608 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110609 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110610 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110611 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110612 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_14
  simp only [← c110607, ← c110608, ← c110609] at e_1811_14
  have e_1811_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_15
  simp only [← c110610, ← c110611, k2, ← c110612] at e_1811_15
  have hr := e_1811_15
  simp only [e_1811_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 71) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6572)) := by
  have c110613 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110614 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110615 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110616 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110617 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110618 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_16
  simp only [← c110613, ← c110614, ← c110615] at e_1811_16
  have e_1811_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_17
  simp only [← c110616, ← c110617, k2, ← c110618] at e_1811_17
  have hr := e_1811_17
  simp only [e_1811_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1811 79) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6573)) := by
  have c110619 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110620 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110621 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110622 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110623 := (publicBatchWrapper86_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110624 := (publicBatchWrapper86_copies3457 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1811_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_18
  simp only [← c110619, ← c110620, ← c110621] at e_1811_18
  have e_1811_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1811 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_19
  simp only [← c110622, ← c110623, k2, ← c110624] at e_1811_19
  have hr := e_1811_19
  simp only [e_1811_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 7) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6574)) := by
  have c110625 := (publicBatchWrapper86_copies3457 a h).2.1
  have c110626 := (publicBatchWrapper86_copies3457 a h).2.2.1
  have c110627 := (publicBatchWrapper86_copies3457 a h).2.2.2.1
  have c110628 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.1
  have c110629 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.1
  have c110630 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_0
  simp only [← c110625, ← c110626, ← c110627] at e_1812_0
  have e_1812_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_1
  simp only [← c110628, ← c110629, k2, ← c110630] at e_1812_1
  have hr := e_1812_1
  simp only [e_1812_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 15) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6575)) := by
  have c110631 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.1
  have c110632 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.1
  have c110633 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.1
  have c110634 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110635 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110636 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_2
  simp only [← c110631, ← c110632, ← c110633] at e_1812_2
  have e_1812_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_3
  simp only [← c110634, ← c110635, k2, ← c110636] at e_1812_3
  have hr := e_1812_3
  simp only [e_1812_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 23) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6576)) := by
  have c110637 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110638 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110639 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110640 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110641 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110642 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_4
  simp only [← c110637, ← c110638, ← c110639] at e_1812_4
  have e_1812_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_5
  simp only [← c110640, ← c110641, k2, ← c110642] at e_1812_5
  have hr := e_1812_5
  simp only [e_1812_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 31) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6577)) := by
  have c110643 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110644 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110645 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110646 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110647 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110648 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_6
  simp only [← c110643, ← c110644, ← c110645] at e_1812_6
  have e_1812_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_7
  simp only [← c110646, ← c110647, k2, ← c110648] at e_1812_7
  have hr := e_1812_7
  simp only [e_1812_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 39) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6578)) := by
  have c110649 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110650 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110651 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110652 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110653 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110654 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_8
  simp only [← c110649, ← c110650, ← c110651] at e_1812_8
  have e_1812_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_9
  simp only [← c110652, ← c110653, k2, ← c110654] at e_1812_9
  have hr := e_1812_9
  simp only [e_1812_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 47) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6579)) := by
  have c110655 := (publicBatchWrapper86_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110656 := (publicBatchWrapper86_copies3458 a h).1
  have c110657 := (publicBatchWrapper86_copies3458 a h).2.1
  have c110658 := (publicBatchWrapper86_copies3458 a h).2.2.1
  have c110659 := (publicBatchWrapper86_copies3458 a h).2.2.2.1
  have c110660 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_10
  simp only [← c110655, ← c110656, ← c110657] at e_1812_10
  have e_1812_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_11
  simp only [← c110658, ← c110659, k2, ← c110660] at e_1812_11
  have hr := e_1812_11
  simp only [e_1812_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 55) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6580)) := by
  have c110661 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.1
  have c110662 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.1
  have c110663 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.1
  have c110664 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.1
  have c110665 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.1
  have c110666 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_12
  simp only [← c110661, ← c110662, ← c110663] at e_1812_12
  have e_1812_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_13
  simp only [← c110664, ← c110665, k2, ← c110666] at e_1812_13
  have hr := e_1812_13
  simp only [e_1812_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 63) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6581)) := by
  have c110667 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110668 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110669 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110670 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110671 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110672 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_14
  simp only [← c110667, ← c110668, ← c110669] at e_1812_14
  have e_1812_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_15
  simp only [← c110670, ← c110671, k2, ← c110672] at e_1812_15
  have hr := e_1812_15
  simp only [e_1812_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 71) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6582)) := by
  have c110673 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110674 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110675 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110676 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110677 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110678 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_16
  simp only [← c110673, ← c110674, ← c110675] at e_1812_16
  have e_1812_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_17
  simp only [← c110676, ← c110677, k2, ← c110678] at e_1812_17
  have hr := e_1812_17
  simp only [e_1812_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1812 79) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6583)) := by
  have c110679 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110680 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110681 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110682 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110683 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110684 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1812_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_18
  simp only [← c110679, ← c110680, ← c110681] at e_1812_18
  have e_1812_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1812 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_19
  simp only [← c110682, ← c110683, k2, ← c110684] at e_1812_19
  have hr := e_1812_19
  simp only [e_1812_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 7) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6584)) := by
  have c110685 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110686 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110687 := (publicBatchWrapper86_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110688 := (publicBatchWrapper86_copies3459 a h).1
  have c110689 := (publicBatchWrapper86_copies3459 a h).2.1
  have c110690 := (publicBatchWrapper86_copies3459 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_0
  simp only [← c110685, ← c110686, ← c110687] at e_1813_0
  have e_1813_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_1
  simp only [← c110688, ← c110689, k2, ← c110690] at e_1813_1
  have hr := e_1813_1
  simp only [e_1813_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 15) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6585)) := by
  have c110691 := (publicBatchWrapper86_copies3459 a h).2.2.2.1
  have c110692 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.1
  have c110693 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.1
  have c110694 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.1
  have c110695 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.1
  have c110696 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_2
  simp only [← c110691, ← c110692, ← c110693] at e_1813_2
  have e_1813_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_3
  simp only [← c110694, ← c110695, k2, ← c110696] at e_1813_3
  have hr := e_1813_3
  simp only [e_1813_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 23) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6586)) := by
  have c110697 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.1
  have c110698 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110699 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110700 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110701 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110702 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_4
  simp only [← c110697, ← c110698, ← c110699] at e_1813_4
  have e_1813_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_5
  simp only [← c110700, ← c110701, k2, ← c110702] at e_1813_5
  have hr := e_1813_5
  simp only [e_1813_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 31) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6587)) := by
  have c110703 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110704 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110705 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110706 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110707 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110708 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_6
  simp only [← c110703, ← c110704, ← c110705] at e_1813_6
  have e_1813_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_7
  simp only [← c110706, ← c110707, k2, ← c110708] at e_1813_7
  have hr := e_1813_7
  simp only [e_1813_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 39) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6588)) := by
  have c110709 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110710 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110711 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110712 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110713 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110714 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_8
  simp only [← c110709, ← c110710, ← c110711] at e_1813_8
  have e_1813_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_9
  simp only [← c110712, ← c110713, k2, ← c110714] at e_1813_9
  have hr := e_1813_9
  simp only [e_1813_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 47) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6589)) := by
  have c110715 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110716 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110717 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110718 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110719 := (publicBatchWrapper86_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110720 := (publicBatchWrapper86_copies3460 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_10
  simp only [← c110715, ← c110716, ← c110717] at e_1813_10
  have e_1813_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_11
  simp only [← c110718, ← c110719, k2, ← c110720] at e_1813_11
  have hr := e_1813_11
  simp only [e_1813_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 55) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6590)) := by
  have c110721 := (publicBatchWrapper86_copies3460 a h).2.1
  have c110722 := (publicBatchWrapper86_copies3460 a h).2.2.1
  have c110723 := (publicBatchWrapper86_copies3460 a h).2.2.2.1
  have c110724 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.1
  have c110725 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.1
  have c110726 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_12
  simp only [← c110721, ← c110722, ← c110723] at e_1813_12
  have e_1813_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_13
  simp only [← c110724, ← c110725, k2, ← c110726] at e_1813_13
  have hr := e_1813_13
  simp only [e_1813_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 63) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6591)) := by
  have c110727 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.1
  have c110728 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.1
  have c110729 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.1
  have c110730 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110731 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110732 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_14
  simp only [← c110727, ← c110728, ← c110729] at e_1813_14
  have e_1813_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_15
  simp only [← c110730, ← c110731, k2, ← c110732] at e_1813_15
  have hr := e_1813_15
  simp only [e_1813_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 71) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6592)) := by
  have c110733 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110734 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110735 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110736 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110737 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110738 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_16
  simp only [← c110733, ← c110734, ← c110735] at e_1813_16
  have e_1813_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_17
  simp only [← c110736, ← c110737, k2, ← c110738] at e_1813_17
  have hr := e_1813_17
  simp only [e_1813_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1813 79) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6593)) := by
  have c110739 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110740 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110741 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110742 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110743 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110744 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1813_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_18
  simp only [← c110739, ← c110740, ← c110741] at e_1813_18
  have e_1813_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1813 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_19
  simp only [← c110742, ← c110743, k2, ← c110744] at e_1813_19
  have hr := e_1813_19
  simp only [e_1813_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 7) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6594)) := by
  have c110745 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110746 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110747 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110748 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110749 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110750 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_0
  simp only [← c110745, ← c110746, ← c110747] at e_1814_0
  have e_1814_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_1
  simp only [← c110748, ← c110749, k2, ← c110750] at e_1814_1
  have hr := e_1814_1
  simp only [e_1814_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 15) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6595)) := by
  have c110751 := (publicBatchWrapper86_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110752 := (publicBatchWrapper86_copies3461 a h).1
  have c110753 := (publicBatchWrapper86_copies3461 a h).2.1
  have c110754 := (publicBatchWrapper86_copies3461 a h).2.2.1
  have c110755 := (publicBatchWrapper86_copies3461 a h).2.2.2.1
  have c110756 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_2
  simp only [← c110751, ← c110752, ← c110753] at e_1814_2
  have e_1814_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_3
  simp only [← c110754, ← c110755, k2, ← c110756] at e_1814_3
  have hr := e_1814_3
  simp only [e_1814_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 23) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6596)) := by
  have c110757 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.1
  have c110758 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.1
  have c110759 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.1
  have c110760 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.1
  have c110761 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.1
  have c110762 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_4
  simp only [← c110757, ← c110758, ← c110759] at e_1814_4
  have e_1814_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_5
  simp only [← c110760, ← c110761, k2, ← c110762] at e_1814_5
  have hr := e_1814_5
  simp only [e_1814_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 31) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6597)) := by
  have c110763 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110764 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110765 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110766 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110767 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110768 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_6
  simp only [← c110763, ← c110764, ← c110765] at e_1814_6
  have e_1814_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_7
  simp only [← c110766, ← c110767, k2, ← c110768] at e_1814_7
  have hr := e_1814_7
  simp only [e_1814_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 39) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6598)) := by
  have c110769 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110770 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110771 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110772 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110773 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110774 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_8
  simp only [← c110769, ← c110770, ← c110771] at e_1814_8
  have e_1814_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_9
  simp only [← c110772, ← c110773, k2, ← c110774] at e_1814_9
  have hr := e_1814_9
  simp only [e_1814_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 47) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6599)) := by
  have c110775 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110776 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110777 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110778 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110779 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110780 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_10
  simp only [← c110775, ← c110776, ← c110777] at e_1814_10
  have e_1814_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_11
  simp only [← c110778, ← c110779, k2, ← c110780] at e_1814_11
  have hr := e_1814_11
  simp only [e_1814_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 55) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6600)) := by
  have c110781 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110782 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110783 := (publicBatchWrapper86_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110784 := (publicBatchWrapper86_copies3462 a h).1
  have c110785 := (publicBatchWrapper86_copies3462 a h).2.1
  have c110786 := (publicBatchWrapper86_copies3462 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_12
  simp only [← c110781, ← c110782, ← c110783] at e_1814_12
  have e_1814_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_13
  simp only [← c110784, ← c110785, k2, ← c110786] at e_1814_13
  have hr := e_1814_13
  simp only [e_1814_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 63) = bselect (a (.wire 32 75)) (a (.virt 200)) (a (.virt 6601)) := by
  have c110787 := (publicBatchWrapper86_copies3462 a h).2.2.2.1
  have c110788 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.1
  have c110789 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.1
  have c110790 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.1
  have c110791 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.1
  have c110792 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_14
  simp only [← c110787, ← c110788, ← c110789] at e_1814_14
  have e_1814_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_15
  simp only [← c110790, ← c110791, k2, ← c110792] at e_1814_15
  have hr := e_1814_15
  simp only [e_1814_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 71) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6770)) := by
  have c110793 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.1
  have c110794 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110795 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110796 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110797 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110798 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_16
  simp only [← c110793, ← c110794, ← c110795] at e_1814_16
  have e_1814_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_17
  simp only [← c110796, ← c110797, k2, ← c110798] at e_1814_17
  have hr := e_1814_17
  simp only [e_1814_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1814 79) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6771)) := by
  have c110799 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110800 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110801 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110802 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110803 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110804 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1814_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_18
  simp only [← c110799, ← c110800, ← c110801] at e_1814_18
  have e_1814_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1814 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_19
  simp only [← c110802, ← c110803, k2, ← c110804] at e_1814_19
  have hr := e_1814_19
  simp only [e_1814_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 7) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6772)) := by
  have c110805 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110806 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110807 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110808 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110809 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110810 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_0
  simp only [← c110805, ← c110806, ← c110807] at e_1815_0
  have e_1815_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_1
  simp only [← c110808, ← c110809, k2, ← c110810] at e_1815_1
  have hr := e_1815_1
  simp only [e_1815_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 15) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6773)) := by
  have c110811 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110812 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110813 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110814 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110815 := (publicBatchWrapper86_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110816 := (publicBatchWrapper86_copies3463 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_2
  simp only [← c110811, ← c110812, ← c110813] at e_1815_2
  have e_1815_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_3
  simp only [← c110814, ← c110815, k2, ← c110816] at e_1815_3
  have hr := e_1815_3
  simp only [e_1815_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 23) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6774)) := by
  have c110817 := (publicBatchWrapper86_copies3463 a h).2.1
  have c110818 := (publicBatchWrapper86_copies3463 a h).2.2.1
  have c110819 := (publicBatchWrapper86_copies3463 a h).2.2.2.1
  have c110820 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.1
  have c110821 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.1
  have c110822 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_4
  simp only [← c110817, ← c110818, ← c110819] at e_1815_4
  have e_1815_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_5
  simp only [← c110820, ← c110821, k2, ← c110822] at e_1815_5
  have hr := e_1815_5
  simp only [e_1815_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 31) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6775)) := by
  have c110823 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.1
  have c110824 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.1
  have c110825 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.1
  have c110826 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110827 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110828 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_6
  simp only [← c110823, ← c110824, ← c110825] at e_1815_6
  have e_1815_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_7
  simp only [← c110826, ← c110827, k2, ← c110828] at e_1815_7
  have hr := e_1815_7
  simp only [e_1815_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 39) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6776)) := by
  have c110829 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110830 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110831 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110832 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110833 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110834 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_8
  simp only [← c110829, ← c110830, ← c110831] at e_1815_8
  have e_1815_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_9
  simp only [← c110832, ← c110833, k2, ← c110834] at e_1815_9
  have hr := e_1815_9
  simp only [e_1815_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 47) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6777)) := by
  have c110835 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110836 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110837 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110838 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110839 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110840 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_10
  simp only [← c110835, ← c110836, ← c110837] at e_1815_10
  have e_1815_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_11
  simp only [← c110838, ← c110839, k2, ← c110840] at e_1815_11
  have hr := e_1815_11
  simp only [e_1815_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 55) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6778)) := by
  have c110841 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110842 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110843 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110844 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110845 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110846 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_12
  simp only [← c110841, ← c110842, ← c110843] at e_1815_12
  have e_1815_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_13
  simp only [← c110844, ← c110845, k2, ← c110846] at e_1815_13
  have hr := e_1815_13
  simp only [e_1815_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 63) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6779)) := by
  have c110847 := (publicBatchWrapper86_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110848 := (publicBatchWrapper86_copies3464 a h).1
  have c110849 := (publicBatchWrapper86_copies3464 a h).2.1
  have c110850 := (publicBatchWrapper86_copies3464 a h).2.2.1
  have c110851 := (publicBatchWrapper86_copies3464 a h).2.2.2.1
  have c110852 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_14
  simp only [← c110847, ← c110848, ← c110849] at e_1815_14
  have e_1815_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_15
  simp only [← c110850, ← c110851, k2, ← c110852] at e_1815_15
  have hr := e_1815_15
  simp only [e_1815_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 71) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6780)) := by
  have c110853 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.1
  have c110854 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.1
  have c110855 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.1
  have c110856 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.1
  have c110857 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.1
  have c110858 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_16
  simp only [← c110853, ← c110854, ← c110855] at e_1815_16
  have e_1815_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_17
  simp only [← c110856, ← c110857, k2, ← c110858] at e_1815_17
  have hr := e_1815_17
  simp only [e_1815_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1815 79) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6781)) := by
  have c110859 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110860 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110861 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110862 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110863 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110864 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1815_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_18
  simp only [← c110859, ← c110860, ← c110861] at e_1815_18
  have e_1815_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1815 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_19
  simp only [← c110862, ← c110863, k2, ← c110864] at e_1815_19
  have hr := e_1815_19
  simp only [e_1815_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 7) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6782)) := by
  have c110865 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110866 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110867 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110868 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110869 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110870 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_0
  simp only [← c110865, ← c110866, ← c110867] at e_1816_0
  have e_1816_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_1
  simp only [← c110868, ← c110869, k2, ← c110870] at e_1816_1
  have hr := e_1816_1
  simp only [e_1816_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 15) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6783)) := by
  have c110871 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110872 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110873 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110874 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110875 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110876 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_2
  simp only [← c110871, ← c110872, ← c110873] at e_1816_2
  have e_1816_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_3
  simp only [← c110874, ← c110875, k2, ← c110876] at e_1816_3
  have hr := e_1816_3
  simp only [e_1816_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 23) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6784)) := by
  have c110877 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110878 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110879 := (publicBatchWrapper86_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110880 := (publicBatchWrapper86_copies3465 a h).1
  have c110881 := (publicBatchWrapper86_copies3465 a h).2.1
  have c110882 := (publicBatchWrapper86_copies3465 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_4
  simp only [← c110877, ← c110878, ← c110879] at e_1816_4
  have e_1816_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_5
  simp only [← c110880, ← c110881, k2, ← c110882] at e_1816_5
  have hr := e_1816_5
  simp only [e_1816_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 31) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6785)) := by
  have c110883 := (publicBatchWrapper86_copies3465 a h).2.2.2.1
  have c110884 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.1
  have c110885 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.1
  have c110886 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.1
  have c110887 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.1
  have c110888 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_6
  simp only [← c110883, ← c110884, ← c110885] at e_1816_6
  have e_1816_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_7
  simp only [← c110886, ← c110887, k2, ← c110888] at e_1816_7
  have hr := e_1816_7
  simp only [e_1816_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 39) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6786)) := by
  have c110889 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.1
  have c110890 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110891 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110892 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110893 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110894 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_8
  simp only [← c110889, ← c110890, ← c110891] at e_1816_8
  have e_1816_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_9
  simp only [← c110892, ← c110893, k2, ← c110894] at e_1816_9
  have hr := e_1816_9
  simp only [e_1816_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 47) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6787)) := by
  have c110895 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110896 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110897 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110898 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110899 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110900 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_10
  simp only [← c110895, ← c110896, ← c110897] at e_1816_10
  have e_1816_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_11
  simp only [← c110898, ← c110899, k2, ← c110900] at e_1816_11
  have hr := e_1816_11
  simp only [e_1816_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 55) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6788)) := by
  have c110901 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110902 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110903 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110904 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110905 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110906 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_12
  simp only [← c110901, ← c110902, ← c110903] at e_1816_12
  have e_1816_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_13
  simp only [← c110904, ← c110905, k2, ← c110906] at e_1816_13
  have hr := e_1816_13
  simp only [e_1816_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 63) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6789)) := by
  have c110907 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110908 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110909 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110910 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110911 := (publicBatchWrapper86_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110912 := (publicBatchWrapper86_copies3466 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_14
  simp only [← c110907, ← c110908, ← c110909] at e_1816_14
  have e_1816_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_15
  simp only [← c110910, ← c110911, k2, ← c110912] at e_1816_15
  have hr := e_1816_15
  simp only [e_1816_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 71) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6790)) := by
  have c110913 := (publicBatchWrapper86_copies3466 a h).2.1
  have c110914 := (publicBatchWrapper86_copies3466 a h).2.2.1
  have c110915 := (publicBatchWrapper86_copies3466 a h).2.2.2.1
  have c110916 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.1
  have c110917 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.1
  have c110918 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_16
  simp only [← c110913, ← c110914, ← c110915] at e_1816_16
  have e_1816_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_17
  simp only [← c110916, ← c110917, k2, ← c110918] at e_1816_17
  have hr := e_1816_17
  simp only [e_1816_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1816 79) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6791)) := by
  have c110919 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.1
  have c110920 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.1
  have c110921 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.1
  have c110922 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110923 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110924 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1816_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_18
  simp only [← c110919, ← c110920, ← c110921] at e_1816_18
  have e_1816_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1816 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_19
  simp only [← c110922, ← c110923, k2, ← c110924] at e_1816_19
  have hr := e_1816_19
  simp only [e_1816_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 7) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6792)) := by
  have c110925 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110926 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110927 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110928 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110929 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110930 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_0
  simp only [← c110925, ← c110926, ← c110927] at e_1817_0
  have e_1817_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_1
  simp only [← c110928, ← c110929, k2, ← c110930] at e_1817_1
  have hr := e_1817_1
  simp only [e_1817_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 15) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6793)) := by
  have c110931 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110932 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110933 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110934 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110935 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110936 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_2
  simp only [← c110931, ← c110932, ← c110933] at e_1817_2
  have e_1817_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_3
  simp only [← c110934, ← c110935, k2, ← c110936] at e_1817_3
  have hr := e_1817_3
  simp only [e_1817_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 23) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6794)) := by
  have c110937 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110938 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110939 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110940 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110941 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110942 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_4
  simp only [← c110937, ← c110938, ← c110939] at e_1817_4
  have e_1817_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_5
  simp only [← c110940, ← c110941, k2, ← c110942] at e_1817_5
  have hr := e_1817_5
  simp only [e_1817_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 31) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6795)) := by
  have c110943 := (publicBatchWrapper86_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110944 := (publicBatchWrapper86_copies3467 a h).1
  have c110945 := (publicBatchWrapper86_copies3467 a h).2.1
  have c110946 := (publicBatchWrapper86_copies3467 a h).2.2.1
  have c110947 := (publicBatchWrapper86_copies3467 a h).2.2.2.1
  have c110948 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_6
  simp only [← c110943, ← c110944, ← c110945] at e_1817_6
  have e_1817_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_7
  simp only [← c110946, ← c110947, k2, ← c110948] at e_1817_7
  have hr := e_1817_7
  simp only [e_1817_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 39) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6796)) := by
  have c110949 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.1
  have c110950 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.1
  have c110951 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.1
  have c110952 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.1
  have c110953 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.1
  have c110954 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_8
  simp only [← c110949, ← c110950, ← c110951] at e_1817_8
  have e_1817_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_9
  simp only [← c110952, ← c110953, k2, ← c110954] at e_1817_9
  have hr := e_1817_9
  simp only [e_1817_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 47) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6797)) := by
  have c110955 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110956 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110957 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110958 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110959 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110960 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_10
  simp only [← c110955, ← c110956, ← c110957] at e_1817_10
  have e_1817_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_11
  simp only [← c110958, ← c110959, k2, ← c110960] at e_1817_11
  have hr := e_1817_11
  simp only [e_1817_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 55) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6798)) := by
  have c110961 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110962 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110963 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110964 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110965 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110966 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_12
  simp only [← c110961, ← c110962, ← c110963] at e_1817_12
  have e_1817_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_13
  simp only [← c110964, ← c110965, k2, ← c110966] at e_1817_13
  have hr := e_1817_13
  simp only [e_1817_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 63) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6799)) := by
  have c110967 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110968 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110969 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110970 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110971 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110972 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_14
  simp only [← c110967, ← c110968, ← c110969] at e_1817_14
  have e_1817_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_15
  simp only [← c110970, ← c110971, k2, ← c110972] at e_1817_15
  have hr := e_1817_15
  simp only [e_1817_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 71) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6800)) := by
  have c110973 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110974 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110975 := (publicBatchWrapper86_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110976 := (publicBatchWrapper86_copies3468 a h).1
  have c110977 := (publicBatchWrapper86_copies3468 a h).2.1
  have c110978 := (publicBatchWrapper86_copies3468 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_16
  simp only [← c110973, ← c110974, ← c110975] at e_1817_16
  have e_1817_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_17
  simp only [← c110976, ← c110977, k2, ← c110978] at e_1817_17
  have hr := e_1817_17
  simp only [e_1817_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1817 79) = bselect (a (.wire 34 39)) (a (.virt 200)) (a (.virt 6801)) := by
  have c110979 := (publicBatchWrapper86_copies3468 a h).2.2.2.1
  have c110980 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.1
  have c110981 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.1
  have c110982 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.1
  have c110983 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.1
  have c110984 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1817_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_18
  simp only [← c110979, ← c110980, ← c110981] at e_1817_18
  have e_1817_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1817 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_19
  simp only [← c110982, ← c110983, k2, ← c110984] at e_1817_19
  have hr := e_1817_19
  simp only [e_1817_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 7) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6970)) := by
  have c110985 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.1
  have c110986 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110987 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110988 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110989 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110990 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_0
  simp only [← c110985, ← c110986, ← c110987] at e_1818_0
  have e_1818_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_1
  simp only [← c110988, ← c110989, k2, ← c110990] at e_1818_1
  have hr := e_1818_1
  simp only [e_1818_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 15) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6971)) := by
  have c110991 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110992 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110993 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110994 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110995 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110996 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_2
  simp only [← c110991, ← c110992, ← c110993] at e_1818_2
  have e_1818_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_3
  simp only [← c110994, ← c110995, k2, ← c110996] at e_1818_3
  have hr := e_1818_3
  simp only [e_1818_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 23) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6972)) := by
  have c110997 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110998 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110999 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111000 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111001 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111002 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_4
  simp only [← c110997, ← c110998, ← c110999] at e_1818_4
  have e_1818_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_5
  simp only [← c111000, ← c111001, k2, ← c111002] at e_1818_5
  have hr := e_1818_5
  simp only [e_1818_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 31) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6973)) := by
  have c111003 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111004 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111005 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111006 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111007 := (publicBatchWrapper86_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111008 := (publicBatchWrapper86_copies3469 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_6
  simp only [← c111003, ← c111004, ← c111005] at e_1818_6
  have e_1818_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_7
  simp only [← c111006, ← c111007, k2, ← c111008] at e_1818_7
  have hr := e_1818_7
  simp only [e_1818_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 39) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6974)) := by
  have c111009 := (publicBatchWrapper86_copies3469 a h).2.1
  have c111010 := (publicBatchWrapper86_copies3469 a h).2.2.1
  have c111011 := (publicBatchWrapper86_copies3469 a h).2.2.2.1
  have c111012 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.1
  have c111013 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.1
  have c111014 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_8
  simp only [← c111009, ← c111010, ← c111011] at e_1818_8
  have e_1818_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_9
  simp only [← c111012, ← c111013, k2, ← c111014] at e_1818_9
  have hr := e_1818_9
  simp only [e_1818_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 47) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6975)) := by
  have c111015 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.1
  have c111016 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.1
  have c111017 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.1
  have c111018 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111019 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111020 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_10
  simp only [← c111015, ← c111016, ← c111017] at e_1818_10
  have e_1818_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_11
  simp only [← c111018, ← c111019, k2, ← c111020] at e_1818_11
  have hr := e_1818_11
  simp only [e_1818_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 55) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6976)) := by
  have c111021 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111022 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111023 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111024 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111025 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111026 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_12
  simp only [← c111021, ← c111022, ← c111023] at e_1818_12
  have e_1818_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_13
  simp only [← c111024, ← c111025, k2, ← c111026] at e_1818_13
  have hr := e_1818_13
  simp only [e_1818_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 63) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6977)) := by
  have c111027 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111028 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111029 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111030 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111031 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111032 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_14
  simp only [← c111027, ← c111028, ← c111029] at e_1818_14
  have e_1818_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_15
  simp only [← c111030, ← c111031, k2, ← c111032] at e_1818_15
  have hr := e_1818_15
  simp only [e_1818_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1818 71) = bselect (a (.wire 35 3)) (a (.virt 200)) (a (.virt 6978)) := by
  have c111033 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111034 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111035 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111036 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111037 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111038 := (publicBatchWrapper86_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1818_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_16
  simp only [← c111033, ← c111034, ← c111035] at e_1818_16
  have e_1818_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1818 (by decide)
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
  have e_1818_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1818 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_18
  simp only [← c111039, ← c111040, ← c111041] at e_1818_18
  have e_1818_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1818 (by decide)
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
  have e_1819_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_0
  simp only [← c111045, ← c111046, ← c111047] at e_1819_0
  have e_1819_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1819 (by decide)
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
  have e_1819_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_2
  simp only [← c111051, ← c111052, ← c111053] at e_1819_2
  have e_1819_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1819 (by decide)
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
  have e_1819_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_4
  simp only [← c111057, ← c111058, ← c111059] at e_1819_4
  have e_1819_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1819 (by decide)
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
  have e_1819_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_6
  simp only [← c111063, ← c111064, ← c111065] at e_1819_6
  have e_1819_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1819 (by decide)
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
  have e_1819_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_8
  simp only [← c111069, ← c111070, ← c111071] at e_1819_8
  have e_1819_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1819 (by decide)
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
  have e_1819_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_10
  simp only [← c111075, ← c111076, ← c111077] at e_1819_10
  have e_1819_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1819 (by decide)
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
  have e_1819_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_12
  simp only [← c111081, ← c111082, ← c111083] at e_1819_12
  have e_1819_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1819 (by decide)
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
  have e_1819_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_14
  simp only [← c111087, ← c111088, ← c111089] at e_1819_14
  have e_1819_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1819 (by decide)
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
  have e_1819_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_16
  simp only [← c111093, ← c111094, ← c111095] at e_1819_16
  have e_1819_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1819 (by decide)
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
  have e_1819_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1819 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1819_18
  simp only [← c111099, ← c111100, ← c111101] at e_1819_18
  have e_1819_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1819 (by decide)
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
  have e_1820_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_0
  simp only [← c111105, ← c111106, ← c111107] at e_1820_0
  have e_1820_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1820 (by decide)
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
  have e_1820_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_2
  simp only [← c111111, ← c111112, ← c111113] at e_1820_2
  have e_1820_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1820 (by decide)
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
  have e_1820_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_4
  simp only [← c111117, ← c111118, ← c111119] at e_1820_4
  have e_1820_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1820 (by decide)
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
  have e_1820_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_6
  simp only [← c111123, ← c111124, ← c111125] at e_1820_6
  have e_1820_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1820 (by decide)
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
  have e_1820_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_8
  simp only [← c111129, ← c111130, ← c111131] at e_1820_8
  have e_1820_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1820 (by decide)
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
  have e_1820_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_10
  simp only [← c111135, ← c111136, ← c111137] at e_1820_10
  have e_1820_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1820 (by decide)
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
  have e_1820_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_12
  simp only [← c111141, ← c111142, ← c111143] at e_1820_12
  have e_1820_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1820 (by decide)
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
  have e_1820_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_14
  simp only [← c111147, ← c111148, ← c111149] at e_1820_14
  have e_1820_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1820 (by decide)
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
  have e_1820_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_16
  simp only [← c111153, ← c111154, ← c111155] at e_1820_16
  have e_1820_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1820 (by decide)
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
  have e_1820_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1820 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1820_18
  simp only [← c111159, ← c111160, ← c111161] at e_1820_18
  have e_1820_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1820 (by decide)
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
  have e_1821_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_0
  simp only [← c111165, ← c111166, ← c111167] at e_1821_0
  have e_1821_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1821 (by decide)
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
  have e_1821_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_2
  simp only [← c111171, ← c111172, ← c111173] at e_1821_2
  have e_1821_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1821 (by decide)
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
  have e_1821_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_4
  simp only [← c111177, ← c111178, ← c111179] at e_1821_4
  have e_1821_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1821 (by decide)
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
  have e_1821_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_6
  simp only [← c111183, ← c111184, ← c111185] at e_1821_6
  have e_1821_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1821 (by decide)
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
  have e_1821_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_8
  simp only [← c111189, ← c111190, ← c111191] at e_1821_8
  have e_1821_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1821 (by decide)
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
  have e_1821_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_10
  simp only [← c111195, ← c111196, ← c111197] at e_1821_10
  have e_1821_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1821 (by decide)
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
  have e_1821_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_12
  simp only [← c111201, ← c111202, ← c111203] at e_1821_12
  have e_1821_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1821 (by decide)
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
  have e_1821_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_14
  simp only [← c111207, ← c111208, ← c111209] at e_1821_14
  have e_1821_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1821 (by decide)
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
  have e_1821_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_16
  simp only [← c111213, ← c111214, ← c111215] at e_1821_16
  have e_1821_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1821 (by decide)
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
  have e_1821_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1821 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1821_18
  simp only [← c111219, ← c111220, ← c111221] at e_1821_18
  have e_1821_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1821 (by decide)
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
  have e_1822_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_0
  simp only [← c111225, ← c111226, ← c111227] at e_1822_0
  have e_1822_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1822 (by decide)
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
  have e_1822_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_2
  simp only [← c111231, ← c111232, ← c111233] at e_1822_2
  have e_1822_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1822 (by decide)
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
  have e_1822_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_4
  simp only [← c111237, ← c111238, ← c111239] at e_1822_4
  have e_1822_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1822 (by decide)
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
  have e_1822_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_6
  simp only [← c111243, ← c111244, ← c111245] at e_1822_6
  have e_1822_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1822 (by decide)
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
  have e_1822_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_8
  simp only [← c111249, ← c111250, ← c111251] at e_1822_8
  have e_1822_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1822 (by decide)
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
  have e_1822_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_10
  simp only [← c111255, ← c111256, ← c111257] at e_1822_10
  have e_1822_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1822 (by decide)
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
  have e_1822_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_12
  simp only [← c111261, ← c111262, ← c111263] at e_1822_12
  have e_1822_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1822 (by decide)
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
  have e_1822_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_14
  simp only [← c111267, ← c111268, ← c111269] at e_1822_14
  have e_1822_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1822 (by decide)
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
  have e_1822_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_16
  simp only [← c111273, ← c111274, ← c111275] at e_1822_16
  have e_1822_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1822 (by decide)
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
  have e_1822_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1822 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1822_18
  simp only [← c111279, ← c111280, ← c111281] at e_1822_18
  have e_1822_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1822 (by decide)
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
  have e_1823_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_0
  simp only [← c111285, ← c111286, ← c111287] at e_1823_0
  have e_1823_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1823 (by decide)
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
  have e_1823_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_2
  simp only [← c111291, ← c111292, ← c111293] at e_1823_2
  have e_1823_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1823 (by decide)
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
  have e_1823_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_4
  simp only [← c111297, ← c111298, ← c111299] at e_1823_4
  have e_1823_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1823 (by decide)
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
  have e_1823_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_6
  simp only [← c111303, ← c111304, ← c111305] at e_1823_6
  have e_1823_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1823 (by decide)
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
  have e_1823_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_8
  simp only [← c111309, ← c111310, ← c111311] at e_1823_8
  have e_1823_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1823 (by decide)
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
  have e_1823_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_10
  simp only [← c111315, ← c111316, ← c111317] at e_1823_10
  have e_1823_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1823 (by decide)
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
  have e_1823_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_12
  simp only [← c111321, ← c111322, ← c111323] at e_1823_12
  have e_1823_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1823 (by decide)
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
  have e_1823_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_14
  simp only [← c111327, ← c111328, ← c111329] at e_1823_14
  have e_1823_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1823 (by decide)
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
  have e_1823_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_16
  simp only [← c111333, ← c111334, ← c111335] at e_1823_16
  have e_1823_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1823 (by decide)
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
  have e_1823_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1823 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1823_18
  simp only [← c111339, ← c111340, ← c111341] at e_1823_18
  have e_1823_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1823 (by decide)
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
  have e_1824_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_0
  simp only [← c111345, ← c111346, ← c111347] at e_1824_0
  have e_1824_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1824 (by decide)
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
  have e_1824_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_2
  simp only [← c111351, ← c111352, ← c111353] at e_1824_2
  have e_1824_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1824 (by decide)
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
  have e_1824_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_4
  simp only [← c111357, ← c111358, ← c111359] at e_1824_4
  have e_1824_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1824 (by decide)
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
  have e_1824_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_6
  simp only [← c111363, ← c111364, ← c111365] at e_1824_6
  have e_1824_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1824 (by decide)
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
  have e_1824_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_8
  simp only [← c111369, ← c111370, ← c111371] at e_1824_8
  have e_1824_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1824 (by decide)
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
  have e_1824_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_10
  simp only [← c111375, ← c111376, ← c111377] at e_1824_10
  have e_1824_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1824 (by decide)
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
  have e_1824_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_12
  simp only [← c111381, ← c111382, ← c111383] at e_1824_12
  have e_1824_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1824 (by decide)
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
  have e_1824_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_14
  simp only [← c111387, ← c111388, ← c111389] at e_1824_14
  have e_1824_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1824 (by decide)
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
  have e_1824_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_16
  simp only [← c111393, ← c111394, ← c111395] at e_1824_16
  have e_1824_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1824 (by decide)
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
  have e_1824_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1824 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1824_18
  simp only [← c111399, ← c111400, ← c111401] at e_1824_18
  have e_1824_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1824 (by decide)
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
  have e_1825_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_0
  simp only [← c111405, ← c111406, ← c111407] at e_1825_0
  have e_1825_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1825 (by decide)
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
  have e_1825_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_2
  simp only [← c111411, ← c111412, ← c111413] at e_1825_2
  have e_1825_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1825 (by decide)
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
  have e_1825_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_4
  simp only [← c111417, ← c111418, ← c111419] at e_1825_4
  have e_1825_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1825 (by decide)
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
  have e_1825_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_6
  simp only [← c111423, ← c111424, ← c111425] at e_1825_6
  have e_1825_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1825 (by decide)
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
  have e_1825_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_8
  simp only [← c111429, ← c111430, ← c111431] at e_1825_8
  have e_1825_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1825 (by decide)
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
  have e_1825_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_10
  simp only [← c111435, ← c111436, ← c111437] at e_1825_10
  have e_1825_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1825 (by decide)
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
  have e_1825_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_12
  simp only [← c111441, ← c111442, ← c111443] at e_1825_12
  have e_1825_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1825 (by decide)
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
  have e_1825_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_14
  simp only [← c111447, ← c111448, ← c111449] at e_1825_14
  have e_1825_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1825 (by decide)
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
  have e_1825_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_16
  simp only [← c111453, ← c111454, ← c111455] at e_1825_16
  have e_1825_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1825 (by decide)
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
  have e_1825_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1825 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1825_18
  simp only [← c111459, ← c111460, ← c111461] at e_1825_18
  have e_1825_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1825 (by decide)
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
  have e_1826_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_0
  simp only [← c111465, ← c111466, ← c111467] at e_1826_0
  have e_1826_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1826 (by decide)
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
  have e_1826_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_2
  simp only [← c111471, ← c111472, ← c111473] at e_1826_2
  have e_1826_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1826 (by decide)
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
  have e_1826_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_4
  simp only [← c111477, ← c111478, ← c111479] at e_1826_4
  have e_1826_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1826 (by decide)
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
  have e_1826_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_6
  simp only [← c111483, ← c111484, ← c111485] at e_1826_6
  have e_1826_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1826 (by decide)
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
  have e_1826_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_8
  simp only [← c111489, ← c111490, ← c111491] at e_1826_8
  have e_1826_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1826 (by decide)
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
  have e_1826_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_10
  simp only [← c111495, ← c111496, ← c111497] at e_1826_10
  have e_1826_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1826 (by decide)
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
  have e_1826_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_12
  simp only [← c111501, ← c111502, ← c111503] at e_1826_12
  have e_1826_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1826 (by decide)
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
  have e_1826_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_14
  simp only [← c111507, ← c111508, ← c111509] at e_1826_14
  have e_1826_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1826 (by decide)
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
  have e_1826_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_16
  simp only [← c111513, ← c111514, ← c111515] at e_1826_16
  have e_1826_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1826 (by decide)
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
  have e_1826_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1826 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1826_18
  simp only [← c111519, ← c111520, ← c111521] at e_1826_18
  have e_1826_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1826 (by decide)
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
  have e_1827_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_0
  simp only [← c111525, ← c111526, ← c111527] at e_1827_0
  have e_1827_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1827 (by decide)
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
  have e_1827_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_2
  simp only [← c111531, ← c111532, ← c111533] at e_1827_2
  have e_1827_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1827 (by decide)
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
  have e_1827_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_4
  simp only [← c111537, ← c111538, ← c111539] at e_1827_4
  have e_1827_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1827 (by decide)
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
  have e_1827_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_6
  simp only [← c111543, ← c111544, ← c111545] at e_1827_6
  have e_1827_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1827 (by decide)
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
  have e_1827_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_8
  simp only [← c111549, ← c111550, ← c111551] at e_1827_8
  have e_1827_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1827 (by decide)
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
  have e_1827_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_10
  simp only [← c111555, ← c111556, ← c111557] at e_1827_10
  have e_1827_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1827 (by decide)
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
  have e_1827_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_12
  simp only [← c111561, ← c111562, ← c111563] at e_1827_12
  have e_1827_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1827 (by decide)
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
  have e_1827_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_14
  simp only [← c111567, ← c111568, ← c111569] at e_1827_14
  have e_1827_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1827 (by decide)
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
  have e_1827_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_16
  simp only [← c111573, ← c111574, ← c111575] at e_1827_16
  have e_1827_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1827 (by decide)
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
  have e_1827_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1827 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1827_18
  simp only [← c111579, ← c111580, ← c111581] at e_1827_18
  have e_1827_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1827 (by decide)
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
  have e_1828_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_0
  simp only [← c111585, ← c111586, ← c111587] at e_1828_0
  have e_1828_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1828 (by decide)
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
  have e_1828_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_2
  simp only [← c111591, ← c111592, ← c111593] at e_1828_2
  have e_1828_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1828 (by decide)
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
  have e_1828_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_4
  simp only [← c111597, ← c111598, ← c111599] at e_1828_4
  have e_1828_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1828 (by decide)
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
  have e_1828_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_6
  simp only [← c111603, ← c111604, ← c111605] at e_1828_6
  have e_1828_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1828 (by decide)
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
  have e_1828_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_8
  simp only [← c111609, ← c111610, ← c111611] at e_1828_8
  have e_1828_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1828 (by decide)
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
  have e_1828_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_10
  simp only [← c111615, ← c111616, ← c111617] at e_1828_10
  have e_1828_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1828 (by decide)
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
  have e_1828_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_12
  simp only [← c111621, ← c111622, ← c111623] at e_1828_12
  have e_1828_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1828 (by decide)
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
  have e_1828_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_14
  simp only [← c111627, ← c111628, ← c111629] at e_1828_14
  have e_1828_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1828 (by decide)
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
  have e_1828_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_16
  simp only [← c111633, ← c111634, ← c111635] at e_1828_16
  have e_1828_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1828 (by decide)
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
  have e_1828_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1828 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1828_18
  simp only [← c111639, ← c111640, ← c111641] at e_1828_18
  have e_1828_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1828 (by decide)
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
  have e_1829_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_0
  simp only [← c111645, ← c111646, ← c111647] at e_1829_0
  have e_1829_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1829 (by decide)
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
  have e_1829_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_2
  simp only [← c111651, ← c111652, ← c111653] at e_1829_2
  have e_1829_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1829 (by decide)
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
  have e_1829_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_4
  simp only [← c111657, ← c111658, ← c111659] at e_1829_4
  have e_1829_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1829 (by decide)
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
  have e_1829_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_6
  simp only [← c111663, ← c111664, ← c111665] at e_1829_6
  have e_1829_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1829 (by decide)
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
  have e_1829_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_8
  simp only [← c111669, ← c111670, ← c111671] at e_1829_8
  have e_1829_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1829 (by decide)
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
  have e_1829_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_10
  simp only [← c111675, ← c111676, ← c111677] at e_1829_10
  have e_1829_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1829 (by decide)
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
  have e_1829_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_12
  simp only [← c111681, ← c111682, ← c111683] at e_1829_12
  have e_1829_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1829 (by decide)
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
  have e_1829_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_14
  simp only [← c111687, ← c111688, ← c111689] at e_1829_14
  have e_1829_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1829 (by decide)
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
  have e_1829_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_16
  simp only [← c111693, ← c111694, ← c111695] at e_1829_16
  have e_1829_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1829 (by decide)
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
  have e_1829_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1829 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1829_18
  simp only [← c111699, ← c111700, ← c111701] at e_1829_18
  have e_1829_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1829 (by decide)
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
  have e_1830_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_0
  simp only [← c111705, ← c111706, ← c111707] at e_1830_0
  have e_1830_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1830 (by decide)
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
  have e_1830_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_2
  simp only [← c111711, ← c111712, ← c111713] at e_1830_2
  have e_1830_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1830 (by decide)
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
  have e_1830_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_4
  simp only [← c111717, ← c111718, ← c111719] at e_1830_4
  have e_1830_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1830 (by decide)
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
  have e_1830_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_6
  simp only [← c111723, ← c111724, ← c111725] at e_1830_6
  have e_1830_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1830 (by decide)
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
  have e_1830_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_8
  simp only [← c111729, ← c111730, ← c111731] at e_1830_8
  have e_1830_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1830 (by decide)
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
  have e_1830_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_10
  simp only [← c111735, ← c111736, ← c111737] at e_1830_10
  have e_1830_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1830 (by decide)
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
  have e_1830_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_12
  simp only [← c111741, ← c111742, ← c111743] at e_1830_12
  have e_1830_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1830 (by decide)
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
  have e_1830_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_14
  simp only [← c111747, ← c111748, ← c111749] at e_1830_14
  have e_1830_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1830 (by decide)
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
  have e_1830_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_16
  simp only [← c111753, ← c111754, ← c111755] at e_1830_16
  have e_1830_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1830 (by decide)
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
  have e_1830_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1830 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1830_18
  simp only [← c111759, ← c111760, ← c111761] at e_1830_18
  have e_1830_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1830 (by decide)
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
  have e_1831_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_0
  simp only [← c111765, ← c111766, ← c111767] at e_1831_0
  have e_1831_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1831 (by decide)
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
  have e_1831_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_2
  simp only [← c111771, ← c111772, ← c111773] at e_1831_2
  have e_1831_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1831 (by decide)
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
  have e_1831_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_4
  simp only [← c111777, ← c111778, ← c111779] at e_1831_4
  have e_1831_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1831 (by decide)
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
  have e_1831_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_6
  simp only [← c111783, ← c111784, ← c111785] at e_1831_6
  have e_1831_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1831 (by decide)
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
  have e_1831_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_8
  simp only [← c111789, ← c111790, ← c111791] at e_1831_8
  have e_1831_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1831 (by decide)
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
  have e_1831_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_10
  simp only [← c111795, ← c111796, ← c111797] at e_1831_10
  have e_1831_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1831 (by decide)
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
  have e_1831_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_12
  simp only [← c111801, ← c111802, ← c111803] at e_1831_12
  have e_1831_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1831 (by decide)
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
  have e_1831_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_14
  simp only [← c111807, ← c111808, ← c111809] at e_1831_14
  have e_1831_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1831 (by decide)
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
  have e_1831_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_16
  simp only [← c111813, ← c111814, ← c111815] at e_1831_16
  have e_1831_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1831 (by decide)
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
  have e_1831_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1831 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1831_18
  simp only [← c111819, ← c111820, ← c111821] at e_1831_18
  have e_1831_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1831 (by decide)
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
  have e_1832_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_0
  simp only [← c111825, ← c111826, ← c111827] at e_1832_0
  have e_1832_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1832 (by decide)
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
  have e_1832_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_2
  simp only [← c111831, ← c111832, ← c111833] at e_1832_2
  have e_1832_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1832 (by decide)
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
  have e_1832_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_4
  simp only [← c111837, ← c111838, ← c111839] at e_1832_4
  have e_1832_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1832 (by decide)
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
  have e_1832_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_6
  simp only [← c111843, ← c111844, ← c111845] at e_1832_6
  have e_1832_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1832 (by decide)
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
  have e_1832_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_8
  simp only [← c111849, ← c111850, ← c111851] at e_1832_8
  have e_1832_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1832 (by decide)
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
  have e_1832_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_10
  simp only [← c111855, ← c111856, ← c111857] at e_1832_10
  have e_1832_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1832 (by decide)
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
  have e_1832_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_12
  simp only [← c111861, ← c111862, ← c111863] at e_1832_12
  have e_1832_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1832 (by decide)
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
  have e_1832_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_14
  simp only [← c111867, ← c111868, ← c111869] at e_1832_14
  have e_1832_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1832 (by decide)
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
  have e_1832_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_16
  simp only [← c111873, ← c111874, ← c111875] at e_1832_16
  have e_1832_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1832 (by decide)
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
  have e_1832_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1832 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1832_18
  simp only [← c111879, ← c111880, ← c111881] at e_1832_18
  have e_1832_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1832 (by decide)
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
  have e_1833_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_0
  simp only [← c111885, ← c111886, ← c111887] at e_1833_0
  have e_1833_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1833 (by decide)
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
  have e_1833_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_2
  simp only [← c111891, ← c111892, ← c111893] at e_1833_2
  have e_1833_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1833 (by decide)
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
  have e_1833_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_4
  simp only [← c111897, ← c111898, ← c111899] at e_1833_4
  have e_1833_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1833 (by decide)
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
  have e_1833_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_6
  simp only [← c111903, ← c111904, ← c111905] at e_1833_6
  have e_1833_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1833 (by decide)
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
  have e_1833_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_8
  simp only [← c111909, ← c111910, ← c111911] at e_1833_8
  have e_1833_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1833 (by decide)
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
  have e_1833_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_10
  simp only [← c111915, ← c111916, ← c111917] at e_1833_10
  have e_1833_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1833 (by decide)
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
  have e_1833_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_12
  simp only [← c111921, ← c111922, ← c111923] at e_1833_12
  have e_1833_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1833 (by decide)
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
  have e_1833_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_14
  simp only [← c111927, ← c111928, ← c111929] at e_1833_14
  have e_1833_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1833 (by decide)
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
  have e_1833_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_16
  simp only [← c111933, ← c111934, ← c111935] at e_1833_16
  have e_1833_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1833 (by decide)
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
  have e_1833_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1833 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1833_18
  simp only [← c111939, ← c111940, ← c111941] at e_1833_18
  have e_1833_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1833 (by decide)
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
  have e_1834_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_0
  simp only [← c111945, ← c111946, ← c111947] at e_1834_0
  have e_1834_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1834 (by decide)
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
  have e_1834_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_2
  simp only [← c111951, ← c111952, ← c111953] at e_1834_2
  have e_1834_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1834 (by decide)
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
  have e_1834_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_4
  simp only [← c111957, ← c111958, ← c111959] at e_1834_4
  have e_1834_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1834 (by decide)
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
  have e_1834_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_6
  simp only [← c111963, ← c111964, ← c111965] at e_1834_6
  have e_1834_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1834 (by decide)
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
  have e_1834_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_8
  simp only [← c111969, ← c111970, ← c111971] at e_1834_8
  have e_1834_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1834 (by decide)
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
  have e_1834_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_10
  simp only [← c111975, ← c111976, ← c111977] at e_1834_10
  have e_1834_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1834 (by decide)
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
  have e_1834_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_12
  simp only [← c111981, ← c111982, ← c111983] at e_1834_12
  have e_1834_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1834 (by decide)
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
  have e_1834_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_14
  simp only [← c111987, ← c111988, ← c111989] at e_1834_14
  have e_1834_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1834 (by decide)
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
  have e_1834_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_16
  simp only [← c111993, ← c111994, ← c111995] at e_1834_16
  have e_1834_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1834 (by decide)
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
  have e_1834_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1834 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1834_18
  simp only [← c111999, ← c112000, ← c112001] at e_1834_18
  have e_1834_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1834 (by decide)
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
  have e_1835_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_0
  simp only [← c112005, ← c112006, ← c112007] at e_1835_0
  have e_1835_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1835 (by decide)
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
  have e_1835_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_2
  simp only [← c112011, ← c112012, ← c112013] at e_1835_2
  have e_1835_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1835 (by decide)
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
  have e_1835_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_4
  simp only [← c112017, ← c112018, ← c112019] at e_1835_4
  have e_1835_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1835 (by decide)
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
  have e_1835_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_6
  simp only [← c112023, ← c112024, ← c112025] at e_1835_6
  have e_1835_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1835 (by decide)
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
  have e_1835_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_8
  simp only [← c112029, ← c112030, ← c112031] at e_1835_8
  have e_1835_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1835 (by decide)
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
  have e_1835_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_10
  simp only [← c112035, ← c112036, ← c112037] at e_1835_10
  have e_1835_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1835 (by decide)
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
  have e_1835_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_12
  simp only [← c112041, ← c112042, ← c112043] at e_1835_12
  have e_1835_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1835 (by decide)
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
  have e_1835_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_14
  simp only [← c112047, ← c112048, ← c112049] at e_1835_14
  have e_1835_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1835 (by decide)
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
  have e_1835_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_16
  simp only [← c112053, ← c112054, ← c112055] at e_1835_16
  have e_1835_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1835 (by decide)
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
  have e_1835_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1835 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1835_18
  simp only [← c112059, ← c112060, ← c112061] at e_1835_18
  have e_1835_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1835 (by decide)
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
  have e_1836_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_0
  simp only [← c112065, ← c112066, ← c112067] at e_1836_0
  have e_1836_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1836 (by decide)
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
  have e_1836_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_2
  simp only [← c112071, ← c112072, ← c112073] at e_1836_2
  have e_1836_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1836 (by decide)
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
  have e_1836_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_4
  simp only [← c112077, ← c112078, ← c112079] at e_1836_4
  have e_1836_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1836 (by decide)
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
  have e_1836_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_6
  simp only [← c112083, ← c112084, ← c112085] at e_1836_6
  have e_1836_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1836 (by decide)
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
  have e_1836_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_8
  simp only [← c112089, ← c112090, ← c112091] at e_1836_8
  have e_1836_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1836 (by decide)
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
  have e_1836_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_10
  simp only [← c112095, ← c112096, ← c112097] at e_1836_10
  have e_1836_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1836 (by decide)
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
  have e_1836_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_12
  simp only [← c112101, ← c112102, ← c112103] at e_1836_12
  have e_1836_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1836 (by decide)
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
  have e_1836_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_14
  simp only [← c112107, ← c112108, ← c112109] at e_1836_14
  have e_1836_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1836 (by decide)
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
  have e_1836_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_16
  simp only [← c112113, ← c112114, ← c112115] at e_1836_16
  have e_1836_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1836 (by decide)
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
  have e_1836_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1836 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1836_18
  simp only [← c112119, ← c112120, ← c112121] at e_1836_18
  have e_1836_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1836 (by decide)
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
  have e_1837_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_0
  simp only [← c112125, ← c112126, ← c112127] at e_1837_0
  have e_1837_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1837 (by decide)
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
  have e_1837_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_2
  simp only [← c112131, ← c112132, ← c112133] at e_1837_2
  have e_1837_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1837 (by decide)
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
  have e_1837_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_4
  simp only [← c112137, ← c112138, ← c112139] at e_1837_4
  have e_1837_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1837 (by decide)
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
  have e_1837_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_6
  simp only [← c112143, ← c112144, ← c112145] at e_1837_6
  have e_1837_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1837 (by decide)
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
  have e_1837_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_8
  simp only [← c112149, ← c112150, ← c112151] at e_1837_8
  have e_1837_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1837 (by decide)
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
  have e_1837_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_10
  simp only [← c112155, ← c112156, ← c112157] at e_1837_10
  have e_1837_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1837 (by decide)
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
  have e_1837_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_12
  simp only [← c112161, ← c112162, ← c112163] at e_1837_12
  have e_1837_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1837 (by decide)
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
  have e_1837_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_14
  simp only [← c112167, ← c112168, ← c112169] at e_1837_14
  have e_1837_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1837 (by decide)
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
  have e_1837_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_16
  simp only [← c112173, ← c112174, ← c112175] at e_1837_16
  have e_1837_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1837 (by decide)
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
  have e_1837_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1837 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1837_18
  simp only [← c112179, ← c112180, ← c112181] at e_1837_18
  have e_1837_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1837 (by decide)
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
  have e_1838_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_0
  simp only [← c112185, ← c112186, ← c112187] at e_1838_0
  have e_1838_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1838 (by decide)
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
  have e_1838_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_2
  simp only [← c112191, ← c112192, ← c112193] at e_1838_2
  have e_1838_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1838 (by decide)
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
  have e_1838_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_4
  simp only [← c112197, ← c112198, ← c112199] at e_1838_4
  have e_1838_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1838 (by decide)
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
  have e_1838_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_6
  simp only [← c112203, ← c112204, ← c112205] at e_1838_6
  have e_1838_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1838 (by decide)
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
  have e_1838_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_8
  simp only [← c112209, ← c112210, ← c112211] at e_1838_8
  have e_1838_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1838 (by decide)
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
  have e_1838_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_10
  simp only [← c112215, ← c112216, ← c112217] at e_1838_10
  have e_1838_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1838 (by decide)
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
  have e_1838_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_12
  simp only [← c112221, ← c112222, ← c112223] at e_1838_12
  have e_1838_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1838 (by decide)
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
  have e_1838_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_14
  simp only [← c112227, ← c112228, ← c112229] at e_1838_14
  have e_1838_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1838 (by decide)
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
  have e_1838_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_16
  simp only [← c112233, ← c112234, ← c112235] at e_1838_16
  have e_1838_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1838 (by decide)
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
  have e_1838_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1838 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1838_18
  simp only [← c112239, ← c112240, ← c112241] at e_1838_18
  have e_1838_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1838 (by decide)
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
  have e_1839_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_0
  simp only [← c112245, ← c112246, ← c112247] at e_1839_0
  have e_1839_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1839 (by decide)
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
  have e_1839_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_2
  simp only [← c112251, ← c112252, ← c112253] at e_1839_2
  have e_1839_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1839 (by decide)
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
  have e_1839_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_4
  simp only [← c112257, ← c112258, ← c112259] at e_1839_4
  have e_1839_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1839 (by decide)
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
  have e_1839_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_6
  simp only [← c112263, ← c112264, ← c112265] at e_1839_6
  have e_1839_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1839 (by decide)
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
  have e_1839_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_8
  simp only [← c112269, ← c112270, ← c112271] at e_1839_8
  have e_1839_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1839 (by decide)
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
  have e_1839_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_10
  simp only [← c112275, ← c112276, ← c112277] at e_1839_10
  have e_1839_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1839 (by decide)
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
  have e_1839_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_12
  simp only [← c112281, ← c112282, ← c112283] at e_1839_12
  have e_1839_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1839 (by decide)
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
  have e_1839_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_14
  simp only [← c112287, ← c112288, ← c112289] at e_1839_14
  have e_1839_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1839 (by decide)
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
  have e_1839_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_16
  simp only [← c112293, ← c112294, ← c112295] at e_1839_16
  have e_1839_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1839 (by decide)
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
  have e_1839_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1839 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1839_18
  simp only [← c112299, ← c112300, ← c112301] at e_1839_18
  have e_1839_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1839 (by decide)
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
  have e_1840_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_0
  simp only [← c112305, ← c112306, ← c112307] at e_1840_0
  have e_1840_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1840 (by decide)
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
  have e_1840_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_2
  simp only [← c112311, ← c112312, ← c112313] at e_1840_2
  have e_1840_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1840 (by decide)
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
  have e_1840_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_4
  simp only [← c112317, ← c112318, ← c112319] at e_1840_4
  have e_1840_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1840 (by decide)
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
  have e_1840_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_6
  simp only [← c112323, ← c112324, ← c112325] at e_1840_6
  have e_1840_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1840 (by decide)
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
  have e_1840_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_8
  simp only [← c112329, ← c112330, ← c112331] at e_1840_8
  have e_1840_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1840 (by decide)
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
  have e_1840_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_10
  simp only [← c112335, ← c112336, ← c112337] at e_1840_10
  have e_1840_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1840 (by decide)
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
  have e_1840_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_12
  simp only [← c112341, ← c112342, ← c112343] at e_1840_12
  have e_1840_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1840 (by decide)
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
  have e_1840_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_14
  simp only [← c112347, ← c112348, ← c112349] at e_1840_14
  have e_1840_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1840 (by decide)
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
  have e_1840_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_16
  simp only [← c112353, ← c112354, ← c112355] at e_1840_16
  have e_1840_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1840 (by decide)
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
  have e_1840_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1840 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1840_18
  simp only [← c112359, ← c112360, ← c112361] at e_1840_18
  have e_1840_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1840 (by decide)
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
  have e_1841_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_0
  simp only [← c112365, ← c112366, ← c112367] at e_1841_0
  have e_1841_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1841 (by decide)
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
  have e_1841_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_2
  simp only [← c112371, ← c112372, ← c112373] at e_1841_2
  have e_1841_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1841 (by decide)
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
  have e_1841_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_4
  simp only [← c112377, ← c112378, ← c112379] at e_1841_4
  have e_1841_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1841 (by decide)
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
  have e_1841_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_6
  simp only [← c112383, ← c112384, ← c112385] at e_1841_6
  have e_1841_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1841 (by decide)
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
  have e_1841_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_8
  simp only [← c112389, ← c112390, ← c112391] at e_1841_8
  have e_1841_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1841 (by decide)
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
  have e_1841_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_10
  simp only [← c112395, ← c112396, ← c112397] at e_1841_10
  have e_1841_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1841 (by decide)
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
  have e_1841_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_12
  simp only [← c112401, ← c112402, ← c112403] at e_1841_12
  have e_1841_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1841 (by decide)
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
  have e_1841_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_14
  simp only [← c112407, ← c112408, ← c112409] at e_1841_14
  have e_1841_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1841 (by decide)
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
  have e_1841_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_16
  simp only [← c112413, ← c112414, ← c112415] at e_1841_16
  have e_1841_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1841 (by decide)
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
  have e_1841_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1841 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1841_18
  simp only [← c112419, ← c112420, ← c112421] at e_1841_18
  have e_1841_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1841 (by decide)
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
  have e_1842_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_0
  simp only [← c112425, ← c112426, ← c112427] at e_1842_0
  have e_1842_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1842 (by decide)
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
  have e_1842_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_2
  simp only [← c112431, ← c112432, ← c112433] at e_1842_2
  have e_1842_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1842 (by decide)
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
  have e_1842_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_4
  simp only [← c112437, ← c112438, ← c112439] at e_1842_4
  have e_1842_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1842 (by decide)
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
  have e_1842_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_6
  simp only [← c112443, ← c112444, ← c112445] at e_1842_6
  have e_1842_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1842 (by decide)
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
  have e_1842_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_8
  simp only [← c112449, ← c112450, ← c112451] at e_1842_8
  have e_1842_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1842 (by decide)
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
  have e_1842_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_10
  simp only [← c112455, ← c112456, ← c112457] at e_1842_10
  have e_1842_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1842 (by decide)
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
  have e_1842_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_12
  simp only [← c112461, ← c112462, ← c112463] at e_1842_12
  have e_1842_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1842 (by decide)
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
  have e_1842_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_14
  simp only [← c112467, ← c112468, ← c112469] at e_1842_14
  have e_1842_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1842 (by decide)
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
  have e_1842_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_16
  simp only [← c112473, ← c112474, ← c112475] at e_1842_16
  have e_1842_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1842 (by decide)
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
  have e_1842_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1842 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1842_18
  simp only [← c112479, ← c112480, ← c112481] at e_1842_18
  have e_1842_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1842 (by decide)
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
  have e_1843_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_0
  simp only [← c112485, ← c112486, ← c112487] at e_1843_0
  have e_1843_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1843 (by decide)
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
  have e_1843_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_2
  simp only [← c112491, ← c112492, ← c112493] at e_1843_2
  have e_1843_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1843 (by decide)
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
  have e_1843_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_4
  simp only [← c112497, ← c112498, ← c112499] at e_1843_4
  have e_1843_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1843 (by decide)
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
  have e_1843_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_6
  simp only [← c112503, ← c112504, ← c112505] at e_1843_6
  have e_1843_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1843 (by decide)
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
  have e_1843_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_8
  simp only [← c112509, ← c112510, ← c112511] at e_1843_8
  have e_1843_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1843 (by decide)
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
  have e_1843_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_10
  simp only [← c112515, ← c112516, ← c112517] at e_1843_10
  have e_1843_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1843 (by decide)
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
  have e_1843_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_12
  simp only [← c112521, ← c112522, ← c112523] at e_1843_12
  have e_1843_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1843 (by decide)
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
  have e_1843_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_14
  simp only [← c112527, ← c112528, ← c112529] at e_1843_14
  have e_1843_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1843 (by decide)
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
  have e_1843_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_16
  simp only [← c112533, ← c112534, ← c112535] at e_1843_16
  have e_1843_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1843 (by decide)
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
  have e_1843_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1843 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1843_18
  simp only [← c112539, ← c112540, ← c112541] at e_1843_18
  have e_1843_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1843 (by decide)
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
  have e_1844_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_0
  simp only [← c112545, ← c112546, ← c112547] at e_1844_0
  have e_1844_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1844 (by decide)
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
  have e_1844_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_2
  simp only [← c112551, ← c112552, ← c112553] at e_1844_2
  have e_1844_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1844 (by decide)
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
  have e_1844_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_4
  simp only [← c112557, ← c112558, ← c112559] at e_1844_4
  have e_1844_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1844 (by decide)
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
  have e_1844_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_6
  simp only [← c112563, ← c112564, ← c112565] at e_1844_6
  have e_1844_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1844 (by decide)
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
  have e_1844_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_8
  simp only [← c112569, ← c112570, ← c112571] at e_1844_8
  have e_1844_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1844 (by decide)
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
  have e_1844_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_10
  simp only [← c112575, ← c112576, ← c112577] at e_1844_10
  have e_1844_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1844 (by decide)
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
  have e_1844_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_12
  simp only [← c112581, ← c112582, ← c112583] at e_1844_12
  have e_1844_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1844 (by decide)
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
  have e_1844_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_14
  simp only [← c112587, ← c112588, ← c112589] at e_1844_14
  have e_1844_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1844 (by decide)
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
  have e_1844_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_16
  simp only [← c112593, ← c112594, ← c112595] at e_1844_16
  have e_1844_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1844 (by decide)
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
  have e_1844_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1844 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1844_18
  simp only [← c112599, ← c112600, ← c112601] at e_1844_18
  have e_1844_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1844 (by decide)
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
  have e_1845_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_0
  simp only [← c112605, ← c112606, ← c112607] at e_1845_0
  have e_1845_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1845 (by decide)
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
  have e_1845_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_2
  simp only [← c112611, ← c112612, ← c112613] at e_1845_2
  have e_1845_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1845 (by decide)
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
  have e_1845_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_4
  simp only [← c112617, ← c112618, ← c112619] at e_1845_4
  have e_1845_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1845 (by decide)
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
  have e_1845_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_6
  simp only [← c112623, ← c112624, ← c112625] at e_1845_6
  have e_1845_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1845 (by decide)
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
  have e_1845_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_8
  simp only [← c112629, ← c112630, ← c112631] at e_1845_8
  have e_1845_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1845 (by decide)
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
  have e_1845_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_10
  simp only [← c112635, ← c112636, ← c112637] at e_1845_10
  have e_1845_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1845 (by decide)
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
  have e_1845_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_12
  simp only [← c112641, ← c112642, ← c112643] at e_1845_12
  have e_1845_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1845 (by decide)
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
  have e_1845_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_14
  simp only [← c112647, ← c112648, ← c112649] at e_1845_14
  have e_1845_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1845 (by decide)
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
  have e_1845_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_16
  simp only [← c112653, ← c112654, ← c112655] at e_1845_16
  have e_1845_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1845 (by decide)
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
  have e_1845_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1845 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1845_18
  simp only [← c112659, ← c112660, ← c112661] at e_1845_18
  have e_1845_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1845 (by decide)
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
  have e_1846_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_0
  simp only [← c112665, ← c112666, ← c112667] at e_1846_0
  have e_1846_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1846 (by decide)
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
  have e_1846_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_2
  simp only [← c112671, ← c112672, ← c112673] at e_1846_2
  have e_1846_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1846 (by decide)
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
  have e_1846_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_4
  simp only [← c112677, ← c112678, ← c112679] at e_1846_4
  have e_1846_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1846 (by decide)
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
  have e_1846_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_6
  simp only [← c112683, ← c112684, ← c112685] at e_1846_6
  have e_1846_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1846 (by decide)
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
  have e_1846_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_8
  simp only [← c112689, ← c112690, ← c112691] at e_1846_8
  have e_1846_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1846 (by decide)
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
  have e_1846_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_10
  simp only [← c112695, ← c112696, ← c112697] at e_1846_10
  have e_1846_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1846 (by decide)
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
  have e_1846_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_12
  simp only [← c112701, ← c112702, ← c112703] at e_1846_12
  have e_1846_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1846 (by decide)
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
  have e_1846_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_14
  simp only [← c112707, ← c112708, ← c112709] at e_1846_14
  have e_1846_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1846 (by decide)
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
  have e_1846_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_16
  simp only [← c112713, ← c112714, ← c112715] at e_1846_16
  have e_1846_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1846 (by decide)
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
  have e_1846_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1846 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1846_18
  simp only [← c112719, ← c112720, ← c112721] at e_1846_18
  have e_1846_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1846 (by decide)
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
  have e_1847_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_0
  simp only [← c112725, ← c112726, ← c112727] at e_1847_0
  have e_1847_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1847 (by decide)
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
  have e_1847_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_2
  simp only [← c112731, ← c112732, ← c112733] at e_1847_2
  have e_1847_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1847 (by decide)
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
  have e_1847_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_4
  simp only [← c112737, ← c112738, ← c112739] at e_1847_4
  have e_1847_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1847 (by decide)
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
  have e_1847_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_6
  simp only [← c112743, ← c112744, ← c112745] at e_1847_6
  have e_1847_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1847 (by decide)
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
  have e_1847_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_8
  simp only [← c112749, ← c112750, ← c112751] at e_1847_8
  have e_1847_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1847 (by decide)
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
  have e_1847_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_10
  simp only [← c112755, ← c112756, ← c112757] at e_1847_10
  have e_1847_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1847 (by decide)
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
  have e_1847_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_12
  simp only [← c112761, ← c112762, ← c112763] at e_1847_12
  have e_1847_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1847 (by decide)
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
  have e_1847_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_14
  simp only [← c112767, ← c112768, ← c112769] at e_1847_14
  have e_1847_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1847 (by decide)
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
  have e_1847_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_16
  simp only [← c112773, ← c112774, ← c112775] at e_1847_16
  have e_1847_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1847 (by decide)
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
  have e_1847_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1847 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1847_18
  simp only [← c112779, ← c112780, ← c112781] at e_1847_18
  have e_1847_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1847 (by decide)
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
  have e_1848_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_0
  simp only [← c112785, ← c112786, ← c112787] at e_1848_0
  have e_1848_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1848 (by decide)
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
  have e_1848_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_2
  simp only [← c112791, ← c112792, ← c112793] at e_1848_2
  have e_1848_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1848 (by decide)
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
  have e_1848_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_4
  simp only [← c112797, ← c112798, ← c112799] at e_1848_4
  have e_1848_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1848 (by decide)
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
  have e_1848_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_6
  simp only [← c112803, ← c112804, ← c112805] at e_1848_6
  have e_1848_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1848 (by decide)
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
  have e_1848_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_8
  simp only [← c112809, ← c112810, ← c112811] at e_1848_8
  have e_1848_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1848 (by decide)
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
  have e_1848_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_10
  simp only [← c112815, ← c112816, ← c112817] at e_1848_10
  have e_1848_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1848 (by decide)
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
  have e_1848_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_12
  simp only [← c112821, ← c112822, ← c112823] at e_1848_12
  have e_1848_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1848 (by decide)
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
  have e_1848_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_14
  simp only [← c112827, ← c112828, ← c112829] at e_1848_14
  have e_1848_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1848 (by decide)
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
  have e_1848_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_16
  simp only [← c112833, ← c112834, ← c112835] at e_1848_16
  have e_1848_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1848 (by decide)
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
  have e_1848_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1848 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1848_18
  simp only [← c112839, ← c112840, ← c112841] at e_1848_18
  have e_1848_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1848 (by decide)
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
  have e_1849_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_0
  simp only [← c112845, ← c112846, ← c112847] at e_1849_0
  have e_1849_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1849 (by decide)
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
  have e_1849_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_2
  simp only [← c112851, ← c112852, ← c112853] at e_1849_2
  have e_1849_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1849 (by decide)
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
  have e_1849_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_4
  simp only [← c112857, ← c112858, ← c112859] at e_1849_4
  have e_1849_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1849 (by decide)
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
  have e_1849_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_6
  simp only [← c112863, ← c112864, ← c112865] at e_1849_6
  have e_1849_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1849 (by decide)
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
  have e_1849_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_8
  simp only [← c112869, ← c112870, ← c112871] at e_1849_8
  have e_1849_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1849 (by decide)
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
  have e_1849_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_10
  simp only [← c112875, ← c112876, ← c112877] at e_1849_10
  have e_1849_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1849 (by decide)
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
  have e_1849_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_12
  simp only [← c112881, ← c112882, ← c112883] at e_1849_12
  have e_1849_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1849 (by decide)
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
  have e_1849_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_14
  simp only [← c112887, ← c112888, ← c112889] at e_1849_14
  have e_1849_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1849 (by decide)
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
  have e_1849_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_16
  simp only [← c112893, ← c112894, ← c112895] at e_1849_16
  have e_1849_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1849 (by decide)
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
  have e_1849_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1849 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1849_18
  simp only [← c112899, ← c112900, ← c112901] at e_1849_18
  have e_1849_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1849 (by decide)
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
  have e_1850_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_0
  simp only [← c112905, ← c112906, ← c112907] at e_1850_0
  have e_1850_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1850 (by decide)
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
  have e_1850_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_2
  simp only [← c112911, ← c112912, ← c112913] at e_1850_2
  have e_1850_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1850 (by decide)
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
  have e_1850_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_4
  simp only [← c112917, ← c112918, ← c112919] at e_1850_4
  have e_1850_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1850 (by decide)
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
  have e_1850_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_6
  simp only [← c112923, ← c112924, ← c112925] at e_1850_6
  have e_1850_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1850 (by decide)
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
  have e_1850_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_8
  simp only [← c112929, ← c112930, ← c112931] at e_1850_8
  have e_1850_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1850 (by decide)
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
  have e_1850_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_10
  simp only [← c112935, ← c112936, ← c112937] at e_1850_10
  have e_1850_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1850 (by decide)
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
  have e_1850_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_12
  simp only [← c112941, ← c112942, ← c112943] at e_1850_12
  have e_1850_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1850 (by decide)
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
  have e_1850_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_14
  simp only [← c112947, ← c112948, ← c112949] at e_1850_14
  have e_1850_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1850 (by decide)
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
  have e_1850_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_16
  simp only [← c112953, ← c112954, ← c112955] at e_1850_16
  have e_1850_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1850 (by decide)
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
  have e_1850_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1850 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1850_18
  simp only [← c112959, ← c112960, ← c112961] at e_1850_18
  have e_1850_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1850 (by decide)
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
  have e_1851_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_0
  simp only [← c112965, ← c112966, ← c112967] at e_1851_0
  have e_1851_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1851 (by decide)
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
  have e_1851_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_2
  simp only [← c112971, ← c112972, ← c112973] at e_1851_2
  have e_1851_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1851 (by decide)
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
  have e_1851_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_4
  simp only [← c112977, ← c112978, ← c112979] at e_1851_4
  have e_1851_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1851 (by decide)
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
  have e_1851_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_6
  simp only [← c112983, ← c112984, ← c112985] at e_1851_6
  have e_1851_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1851 (by decide)
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
  have e_1851_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_8
  simp only [← c112989, ← c112990, ← c112991] at e_1851_8
  have e_1851_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1851 (by decide)
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
  have e_1851_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_10
  simp only [← c112995, ← c112996, ← c112997] at e_1851_10
  have e_1851_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1851 (by decide)
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
  have e_1851_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_12
  simp only [← c113001, ← c113002, ← c113003] at e_1851_12
  have e_1851_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1851 (by decide)
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
  have e_1851_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_14
  simp only [← c113007, ← c113008, ← c113009] at e_1851_14
  have e_1851_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1851 (by decide)
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
  have e_1851_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_16
  simp only [← c113013, ← c113014, ← c113015] at e_1851_16
  have e_1851_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1851 (by decide)
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
  have e_1851_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1851 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1851_18
  simp only [← c113019, ← c113020, ← c113021] at e_1851_18
  have e_1851_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1851 (by decide)
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
  have e_1852_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_0
  simp only [← c113025, ← c113026, ← c113027] at e_1852_0
  have e_1852_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1852 (by decide)
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
  have e_1852_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_2
  simp only [← c113031, ← c113032, ← c113033] at e_1852_2
  have e_1852_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1852 (by decide)
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
  have e_1852_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_4
  simp only [← c113037, ← c113038, ← c113039] at e_1852_4
  have e_1852_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1852 (by decide)
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
  have e_1852_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_6
  simp only [← c113043, ← c113044, ← c113045] at e_1852_6
  have e_1852_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1852 (by decide)
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
  have e_1852_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_8
  simp only [← c113049, ← c113050, ← c113051] at e_1852_8
  have e_1852_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1852 (by decide)
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
  have e_1852_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_10
  simp only [← c113055, ← c113056, ← c113057] at e_1852_10
  have e_1852_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1852 (by decide)
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
  have e_1852_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_12
  simp only [← c113061, ← c113062, ← c113063] at e_1852_12
  have e_1852_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1852 (by decide)
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
  have e_1852_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_14
  simp only [← c113067, ← c113068, ← c113069] at e_1852_14
  have e_1852_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1852 (by decide)
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
  have e_1852_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_16
  simp only [← c113073, ← c113074, ← c113075] at e_1852_16
  have e_1852_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1852 (by decide)
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
  have e_1852_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1852 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1852_18
  simp only [← c113079, ← c113080, ← c113081] at e_1852_18
  have e_1852_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1852 (by decide)
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
  have e_1853_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_0
  simp only [← c113085, ← c113086, ← c113087] at e_1853_0
  have e_1853_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1853 (by decide)
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
  have e_1853_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_2
  simp only [← c113091, ← c113092, ← c113093] at e_1853_2
  have e_1853_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1853 (by decide)
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
  have e_1853_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_4
  simp only [← c113097, ← c113098, ← c113099] at e_1853_4
  have e_1853_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1853 (by decide)
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
  have e_1853_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_6
  simp only [← c113103, ← c113104, ← c113105] at e_1853_6
  have e_1853_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1853 (by decide)
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
  have e_1853_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_8
  simp only [← c113109, ← c113110, ← c113111] at e_1853_8
  have e_1853_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1853 (by decide)
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
  have e_1853_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_10
  simp only [← c113115, ← c113116, ← c113117] at e_1853_10
  have e_1853_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1853 (by decide)
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
  have e_1853_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_12
  simp only [← c113121, ← c113122, ← c113123] at e_1853_12
  have e_1853_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1853 (by decide)
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
  have e_1853_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_14
  simp only [← c113127, ← c113128, ← c113129] at e_1853_14
  have e_1853_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1853 (by decide)
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
  have e_1853_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_16
  simp only [← c113133, ← c113134, ← c113135] at e_1853_16
  have e_1853_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1853 (by decide)
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
  have e_1853_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1853 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1853_18
  simp only [← c113139, ← c113140, ← c113141] at e_1853_18
  have e_1853_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1853 (by decide)
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
  have e_1854_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_0
  simp only [← c113145, ← c113146, ← c113147] at e_1854_0
  have e_1854_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1854 (by decide)
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
  have e_1854_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_2
  simp only [← c113151, ← c113152, ← c113153] at e_1854_2
  have e_1854_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1854 (by decide)
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
  have e_1854_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_4
  simp only [← c113157, ← c113158, ← c113159] at e_1854_4
  have e_1854_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1854 (by decide)
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
  have e_1854_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_6
  simp only [← c113163, ← c113164, ← c113165] at e_1854_6
  have e_1854_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1854 (by decide)
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
  have e_1854_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_8
  simp only [← c113169, ← c113170, ← c113171] at e_1854_8
  have e_1854_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1854 (by decide)
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
  have e_1854_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_10
  simp only [← c113175, ← c113176, ← c113177] at e_1854_10
  have e_1854_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1854 (by decide)
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
  have e_1854_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_12
  simp only [← c113181, ← c113182, ← c113183] at e_1854_12
  have e_1854_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1854 (by decide)
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
  have e_1854_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_14
  simp only [← c113187, ← c113188, ← c113189] at e_1854_14
  have e_1854_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1854 (by decide)
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
  have e_1854_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_16
  simp only [← c113193, ← c113194, ← c113195] at e_1854_16
  have e_1854_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1854 (by decide)
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
  have e_1854_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1854 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1854_18
  simp only [← c113199, ← c113200, ← c113201] at e_1854_18
  have e_1854_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1854 (by decide)
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
  have e_1855_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_0
  simp only [← c113205, ← c113206, ← c113207] at e_1855_0
  have e_1855_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1855 (by decide)
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
  have e_1855_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_2
  simp only [← c113211, ← c113212, ← c113213] at e_1855_2
  have e_1855_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1855 (by decide)
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
  have e_1855_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_4
  simp only [← c113217, ← c113218, ← c113219] at e_1855_4
  have e_1855_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1855 (by decide)
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
  have e_1855_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_6
  simp only [← c113223, ← c113224, ← c113225] at e_1855_6
  have e_1855_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1855 (by decide)
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
  have e_1855_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_8
  simp only [← c113229, ← c113230, ← c113231] at e_1855_8
  have e_1855_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1855 (by decide)
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
  have e_1855_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_10
  simp only [← c113235, ← c113236, ← c113237] at e_1855_10
  have e_1855_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1855 (by decide)
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
  have e_1855_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_12
  simp only [← c113241, ← c113242, ← c113243] at e_1855_12
  have e_1855_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1855 (by decide)
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
  have e_1855_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_14
  simp only [← c113247, ← c113248, ← c113249] at e_1855_14
  have e_1855_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1855 (by decide)
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
  have e_1855_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_16
  simp only [← c113253, ← c113254, ← c113255] at e_1855_16
  have e_1855_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1855 (by decide)
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
  have e_1855_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1855 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1855_18
  simp only [← c113259, ← c113260, ← c113261] at e_1855_18
  have e_1855_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1855 (by decide)
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
  have e_1856_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_0
  simp only [← c113265, ← c113266, ← c113267] at e_1856_0
  have e_1856_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1856 (by decide)
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
  have e_1856_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_2
  simp only [← c113271, ← c113272, ← c113273] at e_1856_2
  have e_1856_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1856 (by decide)
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
  have e_1856_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_4
  simp only [← c113277, ← c113278, ← c113279] at e_1856_4
  have e_1856_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1856 (by decide)
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
  have e_1856_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_6
  simp only [← c113283, ← c113284, ← c113285] at e_1856_6
  have e_1856_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1856 (by decide)
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
  have e_1856_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_8
  simp only [← c113289, ← c113290, ← c113291] at e_1856_8
  have e_1856_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1856 (by decide)
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
  have e_1856_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_10
  simp only [← c113295, ← c113296, ← c113297] at e_1856_10
  have e_1856_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1856 (by decide)
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
  have e_1856_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_12
  simp only [← c113301, ← c113302, ← c113303] at e_1856_12
  have e_1856_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1856 (by decide)
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
  have e_1856_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_14
  simp only [← c113307, ← c113308, ← c113309] at e_1856_14
  have e_1856_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1856 (by decide)
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
  have e_1856_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_16
  simp only [← c113313, ← c113314, ← c113315] at e_1856_16
  have e_1856_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1856 (by decide)
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
  have e_1856_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1856 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1856_18
  simp only [← c113319, ← c113320, ← c113321] at e_1856_18
  have e_1856_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1856 (by decide)
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
  have e_1857_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_0
  simp only [← c113325, ← c113326, ← c113327] at e_1857_0
  have e_1857_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1857 (by decide)
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
  have e_1857_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_2
  simp only [← c113331, ← c113332, ← c113333] at e_1857_2
  have e_1857_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1857 (by decide)
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
  have e_1857_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_4
  simp only [← c113337, ← c113338, ← c113339] at e_1857_4
  have e_1857_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1857 (by decide)
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
  have e_1857_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_6
  simp only [← c113343, ← c113344, ← c113345] at e_1857_6
  have e_1857_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1857 (by decide)
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
  have e_1857_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_8
  simp only [← c113349, ← c113350, ← c113351] at e_1857_8
  have e_1857_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1857 (by decide)
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
  have e_1857_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_10
  simp only [← c113355, ← c113356, ← c113357] at e_1857_10
  have e_1857_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1857 (by decide)
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
  have e_1857_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_12
  simp only [← c113361, ← c113362, ← c113363] at e_1857_12
  have e_1857_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1857 (by decide)
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
  have e_1857_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_14
  simp only [← c113367, ← c113368, ← c113369] at e_1857_14
  have e_1857_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1857 (by decide)
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
  have e_1857_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_16
  simp only [← c113373, ← c113374, ← c113375] at e_1857_16
  have e_1857_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1857 (by decide)
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
  have e_1857_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1857 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1857_18
  simp only [← c113379, ← c113380, ← c113381] at e_1857_18
  have e_1857_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1857 (by decide)
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
  have e_1858_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_0
  simp only [← c113385, ← c113386, ← c113387] at e_1858_0
  have e_1858_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1858 (by decide)
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
  have e_1858_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_2
  simp only [← c113391, ← c113392, ← c113393] at e_1858_2
  have e_1858_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1858 (by decide)
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
  have e_1858_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_4
  simp only [← c113397, ← c113398, ← c113399] at e_1858_4
  have e_1858_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1858 (by decide)
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
  have e_1858_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_6
  simp only [← c113403, ← c113404, ← c113405] at e_1858_6
  have e_1858_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1858 (by decide)
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
  have e_1858_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_8
  simp only [← c113409, ← c113410, ← c113411] at e_1858_8
  have e_1858_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1858 (by decide)
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
  have e_1858_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_10
  simp only [← c113415, ← c113416, ← c113417] at e_1858_10
  have e_1858_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1858 (by decide)
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
  have e_1858_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_12
  simp only [← c113421, ← c113422, ← c113423] at e_1858_12
  have e_1858_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1858 (by decide)
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
  have e_1858_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_14
  simp only [← c113427, ← c113428, ← c113429] at e_1858_14
  have e_1858_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1858 (by decide)
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
  have e_1858_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_16
  simp only [← c113433, ← c113434, ← c113435] at e_1858_16
  have e_1858_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1858 (by decide)
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
  have e_1858_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1858 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1858_18
  simp only [← c113439, ← c113440, ← c113441] at e_1858_18
  have e_1858_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1858 (by decide)
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
  have e_1859_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_0
  simp only [← c113445, ← c113446, ← c113447] at e_1859_0
  have e_1859_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1859 (by decide)
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
  have e_1859_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_2
  simp only [← c113451, ← c113452, ← c113453] at e_1859_2
  have e_1859_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1859 (by decide)
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
  have e_1859_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_4
  simp only [← c113457, ← c113458, ← c113459] at e_1859_4
  have e_1859_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1859 (by decide)
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
  have e_1859_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_6
  simp only [← c113463, ← c113464, ← c113465] at e_1859_6
  have e_1859_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1859 (by decide)
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
  have e_1859_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_8
  simp only [← c113469, ← c113470, ← c113471] at e_1859_8
  have e_1859_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1859 (by decide)
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
  have e_1859_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_10
  simp only [← c113475, ← c113476, ← c113477] at e_1859_10
  have e_1859_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1859 (by decide)
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
  have e_1859_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_12
  simp only [← c113481, ← c113482, ← c113483] at e_1859_12
  have e_1859_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1859 (by decide)
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
  have e_1859_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_14
  simp only [← c113487, ← c113488, ← c113489] at e_1859_14
  have e_1859_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1859 (by decide)
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
  have e_1859_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_16
  simp only [← c113493, ← c113494, ← c113495] at e_1859_16
  have e_1859_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1859 (by decide)
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
  have e_1859_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1859 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1859_18
  simp only [← c113499, ← c113500, ← c113501] at e_1859_18
  have e_1859_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1859 (by decide)
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
  have e_1860_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_0
  simp only [← c113505, ← c113506, ← c113507] at e_1860_0
  have e_1860_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1860 (by decide)
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
  have e_1860_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_2
  simp only [← c113511, ← c113512, ← c113513] at e_1860_2
  have e_1860_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1860 (by decide)
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
  have e_1860_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_4
  simp only [← c113517, ← c113518, ← c113519] at e_1860_4
  have e_1860_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1860 (by decide)
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
  have e_1860_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_6
  simp only [← c113523, ← c113524, ← c113525] at e_1860_6
  have e_1860_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1860 (by decide)
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
  have e_1860_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_8
  simp only [← c113529, ← c113530, ← c113531] at e_1860_8
  have e_1860_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1860 (by decide)
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
  have e_1860_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_10
  simp only [← c113535, ← c113536, ← c113537] at e_1860_10
  have e_1860_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1860 (by decide)
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
  have e_1860_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_12
  simp only [← c113541, ← c113542, ← c113543] at e_1860_12
  have e_1860_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1860 (by decide)
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
  have e_1860_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_14
  simp only [← c113547, ← c113548, ← c113549] at e_1860_14
  have e_1860_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1860 (by decide)
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
  have e_1860_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_16
  simp only [← c113553, ← c113554, ← c113555] at e_1860_16
  have e_1860_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1860 (by decide)
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
  have e_1860_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1860 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1860_18
  simp only [← c113559, ← c113560, ← c113561] at e_1860_18
  have e_1860_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1860 (by decide)
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
  have e_1861_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_0
  simp only [← c113565, ← c113566, ← c113567] at e_1861_0
  have e_1861_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1861 (by decide)
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
  have e_1861_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_2
  simp only [← c113571, ← c113572, ← c113573] at e_1861_2
  have e_1861_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1861 (by decide)
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
  have e_1861_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_4
  simp only [← c113577, ← c113578, ← c113579] at e_1861_4
  have e_1861_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1861 (by decide)
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
  have e_1861_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_6
  simp only [← c113583, ← c113584, ← c113585] at e_1861_6
  have e_1861_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1861 (by decide)
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
  have e_1861_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_8
  simp only [← c113589, ← c113590, ← c113591] at e_1861_8
  have e_1861_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1861 (by decide)
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
  have e_1861_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_10
  simp only [← c113595, ← c113596, ← c113597] at e_1861_10
  have e_1861_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1861 (by decide)
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
  have e_1861_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_12
  simp only [← c113601, ← c113602, ← c113603] at e_1861_12
  have e_1861_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1861 (by decide)
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
  have e_1861_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_14
  simp only [← c113607, ← c113608, ← c113609] at e_1861_14
  have e_1861_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1861 (by decide)
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
  have e_1861_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_16
  simp only [← c113613, ← c113614, ← c113615] at e_1861_16
  have e_1861_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1861 (by decide)
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
  have e_1861_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1861 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1861_18
  simp only [← c113619, ← c113620, ← c113621] at e_1861_18
  have e_1861_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1861 (by decide)
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
  have e_1862_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_0
  simp only [← c113625, ← c113626, ← c113627] at e_1862_0
  have e_1862_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1862 (by decide)
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
  have e_1862_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_2
  simp only [← c113631, ← c113632, ← c113633] at e_1862_2
  have e_1862_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1862 (by decide)
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
  have e_1862_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_4
  simp only [← c113637, ← c113638, ← c113639] at e_1862_4
  have e_1862_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1862 (by decide)
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
  have e_1862_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_6
  simp only [← c113643, ← c113644, ← c113645] at e_1862_6
  have e_1862_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1862 (by decide)
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
  have e_1862_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_8
  simp only [← c113649, ← c113650, ← c113651] at e_1862_8
  have e_1862_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1862 (by decide)
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
  have e_1862_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_10
  simp only [← c113655, ← c113656, ← c113657] at e_1862_10
  have e_1862_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1862 (by decide)
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
  have e_1862_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_12
  simp only [← c113661, ← c113662, ← c113663] at e_1862_12
  have e_1862_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1862 (by decide)
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
  have e_1862_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_14
  simp only [← c113667, ← c113668, ← c113669] at e_1862_14
  have e_1862_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1862 (by decide)
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
  have e_1862_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_16
  simp only [← c113673, ← c113674, ← c113675] at e_1862_16
  have e_1862_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1862 (by decide)
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
  have e_1862_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1862 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1862_18
  simp only [← c113679, ← c113680, ← c113681] at e_1862_18
  have e_1862_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1862 (by decide)
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
  have e_1863_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_0
  simp only [← c113685, ← c113686, ← c113687] at e_1863_0
  have e_1863_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1863 (by decide)
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
  have e_1863_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_2
  simp only [← c113691, ← c113692, ← c113693] at e_1863_2
  have e_1863_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1863 (by decide)
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
  have e_1863_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_4
  simp only [← c113697, ← c113698, ← c113699] at e_1863_4
  have e_1863_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1863 (by decide)
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
  have e_1863_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_6
  simp only [← c113703, ← c113704, ← c113705] at e_1863_6
  have e_1863_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1863 (by decide)
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
  have e_1863_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_8
  simp only [← c113709, ← c113710, ← c113711] at e_1863_8
  have e_1863_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1863 (by decide)
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
  have e_1863_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_10
  simp only [← c113715, ← c113716, ← c113717] at e_1863_10
  have e_1863_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1863 (by decide)
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
  have e_1863_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_12
  simp only [← c113721, ← c113722, ← c113723] at e_1863_12
  have e_1863_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1863 (by decide)
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
  have e_1863_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_14
  simp only [← c113727, ← c113728, ← c113729] at e_1863_14
  have e_1863_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1863 (by decide)
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
  have e_1863_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_16
  simp only [← c113733, ← c113734, ← c113735] at e_1863_16
  have e_1863_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1863 (by decide)
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
  have e_1863_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1863 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1863_18
  simp only [← c113739, ← c113740, ← c113741] at e_1863_18
  have e_1863_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1863 (by decide)
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
  have e_1864_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_0
  simp only [← c113745, ← c113746, ← c113747] at e_1864_0
  have e_1864_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1864 (by decide)
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
  have e_1864_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_2
  simp only [← c113751, ← c113752, ← c113753] at e_1864_2
  have e_1864_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1864 (by decide)
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
  have e_1864_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_4
  simp only [← c113757, ← c113758, ← c113759] at e_1864_4
  have e_1864_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1864 (by decide)
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
  have e_1864_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_6
  simp only [← c113763, ← c113764, ← c113765] at e_1864_6
  have e_1864_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1864 (by decide)
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
  have e_1864_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_8
  simp only [← c113769, ← c113770, ← c113771] at e_1864_8
  have e_1864_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1864 (by decide)
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
  have e_1864_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_10
  simp only [← c113775, ← c113776, ← c113777] at e_1864_10
  have e_1864_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1864 (by decide)
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
  have e_1864_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_12
  simp only [← c113781, ← c113782, ← c113783] at e_1864_12
  have e_1864_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1864 (by decide)
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
  have e_1864_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_14
  simp only [← c113787, ← c113788, ← c113789] at e_1864_14
  have e_1864_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1864 (by decide)
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
  have e_1864_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_16
  simp only [← c113793, ← c113794, ← c113795] at e_1864_16
  have e_1864_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1864 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1864_17
  simp only [← c113796, ← c113797, k2, ← c113798] at e_1864_17
  have hr := e_1864_17
  simp only [e_1864_16] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
