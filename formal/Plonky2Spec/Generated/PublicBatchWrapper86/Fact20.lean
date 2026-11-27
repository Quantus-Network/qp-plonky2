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

theorem publicBatchWrapper86_f15184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15310)) := by
  have c95457 := (publicBatchWrapper86_copies2983 a h).2.1
  have c95458 := (publicBatchWrapper86_copies2983 a h).2.2.1
  have c95459 := (publicBatchWrapper86_copies2983 a h).2.2.2.1
  have c95460 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.1
  have c95461 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.1
  have c95462 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_4
  simp only [← c95457, ← c95458, ← c95459] at e_1559_4
  have e_1559_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_5
  simp only [← c95460, ← c95461, k2, ← c95462] at e_1559_5
  have hr := e_1559_5
  simp only [e_1559_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15311)) := by
  have c95463 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.1
  have c95464 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.1
  have c95465 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.1
  have c95466 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95467 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95468 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_6
  simp only [← c95463, ← c95464, ← c95465] at e_1559_6
  have e_1559_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_7
  simp only [← c95466, ← c95467, k2, ← c95468] at e_1559_7
  have hr := e_1559_7
  simp only [e_1559_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15312)) := by
  have c95469 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95470 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95471 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95472 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95473 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95474 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_8
  simp only [← c95469, ← c95470, ← c95471] at e_1559_8
  have e_1559_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_9
  simp only [← c95472, ← c95473, k2, ← c95474] at e_1559_9
  have hr := e_1559_9
  simp only [e_1559_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15313)) := by
  have c95475 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95476 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95477 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95478 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95479 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95480 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_10
  simp only [← c95475, ← c95476, ← c95477] at e_1559_10
  have e_1559_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_11
  simp only [← c95478, ← c95479, k2, ← c95480] at e_1559_11
  have hr := e_1559_11
  simp only [e_1559_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15314)) := by
  have c95481 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95482 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95483 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95484 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95485 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95486 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_12
  simp only [← c95481, ← c95482, ← c95483] at e_1559_12
  have e_1559_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_13
  simp only [← c95484, ← c95485, k2, ← c95486] at e_1559_13
  have hr := e_1559_13
  simp only [e_1559_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15315)) := by
  have c95487 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95488 := (publicBatchWrapper86_copies2984 a h).1
  have c95489 := (publicBatchWrapper86_copies2984 a h).2.1
  have c95490 := (publicBatchWrapper86_copies2984 a h).2.2.1
  have c95491 := (publicBatchWrapper86_copies2984 a h).2.2.2.1
  have c95492 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_14
  simp only [← c95487, ← c95488, ← c95489] at e_1559_14
  have e_1559_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_15
  simp only [← c95490, ← c95491, k2, ← c95492] at e_1559_15
  have hr := e_1559_15
  simp only [e_1559_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15316)) := by
  have c95493 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.1
  have c95494 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.1
  have c95495 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.1
  have c95496 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.1
  have c95497 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.1
  have c95498 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_16
  simp only [← c95493, ← c95494, ← c95495] at e_1559_16
  have e_1559_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_17
  simp only [← c95496, ← c95497, k2, ← c95498] at e_1559_17
  have hr := e_1559_17
  simp only [e_1559_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15317)) := by
  have c95499 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95500 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95501 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95502 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95503 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95504 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_18
  simp only [← c95499, ← c95500, ← c95501] at e_1559_18
  have e_1559_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_19
  simp only [← c95502, ← c95503, k2, ← c95504] at e_1559_19
  have hr := e_1559_19
  simp only [e_1559_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15318)) := by
  have c95505 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95506 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95507 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95508 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95509 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95510 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_0
  simp only [← c95505, ← c95506, ← c95507] at e_1560_0
  have e_1560_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_1
  simp only [← c95508, ← c95509, k2, ← c95510] at e_1560_1
  have hr := e_1560_1
  simp only [e_1560_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15319)) := by
  have c95511 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95512 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95513 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95514 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95515 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95516 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_2
  simp only [← c95511, ← c95512, ← c95513] at e_1560_2
  have e_1560_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_3
  simp only [← c95514, ← c95515, k2, ← c95516] at e_1560_3
  have hr := e_1560_3
  simp only [e_1560_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15320)) := by
  have c95517 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95518 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95519 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95520 := (publicBatchWrapper86_copies2985 a h).1
  have c95521 := (publicBatchWrapper86_copies2985 a h).2.1
  have c95522 := (publicBatchWrapper86_copies2985 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_4
  simp only [← c95517, ← c95518, ← c95519] at e_1560_4
  have e_1560_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_5
  simp only [← c95520, ← c95521, k2, ← c95522] at e_1560_5
  have hr := e_1560_5
  simp only [e_1560_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15321)) := by
  have c95523 := (publicBatchWrapper86_copies2985 a h).2.2.2.1
  have c95524 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.1
  have c95525 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.1
  have c95526 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.1
  have c95527 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.1
  have c95528 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_6
  simp only [← c95523, ← c95524, ← c95525] at e_1560_6
  have e_1560_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_7
  simp only [← c95526, ← c95527, k2, ← c95528] at e_1560_7
  have hr := e_1560_7
  simp only [e_1560_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15322)) := by
  have c95529 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.1
  have c95530 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95531 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95532 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95533 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95534 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_8
  simp only [← c95529, ← c95530, ← c95531] at e_1560_8
  have e_1560_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_9
  simp only [← c95532, ← c95533, k2, ← c95534] at e_1560_9
  have hr := e_1560_9
  simp only [e_1560_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15323)) := by
  have c95535 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95536 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95537 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95538 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95539 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95540 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_10
  simp only [← c95535, ← c95536, ← c95537] at e_1560_10
  have e_1560_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_11
  simp only [← c95538, ← c95539, k2, ← c95540] at e_1560_11
  have hr := e_1560_11
  simp only [e_1560_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15324)) := by
  have c95541 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95542 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95543 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95544 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95545 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95546 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_12
  simp only [← c95541, ← c95542, ← c95543] at e_1560_12
  have e_1560_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_13
  simp only [← c95544, ← c95545, k2, ← c95546] at e_1560_13
  have hr := e_1560_13
  simp only [e_1560_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15325)) := by
  have c95547 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95548 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95549 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95550 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95551 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95552 := (publicBatchWrapper86_copies2986 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_14
  simp only [← c95547, ← c95548, ← c95549] at e_1560_14
  have e_1560_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_15
  simp only [← c95550, ← c95551, k2, ← c95552] at e_1560_15
  have hr := e_1560_15
  simp only [e_1560_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15326)) := by
  have c95553 := (publicBatchWrapper86_copies2986 a h).2.1
  have c95554 := (publicBatchWrapper86_copies2986 a h).2.2.1
  have c95555 := (publicBatchWrapper86_copies2986 a h).2.2.2.1
  have c95556 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.1
  have c95557 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.1
  have c95558 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_16
  simp only [← c95553, ← c95554, ← c95555] at e_1560_16
  have e_1560_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_17
  simp only [← c95556, ← c95557, k2, ← c95558] at e_1560_17
  have hr := e_1560_17
  simp only [e_1560_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15327)) := by
  have c95559 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.1
  have c95560 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.1
  have c95561 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.1
  have c95562 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95563 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95564 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_18
  simp only [← c95559, ← c95560, ← c95561] at e_1560_18
  have e_1560_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_19
  simp only [← c95562, ← c95563, k2, ← c95564] at e_1560_19
  have hr := e_1560_19
  simp only [e_1560_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15328)) := by
  have c95565 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95566 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95567 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95568 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95569 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95570 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_0
  simp only [← c95565, ← c95566, ← c95567] at e_1561_0
  have e_1561_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_1
  simp only [← c95568, ← c95569, k2, ← c95570] at e_1561_1
  have hr := e_1561_1
  simp only [e_1561_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15329)) := by
  have c95571 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95572 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95573 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95574 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95575 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95576 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_2
  simp only [← c95571, ← c95572, ← c95573] at e_1561_2
  have e_1561_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_3
  simp only [← c95574, ← c95575, k2, ← c95576] at e_1561_3
  have hr := e_1561_3
  simp only [e_1561_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15330)) := by
  have c95577 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95578 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95579 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95580 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95581 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95582 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_4
  simp only [← c95577, ← c95578, ← c95579] at e_1561_4
  have e_1561_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_5
  simp only [← c95580, ← c95581, k2, ← c95582] at e_1561_5
  have hr := e_1561_5
  simp only [e_1561_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15331)) := by
  have c95583 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95584 := (publicBatchWrapper86_copies2987 a h).1
  have c95585 := (publicBatchWrapper86_copies2987 a h).2.1
  have c95586 := (publicBatchWrapper86_copies2987 a h).2.2.1
  have c95587 := (publicBatchWrapper86_copies2987 a h).2.2.2.1
  have c95588 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_6
  simp only [← c95583, ← c95584, ← c95585] at e_1561_6
  have e_1561_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_7
  simp only [← c95586, ← c95587, k2, ← c95588] at e_1561_7
  have hr := e_1561_7
  simp only [e_1561_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15332)) := by
  have c95589 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.1
  have c95590 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.1
  have c95591 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.1
  have c95592 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.1
  have c95593 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.1
  have c95594 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_8
  simp only [← c95589, ← c95590, ← c95591] at e_1561_8
  have e_1561_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_9
  simp only [← c95592, ← c95593, k2, ← c95594] at e_1561_9
  have hr := e_1561_9
  simp only [e_1561_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15333)) := by
  have c95595 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95596 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95597 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95598 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95599 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95600 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_10
  simp only [← c95595, ← c95596, ← c95597] at e_1561_10
  have e_1561_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_11
  simp only [← c95598, ← c95599, k2, ← c95600] at e_1561_11
  have hr := e_1561_11
  simp only [e_1561_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15334)) := by
  have c95601 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95602 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95603 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95604 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95605 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95606 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_12
  simp only [← c95601, ← c95602, ← c95603] at e_1561_12
  have e_1561_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_13
  simp only [← c95604, ← c95605, k2, ← c95606] at e_1561_13
  have hr := e_1561_13
  simp only [e_1561_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15335)) := by
  have c95607 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95608 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95609 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95610 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95611 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95612 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_14
  simp only [← c95607, ← c95608, ← c95609] at e_1561_14
  have e_1561_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_15
  simp only [← c95610, ← c95611, k2, ← c95612] at e_1561_15
  have hr := e_1561_15
  simp only [e_1561_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15336)) := by
  have c95613 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95614 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95615 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95616 := (publicBatchWrapper86_copies2988 a h).1
  have c95617 := (publicBatchWrapper86_copies2988 a h).2.1
  have c95618 := (publicBatchWrapper86_copies2988 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_16
  simp only [← c95613, ← c95614, ← c95615] at e_1561_16
  have e_1561_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_17
  simp only [← c95616, ← c95617, k2, ← c95618] at e_1561_17
  have hr := e_1561_17
  simp only [e_1561_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15337)) := by
  have c95619 := (publicBatchWrapper86_copies2988 a h).2.2.2.1
  have c95620 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.1
  have c95621 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.1
  have c95622 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.1
  have c95623 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.1
  have c95624 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_18
  simp only [← c95619, ← c95620, ← c95621] at e_1561_18
  have e_1561_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_19
  simp only [← c95622, ← c95623, k2, ← c95624] at e_1561_19
  have hr := e_1561_19
  simp only [e_1561_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15338)) := by
  have c95625 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.1
  have c95626 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95627 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95628 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95629 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95630 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_0
  simp only [← c95625, ← c95626, ← c95627] at e_1562_0
  have e_1562_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_1
  simp only [← c95628, ← c95629, k2, ← c95630] at e_1562_1
  have hr := e_1562_1
  simp only [e_1562_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15339)) := by
  have c95631 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95632 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95633 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95634 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95635 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95636 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_2
  simp only [← c95631, ← c95632, ← c95633] at e_1562_2
  have e_1562_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_3
  simp only [← c95634, ← c95635, k2, ← c95636] at e_1562_3
  have hr := e_1562_3
  simp only [e_1562_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15340)) := by
  have c95637 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95638 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95639 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95640 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95641 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95642 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_4
  simp only [← c95637, ← c95638, ← c95639] at e_1562_4
  have e_1562_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_5
  simp only [← c95640, ← c95641, k2, ← c95642] at e_1562_5
  have hr := e_1562_5
  simp only [e_1562_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15341)) := by
  have c95643 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95644 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95645 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95646 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95647 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95648 := (publicBatchWrapper86_copies2989 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_6
  simp only [← c95643, ← c95644, ← c95645] at e_1562_6
  have e_1562_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_7
  simp only [← c95646, ← c95647, k2, ← c95648] at e_1562_7
  have hr := e_1562_7
  simp only [e_1562_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15342)) := by
  have c95649 := (publicBatchWrapper86_copies2989 a h).2.1
  have c95650 := (publicBatchWrapper86_copies2989 a h).2.2.1
  have c95651 := (publicBatchWrapper86_copies2989 a h).2.2.2.1
  have c95652 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.1
  have c95653 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.1
  have c95654 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_8
  simp only [← c95649, ← c95650, ← c95651] at e_1562_8
  have e_1562_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_9
  simp only [← c95652, ← c95653, k2, ← c95654] at e_1562_9
  have hr := e_1562_9
  simp only [e_1562_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15343)) := by
  have c95655 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.1
  have c95656 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.1
  have c95657 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.1
  have c95658 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95659 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95660 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_10
  simp only [← c95655, ← c95656, ← c95657] at e_1562_10
  have e_1562_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_11
  simp only [← c95658, ← c95659, k2, ← c95660] at e_1562_11
  have hr := e_1562_11
  simp only [e_1562_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15344)) := by
  have c95661 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95662 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95663 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95664 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95665 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95666 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_12
  simp only [← c95661, ← c95662, ← c95663] at e_1562_12
  have e_1562_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_13
  simp only [← c95664, ← c95665, k2, ← c95666] at e_1562_13
  have hr := e_1562_13
  simp only [e_1562_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15345)) := by
  have c95667 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95668 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95669 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95670 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95671 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95672 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_14
  simp only [← c95667, ← c95668, ← c95669] at e_1562_14
  have e_1562_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_15
  simp only [← c95670, ← c95671, k2, ← c95672] at e_1562_15
  have hr := e_1562_15
  simp only [e_1562_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15346)) := by
  have c95673 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95674 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95675 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95676 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95677 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95678 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_16
  simp only [← c95673, ← c95674, ← c95675] at e_1562_16
  have e_1562_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_17
  simp only [← c95676, ← c95677, k2, ← c95678] at e_1562_17
  have hr := e_1562_17
  simp only [e_1562_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15347)) := by
  have c95679 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95680 := (publicBatchWrapper86_copies2990 a h).1
  have c95681 := (publicBatchWrapper86_copies2990 a h).2.1
  have c95682 := (publicBatchWrapper86_copies2990 a h).2.2.1
  have c95683 := (publicBatchWrapper86_copies2990 a h).2.2.2.1
  have c95684 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_18
  simp only [← c95679, ← c95680, ← c95681] at e_1562_18
  have e_1562_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_19
  simp only [← c95682, ← c95683, k2, ← c95684] at e_1562_19
  have hr := e_1562_19
  simp only [e_1562_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15348)) := by
  have c95685 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.1
  have c95686 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.1
  have c95687 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.1
  have c95688 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.1
  have c95689 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.1
  have c95690 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_0
  simp only [← c95685, ← c95686, ← c95687] at e_1563_0
  have e_1563_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_1
  simp only [← c95688, ← c95689, k2, ← c95690] at e_1563_1
  have hr := e_1563_1
  simp only [e_1563_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15349)) := by
  have c95691 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95692 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95693 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95694 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95695 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95696 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_2
  simp only [← c95691, ← c95692, ← c95693] at e_1563_2
  have e_1563_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_3
  simp only [← c95694, ← c95695, k2, ← c95696] at e_1563_3
  have hr := e_1563_3
  simp only [e_1563_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15350)) := by
  have c95697 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95698 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95699 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95700 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95701 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95702 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_4
  simp only [← c95697, ← c95698, ← c95699] at e_1563_4
  have e_1563_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_5
  simp only [← c95700, ← c95701, k2, ← c95702] at e_1563_5
  have hr := e_1563_5
  simp only [e_1563_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15351)) := by
  have c95703 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95704 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95705 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95706 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95707 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95708 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_6
  simp only [← c95703, ← c95704, ← c95705] at e_1563_6
  have e_1563_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_7
  simp only [← c95706, ← c95707, k2, ← c95708] at e_1563_7
  have hr := e_1563_7
  simp only [e_1563_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15352)) := by
  have c95709 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95710 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95711 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95712 := (publicBatchWrapper86_copies2991 a h).1
  have c95713 := (publicBatchWrapper86_copies2991 a h).2.1
  have c95714 := (publicBatchWrapper86_copies2991 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_8
  simp only [← c95709, ← c95710, ← c95711] at e_1563_8
  have e_1563_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_9
  simp only [← c95712, ← c95713, k2, ← c95714] at e_1563_9
  have hr := e_1563_9
  simp only [e_1563_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15353)) := by
  have c95715 := (publicBatchWrapper86_copies2991 a h).2.2.2.1
  have c95716 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.1
  have c95717 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.1
  have c95718 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.1
  have c95719 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.1
  have c95720 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_10
  simp only [← c95715, ← c95716, ← c95717] at e_1563_10
  have e_1563_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_11
  simp only [← c95718, ← c95719, k2, ← c95720] at e_1563_11
  have hr := e_1563_11
  simp only [e_1563_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15354)) := by
  have c95721 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.1
  have c95722 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95723 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95724 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95725 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95726 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_12
  simp only [← c95721, ← c95722, ← c95723] at e_1563_12
  have e_1563_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_13
  simp only [← c95724, ← c95725, k2, ← c95726] at e_1563_13
  have hr := e_1563_13
  simp only [e_1563_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15355)) := by
  have c95727 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95728 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95729 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95730 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95731 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95732 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_14
  simp only [← c95727, ← c95728, ← c95729] at e_1563_14
  have e_1563_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_15
  simp only [← c95730, ← c95731, k2, ← c95732] at e_1563_15
  have hr := e_1563_15
  simp only [e_1563_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15356)) := by
  have c95733 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95734 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95735 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95736 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95737 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95738 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_16
  simp only [← c95733, ← c95734, ← c95735] at e_1563_16
  have e_1563_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_17
  simp only [← c95736, ← c95737, k2, ← c95738] at e_1563_17
  have hr := e_1563_17
  simp only [e_1563_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15357)) := by
  have c95739 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95740 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95741 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95742 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95743 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95744 := (publicBatchWrapper86_copies2992 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_18
  simp only [← c95739, ← c95740, ← c95741] at e_1563_18
  have e_1563_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_19
  simp only [← c95742, ← c95743, k2, ← c95744] at e_1563_19
  have hr := e_1563_19
  simp only [e_1563_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15358)) := by
  have c95745 := (publicBatchWrapper86_copies2992 a h).2.1
  have c95746 := (publicBatchWrapper86_copies2992 a h).2.2.1
  have c95747 := (publicBatchWrapper86_copies2992 a h).2.2.2.1
  have c95748 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.1
  have c95749 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.1
  have c95750 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_0
  simp only [← c95745, ← c95746, ← c95747] at e_1564_0
  have e_1564_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_1
  simp only [← c95748, ← c95749, k2, ← c95750] at e_1564_1
  have hr := e_1564_1
  simp only [e_1564_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15359)) := by
  have c95751 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.1
  have c95752 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.1
  have c95753 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.1
  have c95754 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95755 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95756 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_2
  simp only [← c95751, ← c95752, ← c95753] at e_1564_2
  have e_1564_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_3
  simp only [← c95754, ← c95755, k2, ← c95756] at e_1564_3
  have hr := e_1564_3
  simp only [e_1564_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15360)) := by
  have c95757 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95758 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95759 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95760 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95761 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95762 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_4
  simp only [← c95757, ← c95758, ← c95759] at e_1564_4
  have e_1564_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_5
  simp only [← c95760, ← c95761, k2, ← c95762] at e_1564_5
  have hr := e_1564_5
  simp only [e_1564_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15361)) := by
  have c95763 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95764 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95765 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95766 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95767 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95768 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_6
  simp only [← c95763, ← c95764, ← c95765] at e_1564_6
  have e_1564_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_7
  simp only [← c95766, ← c95767, k2, ← c95768] at e_1564_7
  have hr := e_1564_7
  simp only [e_1564_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15362)) := by
  have c95769 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95770 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95771 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95772 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95773 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95774 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_8
  simp only [← c95769, ← c95770, ← c95771] at e_1564_8
  have e_1564_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_9
  simp only [← c95772, ← c95773, k2, ← c95774] at e_1564_9
  have hr := e_1564_9
  simp only [e_1564_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15363)) := by
  have c95775 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95776 := (publicBatchWrapper86_copies2993 a h).1
  have c95777 := (publicBatchWrapper86_copies2993 a h).2.1
  have c95778 := (publicBatchWrapper86_copies2993 a h).2.2.1
  have c95779 := (publicBatchWrapper86_copies2993 a h).2.2.2.1
  have c95780 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_10
  simp only [← c95775, ← c95776, ← c95777] at e_1564_10
  have e_1564_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_11
  simp only [← c95778, ← c95779, k2, ← c95780] at e_1564_11
  have hr := e_1564_11
  simp only [e_1564_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15364)) := by
  have c95781 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.1
  have c95782 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.1
  have c95783 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.1
  have c95784 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.1
  have c95785 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.1
  have c95786 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_12
  simp only [← c95781, ← c95782, ← c95783] at e_1564_12
  have e_1564_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_13
  simp only [← c95784, ← c95785, k2, ← c95786] at e_1564_13
  have hr := e_1564_13
  simp only [e_1564_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15365)) := by
  have c95787 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95788 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95789 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95790 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95791 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95792 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_14
  simp only [← c95787, ← c95788, ← c95789] at e_1564_14
  have e_1564_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_15
  simp only [← c95790, ← c95791, k2, ← c95792] at e_1564_15
  have hr := e_1564_15
  simp only [e_1564_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15366)) := by
  have c95793 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95794 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95795 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95796 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95797 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95798 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_16
  simp only [← c95793, ← c95794, ← c95795] at e_1564_16
  have e_1564_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_17
  simp only [← c95796, ← c95797, k2, ← c95798] at e_1564_17
  have hr := e_1564_17
  simp only [e_1564_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15367)) := by
  have c95799 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95800 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95801 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95802 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95803 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95804 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_18
  simp only [← c95799, ← c95800, ← c95801] at e_1564_18
  have e_1564_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_19
  simp only [← c95802, ← c95803, k2, ← c95804] at e_1564_19
  have hr := e_1564_19
  simp only [e_1564_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15368)) := by
  have c95805 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95806 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95807 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95808 := (publicBatchWrapper86_copies2994 a h).1
  have c95809 := (publicBatchWrapper86_copies2994 a h).2.1
  have c95810 := (publicBatchWrapper86_copies2994 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_0
  simp only [← c95805, ← c95806, ← c95807] at e_1565_0
  have e_1565_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_1
  simp only [← c95808, ← c95809, k2, ← c95810] at e_1565_1
  have hr := e_1565_1
  simp only [e_1565_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15369)) := by
  have c95811 := (publicBatchWrapper86_copies2994 a h).2.2.2.1
  have c95812 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.1
  have c95813 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.1
  have c95814 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.1
  have c95815 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.1
  have c95816 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_2
  simp only [← c95811, ← c95812, ← c95813] at e_1565_2
  have e_1565_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_3
  simp only [← c95814, ← c95815, k2, ← c95816] at e_1565_3
  have hr := e_1565_3
  simp only [e_1565_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15410)) := by
  have c95817 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.1
  have c95818 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95819 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95820 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95821 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95822 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_4
  simp only [← c95817, ← c95818, ← c95819] at e_1565_4
  have e_1565_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_5
  simp only [← c95820, ← c95821, k2, ← c95822] at e_1565_5
  have hr := e_1565_5
  simp only [e_1565_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15411)) := by
  have c95823 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95824 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95825 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95826 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95827 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95828 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_6
  simp only [← c95823, ← c95824, ← c95825] at e_1565_6
  have e_1565_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_7
  simp only [← c95826, ← c95827, k2, ← c95828] at e_1565_7
  have hr := e_1565_7
  simp only [e_1565_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15412)) := by
  have c95829 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95830 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95831 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95832 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95833 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95834 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_8
  simp only [← c95829, ← c95830, ← c95831] at e_1565_8
  have e_1565_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_9
  simp only [← c95832, ← c95833, k2, ← c95834] at e_1565_9
  have hr := e_1565_9
  simp only [e_1565_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15413)) := by
  have c95835 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95836 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95837 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95838 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95839 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95840 := (publicBatchWrapper86_copies2995 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_10
  simp only [← c95835, ← c95836, ← c95837] at e_1565_10
  have e_1565_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_11
  simp only [← c95838, ← c95839, k2, ← c95840] at e_1565_11
  have hr := e_1565_11
  simp only [e_1565_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15414)) := by
  have c95841 := (publicBatchWrapper86_copies2995 a h).2.1
  have c95842 := (publicBatchWrapper86_copies2995 a h).2.2.1
  have c95843 := (publicBatchWrapper86_copies2995 a h).2.2.2.1
  have c95844 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.1
  have c95845 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.1
  have c95846 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_12
  simp only [← c95841, ← c95842, ← c95843] at e_1565_12
  have e_1565_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_13
  simp only [← c95844, ← c95845, k2, ← c95846] at e_1565_13
  have hr := e_1565_13
  simp only [e_1565_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15415)) := by
  have c95847 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.1
  have c95848 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.1
  have c95849 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.1
  have c95850 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95851 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95852 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_14
  simp only [← c95847, ← c95848, ← c95849] at e_1565_14
  have e_1565_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_15
  simp only [← c95850, ← c95851, k2, ← c95852] at e_1565_15
  have hr := e_1565_15
  simp only [e_1565_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15416)) := by
  have c95853 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95854 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95855 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95856 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95857 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95858 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_16
  simp only [← c95853, ← c95854, ← c95855] at e_1565_16
  have e_1565_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_17
  simp only [← c95856, ← c95857, k2, ← c95858] at e_1565_17
  have hr := e_1565_17
  simp only [e_1565_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15417)) := by
  have c95859 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95860 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95861 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95862 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95863 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95864 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_18
  simp only [← c95859, ← c95860, ← c95861] at e_1565_18
  have e_1565_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_19
  simp only [← c95862, ← c95863, k2, ← c95864] at e_1565_19
  have hr := e_1565_19
  simp only [e_1565_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15418)) := by
  have c95865 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95866 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95867 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95868 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95869 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95870 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_0
  simp only [← c95865, ← c95866, ← c95867] at e_1566_0
  have e_1566_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_1
  simp only [← c95868, ← c95869, k2, ← c95870] at e_1566_1
  have hr := e_1566_1
  simp only [e_1566_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15419)) := by
  have c95871 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95872 := (publicBatchWrapper86_copies2996 a h).1
  have c95873 := (publicBatchWrapper86_copies2996 a h).2.1
  have c95874 := (publicBatchWrapper86_copies2996 a h).2.2.1
  have c95875 := (publicBatchWrapper86_copies2996 a h).2.2.2.1
  have c95876 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_2
  simp only [← c95871, ← c95872, ← c95873] at e_1566_2
  have e_1566_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_3
  simp only [← c95874, ← c95875, k2, ← c95876] at e_1566_3
  have hr := e_1566_3
  simp only [e_1566_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15420)) := by
  have c95877 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.1
  have c95878 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.1
  have c95879 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.1
  have c95880 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.1
  have c95881 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.1
  have c95882 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_4
  simp only [← c95877, ← c95878, ← c95879] at e_1566_4
  have e_1566_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_5
  simp only [← c95880, ← c95881, k2, ← c95882] at e_1566_5
  have hr := e_1566_5
  simp only [e_1566_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15421)) := by
  have c95883 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95884 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95885 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95886 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95887 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95888 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_6
  simp only [← c95883, ← c95884, ← c95885] at e_1566_6
  have e_1566_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_7
  simp only [← c95886, ← c95887, k2, ← c95888] at e_1566_7
  have hr := e_1566_7
  simp only [e_1566_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15422)) := by
  have c95889 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95890 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95891 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95892 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95893 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95894 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_8
  simp only [← c95889, ← c95890, ← c95891] at e_1566_8
  have e_1566_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_9
  simp only [← c95892, ← c95893, k2, ← c95894] at e_1566_9
  have hr := e_1566_9
  simp only [e_1566_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15423)) := by
  have c95895 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95896 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95897 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95898 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95899 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95900 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_10
  simp only [← c95895, ← c95896, ← c95897] at e_1566_10
  have e_1566_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_11
  simp only [← c95898, ← c95899, k2, ← c95900] at e_1566_11
  have hr := e_1566_11
  simp only [e_1566_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15424)) := by
  have c95901 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95902 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95903 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95904 := (publicBatchWrapper86_copies2997 a h).1
  have c95905 := (publicBatchWrapper86_copies2997 a h).2.1
  have c95906 := (publicBatchWrapper86_copies2997 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_12
  simp only [← c95901, ← c95902, ← c95903] at e_1566_12
  have e_1566_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_13
  simp only [← c95904, ← c95905, k2, ← c95906] at e_1566_13
  have hr := e_1566_13
  simp only [e_1566_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15425)) := by
  have c95907 := (publicBatchWrapper86_copies2997 a h).2.2.2.1
  have c95908 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.1
  have c95909 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.1
  have c95910 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.1
  have c95911 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.1
  have c95912 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_14
  simp only [← c95907, ← c95908, ← c95909] at e_1566_14
  have e_1566_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_15
  simp only [← c95910, ← c95911, k2, ← c95912] at e_1566_15
  have hr := e_1566_15
  simp only [e_1566_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15426)) := by
  have c95913 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.1
  have c95914 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95915 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95916 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95917 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95918 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_16
  simp only [← c95913, ← c95914, ← c95915] at e_1566_16
  have e_1566_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_17
  simp only [← c95916, ← c95917, k2, ← c95918] at e_1566_17
  have hr := e_1566_17
  simp only [e_1566_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15427)) := by
  have c95919 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95920 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95921 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95922 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95923 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95924 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_18
  simp only [← c95919, ← c95920, ← c95921] at e_1566_18
  have e_1566_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_19
  simp only [← c95922, ← c95923, k2, ← c95924] at e_1566_19
  have hr := e_1566_19
  simp only [e_1566_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15428)) := by
  have c95925 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95926 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95927 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95928 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95929 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95930 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_0
  simp only [← c95925, ← c95926, ← c95927] at e_1567_0
  have e_1567_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_1
  simp only [← c95928, ← c95929, k2, ← c95930] at e_1567_1
  have hr := e_1567_1
  simp only [e_1567_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15429)) := by
  have c95931 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95932 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95933 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95934 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95935 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95936 := (publicBatchWrapper86_copies2998 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_2
  simp only [← c95931, ← c95932, ← c95933] at e_1567_2
  have e_1567_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_3
  simp only [← c95934, ← c95935, k2, ← c95936] at e_1567_3
  have hr := e_1567_3
  simp only [e_1567_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15430)) := by
  have c95937 := (publicBatchWrapper86_copies2998 a h).2.1
  have c95938 := (publicBatchWrapper86_copies2998 a h).2.2.1
  have c95939 := (publicBatchWrapper86_copies2998 a h).2.2.2.1
  have c95940 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.1
  have c95941 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.1
  have c95942 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_4
  simp only [← c95937, ← c95938, ← c95939] at e_1567_4
  have e_1567_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_5
  simp only [← c95940, ← c95941, k2, ← c95942] at e_1567_5
  have hr := e_1567_5
  simp only [e_1567_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15431)) := by
  have c95943 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.1
  have c95944 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.1
  have c95945 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.1
  have c95946 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95947 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95948 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_6
  simp only [← c95943, ← c95944, ← c95945] at e_1567_6
  have e_1567_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_7
  simp only [← c95946, ← c95947, k2, ← c95948] at e_1567_7
  have hr := e_1567_7
  simp only [e_1567_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15432)) := by
  have c95949 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95950 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95951 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95952 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95953 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95954 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_8
  simp only [← c95949, ← c95950, ← c95951] at e_1567_8
  have e_1567_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_9
  simp only [← c95952, ← c95953, k2, ← c95954] at e_1567_9
  have hr := e_1567_9
  simp only [e_1567_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15433)) := by
  have c95955 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95956 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95957 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95958 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95959 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95960 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_10
  simp only [← c95955, ← c95956, ← c95957] at e_1567_10
  have e_1567_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_11
  simp only [← c95958, ← c95959, k2, ← c95960] at e_1567_11
  have hr := e_1567_11
  simp only [e_1567_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15434)) := by
  have c95961 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95962 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95963 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95964 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95965 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95966 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_12
  simp only [← c95961, ← c95962, ← c95963] at e_1567_12
  have e_1567_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_13
  simp only [← c95964, ← c95965, k2, ← c95966] at e_1567_13
  have hr := e_1567_13
  simp only [e_1567_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15435)) := by
  have c95967 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95968 := (publicBatchWrapper86_copies2999 a h).1
  have c95969 := (publicBatchWrapper86_copies2999 a h).2.1
  have c95970 := (publicBatchWrapper86_copies2999 a h).2.2.1
  have c95971 := (publicBatchWrapper86_copies2999 a h).2.2.2.1
  have c95972 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_14
  simp only [← c95967, ← c95968, ← c95969] at e_1567_14
  have e_1567_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_15
  simp only [← c95970, ← c95971, k2, ← c95972] at e_1567_15
  have hr := e_1567_15
  simp only [e_1567_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15436)) := by
  have c95973 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.1
  have c95974 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.1
  have c95975 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.1
  have c95976 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.1
  have c95977 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.1
  have c95978 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_16
  simp only [← c95973, ← c95974, ← c95975] at e_1567_16
  have e_1567_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_17
  simp only [← c95976, ← c95977, k2, ← c95978] at e_1567_17
  have hr := e_1567_17
  simp only [e_1567_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15437)) := by
  have c95979 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95980 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95981 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95982 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95983 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95984 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_18
  simp only [← c95979, ← c95980, ← c95981] at e_1567_18
  have e_1567_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_19
  simp only [← c95982, ← c95983, k2, ← c95984] at e_1567_19
  have hr := e_1567_19
  simp only [e_1567_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15438)) := by
  have c95985 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95986 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95987 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95988 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95989 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95990 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_0
  simp only [← c95985, ← c95986, ← c95987] at e_1568_0
  have e_1568_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_1
  simp only [← c95988, ← c95989, k2, ← c95990] at e_1568_1
  have hr := e_1568_1
  simp only [e_1568_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15439)) := by
  have c95991 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95992 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95993 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95994 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95995 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95996 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_2
  simp only [← c95991, ← c95992, ← c95993] at e_1568_2
  have e_1568_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_3
  simp only [← c95994, ← c95995, k2, ← c95996] at e_1568_3
  have hr := e_1568_3
  simp only [e_1568_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15440)) := by
  have c95997 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95998 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95999 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96000 := (publicBatchWrapper86_copies3000 a h).1
  have c96001 := (publicBatchWrapper86_copies3000 a h).2.1
  have c96002 := (publicBatchWrapper86_copies3000 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_4
  simp only [← c95997, ← c95998, ← c95999] at e_1568_4
  have e_1568_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_5
  simp only [← c96000, ← c96001, k2, ← c96002] at e_1568_5
  have hr := e_1568_5
  simp only [e_1568_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15441)) := by
  have c96003 := (publicBatchWrapper86_copies3000 a h).2.2.2.1
  have c96004 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.1
  have c96005 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.1
  have c96006 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.1
  have c96007 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.1
  have c96008 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_6
  simp only [← c96003, ← c96004, ← c96005] at e_1568_6
  have e_1568_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_7
  simp only [← c96006, ← c96007, k2, ← c96008] at e_1568_7
  have hr := e_1568_7
  simp only [e_1568_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15442)) := by
  have c96009 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.1
  have c96010 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96011 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96012 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96013 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96014 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_8
  simp only [← c96009, ← c96010, ← c96011] at e_1568_8
  have e_1568_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_9
  simp only [← c96012, ← c96013, k2, ← c96014] at e_1568_9
  have hr := e_1568_9
  simp only [e_1568_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15443)) := by
  have c96015 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96016 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96017 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96018 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96019 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96020 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_10
  simp only [← c96015, ← c96016, ← c96017] at e_1568_10
  have e_1568_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_11
  simp only [← c96018, ← c96019, k2, ← c96020] at e_1568_11
  have hr := e_1568_11
  simp only [e_1568_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15444)) := by
  have c96021 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96022 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96023 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96024 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96025 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96026 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_12
  simp only [← c96021, ← c96022, ← c96023] at e_1568_12
  have e_1568_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_13
  simp only [← c96024, ← c96025, k2, ← c96026] at e_1568_13
  have hr := e_1568_13
  simp only [e_1568_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15445)) := by
  have c96027 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96028 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96029 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96030 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96031 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96032 := (publicBatchWrapper86_copies3001 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_14
  simp only [← c96027, ← c96028, ← c96029] at e_1568_14
  have e_1568_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_15
  simp only [← c96030, ← c96031, k2, ← c96032] at e_1568_15
  have hr := e_1568_15
  simp only [e_1568_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15446)) := by
  have c96033 := (publicBatchWrapper86_copies3001 a h).2.1
  have c96034 := (publicBatchWrapper86_copies3001 a h).2.2.1
  have c96035 := (publicBatchWrapper86_copies3001 a h).2.2.2.1
  have c96036 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.1
  have c96037 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.1
  have c96038 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_16
  simp only [← c96033, ← c96034, ← c96035] at e_1568_16
  have e_1568_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_17
  simp only [← c96036, ← c96037, k2, ← c96038] at e_1568_17
  have hr := e_1568_17
  simp only [e_1568_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15447)) := by
  have c96039 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.1
  have c96040 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.1
  have c96041 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.1
  have c96042 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96043 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96044 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_18
  simp only [← c96039, ← c96040, ← c96041] at e_1568_18
  have e_1568_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_19
  simp only [← c96042, ← c96043, k2, ← c96044] at e_1568_19
  have hr := e_1568_19
  simp only [e_1568_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15448)) := by
  have c96045 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96046 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96047 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96048 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96049 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96050 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_0
  simp only [← c96045, ← c96046, ← c96047] at e_1569_0
  have e_1569_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_1
  simp only [← c96048, ← c96049, k2, ← c96050] at e_1569_1
  have hr := e_1569_1
  simp only [e_1569_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15449)) := by
  have c96051 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96052 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96053 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96054 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96055 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96056 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_2
  simp only [← c96051, ← c96052, ← c96053] at e_1569_2
  have e_1569_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_3
  simp only [← c96054, ← c96055, k2, ← c96056] at e_1569_3
  have hr := e_1569_3
  simp only [e_1569_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15450)) := by
  have c96057 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96058 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96059 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96060 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96061 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96062 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_4
  simp only [← c96057, ← c96058, ← c96059] at e_1569_4
  have e_1569_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_5
  simp only [← c96060, ← c96061, k2, ← c96062] at e_1569_5
  have hr := e_1569_5
  simp only [e_1569_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15451)) := by
  have c96063 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96064 := (publicBatchWrapper86_copies3002 a h).1
  have c96065 := (publicBatchWrapper86_copies3002 a h).2.1
  have c96066 := (publicBatchWrapper86_copies3002 a h).2.2.1
  have c96067 := (publicBatchWrapper86_copies3002 a h).2.2.2.1
  have c96068 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_6
  simp only [← c96063, ← c96064, ← c96065] at e_1569_6
  have e_1569_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_7
  simp only [← c96066, ← c96067, k2, ← c96068] at e_1569_7
  have hr := e_1569_7
  simp only [e_1569_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15452)) := by
  have c96069 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.1
  have c96070 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.1
  have c96071 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.1
  have c96072 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.1
  have c96073 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.1
  have c96074 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_8
  simp only [← c96069, ← c96070, ← c96071] at e_1569_8
  have e_1569_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_9
  simp only [← c96072, ← c96073, k2, ← c96074] at e_1569_9
  have hr := e_1569_9
  simp only [e_1569_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15453)) := by
  have c96075 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96076 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96077 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96078 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96079 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96080 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_10
  simp only [← c96075, ← c96076, ← c96077] at e_1569_10
  have e_1569_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_11
  simp only [← c96078, ← c96079, k2, ← c96080] at e_1569_11
  have hr := e_1569_11
  simp only [e_1569_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15454)) := by
  have c96081 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96082 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96083 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96084 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96085 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96086 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_12
  simp only [← c96081, ← c96082, ← c96083] at e_1569_12
  have e_1569_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_13
  simp only [← c96084, ← c96085, k2, ← c96086] at e_1569_13
  have hr := e_1569_13
  simp only [e_1569_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15455)) := by
  have c96087 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96088 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96089 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96090 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96091 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96092 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_14
  simp only [← c96087, ← c96088, ← c96089] at e_1569_14
  have e_1569_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_15
  simp only [← c96090, ← c96091, k2, ← c96092] at e_1569_15
  have hr := e_1569_15
  simp only [e_1569_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15456)) := by
  have c96093 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96094 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96095 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96096 := (publicBatchWrapper86_copies3003 a h).1
  have c96097 := (publicBatchWrapper86_copies3003 a h).2.1
  have c96098 := (publicBatchWrapper86_copies3003 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_16
  simp only [← c96093, ← c96094, ← c96095] at e_1569_16
  have e_1569_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_17
  simp only [← c96096, ← c96097, k2, ← c96098] at e_1569_17
  have hr := e_1569_17
  simp only [e_1569_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15457)) := by
  have c96099 := (publicBatchWrapper86_copies3003 a h).2.2.2.1
  have c96100 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.1
  have c96101 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.1
  have c96102 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.1
  have c96103 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.1
  have c96104 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_18
  simp only [← c96099, ← c96100, ← c96101] at e_1569_18
  have e_1569_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_19
  simp only [← c96102, ← c96103, k2, ← c96104] at e_1569_19
  have hr := e_1569_19
  simp only [e_1569_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15458)) := by
  have c96105 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.1
  have c96106 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96107 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96108 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96109 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96110 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_0
  simp only [← c96105, ← c96106, ← c96107] at e_1570_0
  have e_1570_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_1
  simp only [← c96108, ← c96109, k2, ← c96110] at e_1570_1
  have hr := e_1570_1
  simp only [e_1570_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15459)) := by
  have c96111 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96112 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96113 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96114 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96115 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96116 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_2
  simp only [← c96111, ← c96112, ← c96113] at e_1570_2
  have e_1570_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_3
  simp only [← c96114, ← c96115, k2, ← c96116] at e_1570_3
  have hr := e_1570_3
  simp only [e_1570_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15460)) := by
  have c96117 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96118 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96119 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96120 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96121 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96122 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_4
  simp only [← c96117, ← c96118, ← c96119] at e_1570_4
  have e_1570_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_5
  simp only [← c96120, ← c96121, k2, ← c96122] at e_1570_5
  have hr := e_1570_5
  simp only [e_1570_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15461)) := by
  have c96123 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96124 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96125 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96126 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96127 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96128 := (publicBatchWrapper86_copies3004 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_6
  simp only [← c96123, ← c96124, ← c96125] at e_1570_6
  have e_1570_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_7
  simp only [← c96126, ← c96127, k2, ← c96128] at e_1570_7
  have hr := e_1570_7
  simp only [e_1570_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15462)) := by
  have c96129 := (publicBatchWrapper86_copies3004 a h).2.1
  have c96130 := (publicBatchWrapper86_copies3004 a h).2.2.1
  have c96131 := (publicBatchWrapper86_copies3004 a h).2.2.2.1
  have c96132 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.1
  have c96133 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.1
  have c96134 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_8
  simp only [← c96129, ← c96130, ← c96131] at e_1570_8
  have e_1570_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_9
  simp only [← c96132, ← c96133, k2, ← c96134] at e_1570_9
  have hr := e_1570_9
  simp only [e_1570_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15463)) := by
  have c96135 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.1
  have c96136 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.1
  have c96137 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.1
  have c96138 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96139 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96140 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_10
  simp only [← c96135, ← c96136, ← c96137] at e_1570_10
  have e_1570_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_11
  simp only [← c96138, ← c96139, k2, ← c96140] at e_1570_11
  have hr := e_1570_11
  simp only [e_1570_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15464)) := by
  have c96141 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96142 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96143 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96144 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96145 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96146 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_12
  simp only [← c96141, ← c96142, ← c96143] at e_1570_12
  have e_1570_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_13
  simp only [← c96144, ← c96145, k2, ← c96146] at e_1570_13
  have hr := e_1570_13
  simp only [e_1570_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15465)) := by
  have c96147 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96148 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96149 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96150 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96151 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96152 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_14
  simp only [← c96147, ← c96148, ← c96149] at e_1570_14
  have e_1570_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_15
  simp only [← c96150, ← c96151, k2, ← c96152] at e_1570_15
  have hr := e_1570_15
  simp only [e_1570_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15466)) := by
  have c96153 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96154 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96155 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96156 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96157 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96158 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_16
  simp only [← c96153, ← c96154, ← c96155] at e_1570_16
  have e_1570_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_17
  simp only [← c96156, ← c96157, k2, ← c96158] at e_1570_17
  have hr := e_1570_17
  simp only [e_1570_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15467)) := by
  have c96159 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96160 := (publicBatchWrapper86_copies3005 a h).1
  have c96161 := (publicBatchWrapper86_copies3005 a h).2.1
  have c96162 := (publicBatchWrapper86_copies3005 a h).2.2.1
  have c96163 := (publicBatchWrapper86_copies3005 a h).2.2.2.1
  have c96164 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_18
  simp only [← c96159, ← c96160, ← c96161] at e_1570_18
  have e_1570_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_19
  simp only [← c96162, ← c96163, k2, ← c96164] at e_1570_19
  have hr := e_1570_19
  simp only [e_1570_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15468)) := by
  have c96165 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.1
  have c96166 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.1
  have c96167 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.1
  have c96168 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.1
  have c96169 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.1
  have c96170 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_0
  simp only [← c96165, ← c96166, ← c96167] at e_1571_0
  have e_1571_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_1
  simp only [← c96168, ← c96169, k2, ← c96170] at e_1571_1
  have hr := e_1571_1
  simp only [e_1571_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15469)) := by
  have c96171 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96172 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96173 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96174 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96175 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96176 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_2
  simp only [← c96171, ← c96172, ← c96173] at e_1571_2
  have e_1571_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_3
  simp only [← c96174, ← c96175, k2, ← c96176] at e_1571_3
  have hr := e_1571_3
  simp only [e_1571_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15470)) := by
  have c96177 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96178 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96179 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96180 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96181 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96182 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_4
  simp only [← c96177, ← c96178, ← c96179] at e_1571_4
  have e_1571_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_5
  simp only [← c96180, ← c96181, k2, ← c96182] at e_1571_5
  have hr := e_1571_5
  simp only [e_1571_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15471)) := by
  have c96183 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96184 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96185 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96186 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96187 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96188 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_6
  simp only [← c96183, ← c96184, ← c96185] at e_1571_6
  have e_1571_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_7
  simp only [← c96186, ← c96187, k2, ← c96188] at e_1571_7
  have hr := e_1571_7
  simp only [e_1571_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15472)) := by
  have c96189 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96190 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96191 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96192 := (publicBatchWrapper86_copies3006 a h).1
  have c96193 := (publicBatchWrapper86_copies3006 a h).2.1
  have c96194 := (publicBatchWrapper86_copies3006 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_8
  simp only [← c96189, ← c96190, ← c96191] at e_1571_8
  have e_1571_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_9
  simp only [← c96192, ← c96193, k2, ← c96194] at e_1571_9
  have hr := e_1571_9
  simp only [e_1571_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15473)) := by
  have c96195 := (publicBatchWrapper86_copies3006 a h).2.2.2.1
  have c96196 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.1
  have c96197 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.1
  have c96198 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.1
  have c96199 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.1
  have c96200 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_10
  simp only [← c96195, ← c96196, ← c96197] at e_1571_10
  have e_1571_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_11
  simp only [← c96198, ← c96199, k2, ← c96200] at e_1571_11
  have hr := e_1571_11
  simp only [e_1571_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15474)) := by
  have c96201 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.1
  have c96202 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96203 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96204 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96205 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96206 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_12
  simp only [← c96201, ← c96202, ← c96203] at e_1571_12
  have e_1571_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_13
  simp only [← c96204, ← c96205, k2, ← c96206] at e_1571_13
  have hr := e_1571_13
  simp only [e_1571_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15475)) := by
  have c96207 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96208 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96209 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96210 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96211 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96212 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_14
  simp only [← c96207, ← c96208, ← c96209] at e_1571_14
  have e_1571_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_15
  simp only [← c96210, ← c96211, k2, ← c96212] at e_1571_15
  have hr := e_1571_15
  simp only [e_1571_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15476)) := by
  have c96213 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96214 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96215 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96216 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96217 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96218 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_16
  simp only [← c96213, ← c96214, ← c96215] at e_1571_16
  have e_1571_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_17
  simp only [← c96216, ← c96217, k2, ← c96218] at e_1571_17
  have hr := e_1571_17
  simp only [e_1571_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15477)) := by
  have c96219 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96220 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96221 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96222 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96223 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96224 := (publicBatchWrapper86_copies3007 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_18
  simp only [← c96219, ← c96220, ← c96221] at e_1571_18
  have e_1571_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_19
  simp only [← c96222, ← c96223, k2, ← c96224] at e_1571_19
  have hr := e_1571_19
  simp only [e_1571_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15478)) := by
  have c96225 := (publicBatchWrapper86_copies3007 a h).2.1
  have c96226 := (publicBatchWrapper86_copies3007 a h).2.2.1
  have c96227 := (publicBatchWrapper86_copies3007 a h).2.2.2.1
  have c96228 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.1
  have c96229 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.1
  have c96230 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_0
  simp only [← c96225, ← c96226, ← c96227] at e_1572_0
  have e_1572_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_1
  simp only [← c96228, ← c96229, k2, ← c96230] at e_1572_1
  have hr := e_1572_1
  simp only [e_1572_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15479)) := by
  have c96231 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.1
  have c96232 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.1
  have c96233 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.1
  have c96234 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96235 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96236 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_2
  simp only [← c96231, ← c96232, ← c96233] at e_1572_2
  have e_1572_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_3
  simp only [← c96234, ← c96235, k2, ← c96236] at e_1572_3
  have hr := e_1572_3
  simp only [e_1572_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15480)) := by
  have c96237 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96238 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96239 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96240 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96241 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96242 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_4
  simp only [← c96237, ← c96238, ← c96239] at e_1572_4
  have e_1572_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_5
  simp only [← c96240, ← c96241, k2, ← c96242] at e_1572_5
  have hr := e_1572_5
  simp only [e_1572_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15481)) := by
  have c96243 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96244 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96245 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96246 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96247 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96248 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_6
  simp only [← c96243, ← c96244, ← c96245] at e_1572_6
  have e_1572_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_7
  simp only [← c96246, ← c96247, k2, ← c96248] at e_1572_7
  have hr := e_1572_7
  simp only [e_1572_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15482)) := by
  have c96249 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96250 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96251 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96252 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96253 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96254 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_8
  simp only [← c96249, ← c96250, ← c96251] at e_1572_8
  have e_1572_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_9
  simp only [← c96252, ← c96253, k2, ← c96254] at e_1572_9
  have hr := e_1572_9
  simp only [e_1572_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15483)) := by
  have c96255 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96256 := (publicBatchWrapper86_copies3008 a h).1
  have c96257 := (publicBatchWrapper86_copies3008 a h).2.1
  have c96258 := (publicBatchWrapper86_copies3008 a h).2.2.1
  have c96259 := (publicBatchWrapper86_copies3008 a h).2.2.2.1
  have c96260 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_10
  simp only [← c96255, ← c96256, ← c96257] at e_1572_10
  have e_1572_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_11
  simp only [← c96258, ← c96259, k2, ← c96260] at e_1572_11
  have hr := e_1572_11
  simp only [e_1572_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15484)) := by
  have c96261 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.1
  have c96262 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.1
  have c96263 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.1
  have c96264 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.1
  have c96265 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.1
  have c96266 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_12
  simp only [← c96261, ← c96262, ← c96263] at e_1572_12
  have e_1572_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_13
  simp only [← c96264, ← c96265, k2, ← c96266] at e_1572_13
  have hr := e_1572_13
  simp only [e_1572_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15485)) := by
  have c96267 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96268 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96269 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96270 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96271 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96272 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_14
  simp only [← c96267, ← c96268, ← c96269] at e_1572_14
  have e_1572_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_15
  simp only [← c96270, ← c96271, k2, ← c96272] at e_1572_15
  have hr := e_1572_15
  simp only [e_1572_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15486)) := by
  have c96273 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96274 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96275 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96276 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96277 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96278 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_16
  simp only [← c96273, ← c96274, ← c96275] at e_1572_16
  have e_1572_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_17
  simp only [← c96276, ← c96277, k2, ← c96278] at e_1572_17
  have hr := e_1572_17
  simp only [e_1572_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15487)) := by
  have c96279 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96280 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96281 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96282 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96283 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96284 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_18
  simp only [← c96279, ← c96280, ← c96281] at e_1572_18
  have e_1572_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_19
  simp only [← c96282, ← c96283, k2, ← c96284] at e_1572_19
  have hr := e_1572_19
  simp only [e_1572_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15488)) := by
  have c96285 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96286 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96287 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96288 := (publicBatchWrapper86_copies3009 a h).1
  have c96289 := (publicBatchWrapper86_copies3009 a h).2.1
  have c96290 := (publicBatchWrapper86_copies3009 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_0
  simp only [← c96285, ← c96286, ← c96287] at e_1573_0
  have e_1573_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_1
  simp only [← c96288, ← c96289, k2, ← c96290] at e_1573_1
  have hr := e_1573_1
  simp only [e_1573_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15489)) := by
  have c96291 := (publicBatchWrapper86_copies3009 a h).2.2.2.1
  have c96292 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.1
  have c96293 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.1
  have c96294 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.1
  have c96295 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.1
  have c96296 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_2
  simp only [← c96291, ← c96292, ← c96293] at e_1573_2
  have e_1573_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_3
  simp only [← c96294, ← c96295, k2, ← c96296] at e_1573_3
  have hr := e_1573_3
  simp only [e_1573_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15490)) := by
  have c96297 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.1
  have c96298 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96299 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96300 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96301 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96302 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_4
  simp only [← c96297, ← c96298, ← c96299] at e_1573_4
  have e_1573_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_5
  simp only [← c96300, ← c96301, k2, ← c96302] at e_1573_5
  have hr := e_1573_5
  simp only [e_1573_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15491)) := by
  have c96303 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96304 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96305 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96306 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96307 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96308 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_6
  simp only [← c96303, ← c96304, ← c96305] at e_1573_6
  have e_1573_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_7
  simp only [← c96306, ← c96307, k2, ← c96308] at e_1573_7
  have hr := e_1573_7
  simp only [e_1573_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15492)) := by
  have c96309 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96310 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96311 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96312 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96313 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96314 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_8
  simp only [← c96309, ← c96310, ← c96311] at e_1573_8
  have e_1573_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_9
  simp only [← c96312, ← c96313, k2, ← c96314] at e_1573_9
  have hr := e_1573_9
  simp only [e_1573_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15493)) := by
  have c96315 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96316 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96317 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96318 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96319 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96320 := (publicBatchWrapper86_copies3010 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_10
  simp only [← c96315, ← c96316, ← c96317] at e_1573_10
  have e_1573_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_11
  simp only [← c96318, ← c96319, k2, ← c96320] at e_1573_11
  have hr := e_1573_11
  simp only [e_1573_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15494)) := by
  have c96321 := (publicBatchWrapper86_copies3010 a h).2.1
  have c96322 := (publicBatchWrapper86_copies3010 a h).2.2.1
  have c96323 := (publicBatchWrapper86_copies3010 a h).2.2.2.1
  have c96324 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.1
  have c96325 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.1
  have c96326 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_12
  simp only [← c96321, ← c96322, ← c96323] at e_1573_12
  have e_1573_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_13
  simp only [← c96324, ← c96325, k2, ← c96326] at e_1573_13
  have hr := e_1573_13
  simp only [e_1573_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15495)) := by
  have c96327 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.1
  have c96328 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.1
  have c96329 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.1
  have c96330 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96331 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96332 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_14
  simp only [← c96327, ← c96328, ← c96329] at e_1573_14
  have e_1573_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_15
  simp only [← c96330, ← c96331, k2, ← c96332] at e_1573_15
  have hr := e_1573_15
  simp only [e_1573_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15496)) := by
  have c96333 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96334 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96335 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96336 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96337 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96338 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_16
  simp only [← c96333, ← c96334, ← c96335] at e_1573_16
  have e_1573_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_17
  simp only [← c96336, ← c96337, k2, ← c96338] at e_1573_17
  have hr := e_1573_17
  simp only [e_1573_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15497)) := by
  have c96339 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96340 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96341 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96342 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96343 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96344 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_18
  simp only [← c96339, ← c96340, ← c96341] at e_1573_18
  have e_1573_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_19
  simp only [← c96342, ← c96343, k2, ← c96344] at e_1573_19
  have hr := e_1573_19
  simp only [e_1573_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15498)) := by
  have c96345 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96346 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96347 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96348 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96349 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96350 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_0
  simp only [← c96345, ← c96346, ← c96347] at e_1574_0
  have e_1574_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_1
  simp only [← c96348, ← c96349, k2, ← c96350] at e_1574_1
  have hr := e_1574_1
  simp only [e_1574_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15499)) := by
  have c96351 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96352 := (publicBatchWrapper86_copies3011 a h).1
  have c96353 := (publicBatchWrapper86_copies3011 a h).2.1
  have c96354 := (publicBatchWrapper86_copies3011 a h).2.2.1
  have c96355 := (publicBatchWrapper86_copies3011 a h).2.2.2.1
  have c96356 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_2
  simp only [← c96351, ← c96352, ← c96353] at e_1574_2
  have e_1574_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_3
  simp only [← c96354, ← c96355, k2, ← c96356] at e_1574_3
  have hr := e_1574_3
  simp only [e_1574_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15500)) := by
  have c96357 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.1
  have c96358 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.1
  have c96359 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.1
  have c96360 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.1
  have c96361 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.1
  have c96362 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_4
  simp only [← c96357, ← c96358, ← c96359] at e_1574_4
  have e_1574_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_5
  simp only [← c96360, ← c96361, k2, ← c96362] at e_1574_5
  have hr := e_1574_5
  simp only [e_1574_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15501)) := by
  have c96363 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96364 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96365 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96366 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96367 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96368 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_6
  simp only [← c96363, ← c96364, ← c96365] at e_1574_6
  have e_1574_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_7
  simp only [← c96366, ← c96367, k2, ← c96368] at e_1574_7
  have hr := e_1574_7
  simp only [e_1574_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15502)) := by
  have c96369 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96370 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96371 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96372 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96373 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96374 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_8
  simp only [← c96369, ← c96370, ← c96371] at e_1574_8
  have e_1574_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_9
  simp only [← c96372, ← c96373, k2, ← c96374] at e_1574_9
  have hr := e_1574_9
  simp only [e_1574_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15503)) := by
  have c96375 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96376 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96377 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96378 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96379 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96380 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_10
  simp only [← c96375, ← c96376, ← c96377] at e_1574_10
  have e_1574_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_11
  simp only [← c96378, ← c96379, k2, ← c96380] at e_1574_11
  have hr := e_1574_11
  simp only [e_1574_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15504)) := by
  have c96381 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96382 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96383 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96384 := (publicBatchWrapper86_copies3012 a h).1
  have c96385 := (publicBatchWrapper86_copies3012 a h).2.1
  have c96386 := (publicBatchWrapper86_copies3012 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_12
  simp only [← c96381, ← c96382, ← c96383] at e_1574_12
  have e_1574_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_13
  simp only [← c96384, ← c96385, k2, ← c96386] at e_1574_13
  have hr := e_1574_13
  simp only [e_1574_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15505)) := by
  have c96387 := (publicBatchWrapper86_copies3012 a h).2.2.2.1
  have c96388 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.1
  have c96389 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.1
  have c96390 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.1
  have c96391 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.1
  have c96392 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_14
  simp only [← c96387, ← c96388, ← c96389] at e_1574_14
  have e_1574_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_15
  simp only [← c96390, ← c96391, k2, ← c96392] at e_1574_15
  have hr := e_1574_15
  simp only [e_1574_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15506)) := by
  have c96393 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.1
  have c96394 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96395 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96396 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96397 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96398 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_16
  simp only [← c96393, ← c96394, ← c96395] at e_1574_16
  have e_1574_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_17
  simp only [← c96396, ← c96397, k2, ← c96398] at e_1574_17
  have hr := e_1574_17
  simp only [e_1574_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15507)) := by
  have c96399 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96400 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96401 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96402 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96403 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96404 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_18
  simp only [← c96399, ← c96400, ← c96401] at e_1574_18
  have e_1574_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_19
  simp only [← c96402, ← c96403, k2, ← c96404] at e_1574_19
  have hr := e_1574_19
  simp only [e_1574_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15508)) := by
  have c96405 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96406 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96407 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96408 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96409 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96410 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_0
  simp only [← c96405, ← c96406, ← c96407] at e_1575_0
  have e_1575_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_1
  simp only [← c96408, ← c96409, k2, ← c96410] at e_1575_1
  have hr := e_1575_1
  simp only [e_1575_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15509)) := by
  have c96411 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96412 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96413 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96414 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96415 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96416 := (publicBatchWrapper86_copies3013 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_2
  simp only [← c96411, ← c96412, ← c96413] at e_1575_2
  have e_1575_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_3
  simp only [← c96414, ← c96415, k2, ← c96416] at e_1575_3
  have hr := e_1575_3
  simp only [e_1575_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15510)) := by
  have c96417 := (publicBatchWrapper86_copies3013 a h).2.1
  have c96418 := (publicBatchWrapper86_copies3013 a h).2.2.1
  have c96419 := (publicBatchWrapper86_copies3013 a h).2.2.2.1
  have c96420 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.1
  have c96421 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.1
  have c96422 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_4
  simp only [← c96417, ← c96418, ← c96419] at e_1575_4
  have e_1575_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_5
  simp only [← c96420, ← c96421, k2, ← c96422] at e_1575_5
  have hr := e_1575_5
  simp only [e_1575_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15511)) := by
  have c96423 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.1
  have c96424 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.1
  have c96425 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.1
  have c96426 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96427 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96428 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_6
  simp only [← c96423, ← c96424, ← c96425] at e_1575_6
  have e_1575_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_7
  simp only [← c96426, ← c96427, k2, ← c96428] at e_1575_7
  have hr := e_1575_7
  simp only [e_1575_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15512)) := by
  have c96429 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96430 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96431 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96432 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96433 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96434 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_8
  simp only [← c96429, ← c96430, ← c96431] at e_1575_8
  have e_1575_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_9
  simp only [← c96432, ← c96433, k2, ← c96434] at e_1575_9
  have hr := e_1575_9
  simp only [e_1575_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15513)) := by
  have c96435 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96436 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96437 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96438 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96439 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96440 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_10
  simp only [← c96435, ← c96436, ← c96437] at e_1575_10
  have e_1575_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_11
  simp only [← c96438, ← c96439, k2, ← c96440] at e_1575_11
  have hr := e_1575_11
  simp only [e_1575_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15514)) := by
  have c96441 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96442 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96443 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96444 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96445 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96446 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_12
  simp only [← c96441, ← c96442, ← c96443] at e_1575_12
  have e_1575_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_13
  simp only [← c96444, ← c96445, k2, ← c96446] at e_1575_13
  have hr := e_1575_13
  simp only [e_1575_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15515)) := by
  have c96447 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96448 := (publicBatchWrapper86_copies3014 a h).1
  have c96449 := (publicBatchWrapper86_copies3014 a h).2.1
  have c96450 := (publicBatchWrapper86_copies3014 a h).2.2.1
  have c96451 := (publicBatchWrapper86_copies3014 a h).2.2.2.1
  have c96452 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_14
  simp only [← c96447, ← c96448, ← c96449] at e_1575_14
  have e_1575_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_15
  simp only [← c96450, ← c96451, k2, ← c96452] at e_1575_15
  have hr := e_1575_15
  simp only [e_1575_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15516)) := by
  have c96453 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.1
  have c96454 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.1
  have c96455 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.1
  have c96456 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.1
  have c96457 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.1
  have c96458 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_16
  simp only [← c96453, ← c96454, ← c96455] at e_1575_16
  have e_1575_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_17
  simp only [← c96456, ← c96457, k2, ← c96458] at e_1575_17
  have hr := e_1575_17
  simp only [e_1575_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15517)) := by
  have c96459 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96460 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96461 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96462 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96463 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96464 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_18
  simp only [← c96459, ← c96460, ← c96461] at e_1575_18
  have e_1575_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_19
  simp only [← c96462, ← c96463, k2, ← c96464] at e_1575_19
  have hr := e_1575_19
  simp only [e_1575_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15518)) := by
  have c96465 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96466 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96467 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96468 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96469 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96470 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_0
  simp only [← c96465, ← c96466, ← c96467] at e_1576_0
  have e_1576_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_1
  simp only [← c96468, ← c96469, k2, ← c96470] at e_1576_1
  have hr := e_1576_1
  simp only [e_1576_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15519)) := by
  have c96471 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96472 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96473 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96474 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96475 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96476 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_2
  simp only [← c96471, ← c96472, ← c96473] at e_1576_2
  have e_1576_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_3
  simp only [← c96474, ← c96475, k2, ← c96476] at e_1576_3
  have hr := e_1576_3
  simp only [e_1576_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15520)) := by
  have c96477 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96478 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96479 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96480 := (publicBatchWrapper86_copies3015 a h).1
  have c96481 := (publicBatchWrapper86_copies3015 a h).2.1
  have c96482 := (publicBatchWrapper86_copies3015 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_4
  simp only [← c96477, ← c96478, ← c96479] at e_1576_4
  have e_1576_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_5
  simp only [← c96480, ← c96481, k2, ← c96482] at e_1576_5
  have hr := e_1576_5
  simp only [e_1576_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15521)) := by
  have c96483 := (publicBatchWrapper86_copies3015 a h).2.2.2.1
  have c96484 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.1
  have c96485 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.1
  have c96486 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.1
  have c96487 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.1
  have c96488 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_6
  simp only [← c96483, ← c96484, ← c96485] at e_1576_6
  have e_1576_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_7
  simp only [← c96486, ← c96487, k2, ← c96488] at e_1576_7
  have hr := e_1576_7
  simp only [e_1576_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15522)) := by
  have c96489 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.1
  have c96490 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96491 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96492 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96493 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96494 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_8
  simp only [← c96489, ← c96490, ← c96491] at e_1576_8
  have e_1576_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_9
  simp only [← c96492, ← c96493, k2, ← c96494] at e_1576_9
  have hr := e_1576_9
  simp only [e_1576_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15523)) := by
  have c96495 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96496 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96497 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96498 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96499 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96500 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_10
  simp only [← c96495, ← c96496, ← c96497] at e_1576_10
  have e_1576_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_11
  simp only [← c96498, ← c96499, k2, ← c96500] at e_1576_11
  have hr := e_1576_11
  simp only [e_1576_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15524)) := by
  have c96501 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96502 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96503 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96504 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96505 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96506 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_12
  simp only [← c96501, ← c96502, ← c96503] at e_1576_12
  have e_1576_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_13
  simp only [← c96504, ← c96505, k2, ← c96506] at e_1576_13
  have hr := e_1576_13
  simp only [e_1576_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15525)) := by
  have c96507 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96508 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96509 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96510 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96511 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96512 := (publicBatchWrapper86_copies3016 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_14
  simp only [← c96507, ← c96508, ← c96509] at e_1576_14
  have e_1576_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_15
  simp only [← c96510, ← c96511, k2, ← c96512] at e_1576_15
  have hr := e_1576_15
  simp only [e_1576_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15526)) := by
  have c96513 := (publicBatchWrapper86_copies3016 a h).2.1
  have c96514 := (publicBatchWrapper86_copies3016 a h).2.2.1
  have c96515 := (publicBatchWrapper86_copies3016 a h).2.2.2.1
  have c96516 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.1
  have c96517 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.1
  have c96518 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_16
  simp only [← c96513, ← c96514, ← c96515] at e_1576_16
  have e_1576_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_17
  simp only [← c96516, ← c96517, k2, ← c96518] at e_1576_17
  have hr := e_1576_17
  simp only [e_1576_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15527)) := by
  have c96519 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.1
  have c96520 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.1
  have c96521 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.1
  have c96522 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96523 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96524 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_18
  simp only [← c96519, ← c96520, ← c96521] at e_1576_18
  have e_1576_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_19
  simp only [← c96522, ← c96523, k2, ← c96524] at e_1576_19
  have hr := e_1576_19
  simp only [e_1576_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15528)) := by
  have c96525 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96526 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96527 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96528 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96529 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96530 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_0
  simp only [← c96525, ← c96526, ← c96527] at e_1577_0
  have e_1577_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_1
  simp only [← c96528, ← c96529, k2, ← c96530] at e_1577_1
  have hr := e_1577_1
  simp only [e_1577_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15529)) := by
  have c96531 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96532 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96533 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96534 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96535 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96536 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_2
  simp only [← c96531, ← c96532, ← c96533] at e_1577_2
  have e_1577_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_3
  simp only [← c96534, ← c96535, k2, ← c96536] at e_1577_3
  have hr := e_1577_3
  simp only [e_1577_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15530)) := by
  have c96537 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96538 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96539 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96540 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96541 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96542 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_4
  simp only [← c96537, ← c96538, ← c96539] at e_1577_4
  have e_1577_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_5
  simp only [← c96540, ← c96541, k2, ← c96542] at e_1577_5
  have hr := e_1577_5
  simp only [e_1577_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15531)) := by
  have c96543 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96544 := (publicBatchWrapper86_copies3017 a h).1
  have c96545 := (publicBatchWrapper86_copies3017 a h).2.1
  have c96546 := (publicBatchWrapper86_copies3017 a h).2.2.1
  have c96547 := (publicBatchWrapper86_copies3017 a h).2.2.2.1
  have c96548 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_6
  simp only [← c96543, ← c96544, ← c96545] at e_1577_6
  have e_1577_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_7
  simp only [← c96546, ← c96547, k2, ← c96548] at e_1577_7
  have hr := e_1577_7
  simp only [e_1577_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15532)) := by
  have c96549 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.1
  have c96550 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.1
  have c96551 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.1
  have c96552 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.1
  have c96553 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.1
  have c96554 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_8
  simp only [← c96549, ← c96550, ← c96551] at e_1577_8
  have e_1577_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_9
  simp only [← c96552, ← c96553, k2, ← c96554] at e_1577_9
  have hr := e_1577_9
  simp only [e_1577_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15533)) := by
  have c96555 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96556 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96557 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96558 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96559 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96560 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_10
  simp only [← c96555, ← c96556, ← c96557] at e_1577_10
  have e_1577_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_11
  simp only [← c96558, ← c96559, k2, ← c96560] at e_1577_11
  have hr := e_1577_11
  simp only [e_1577_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15534)) := by
  have c96561 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96562 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96563 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96564 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96565 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96566 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_12
  simp only [← c96561, ← c96562, ← c96563] at e_1577_12
  have e_1577_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_13
  simp only [← c96564, ← c96565, k2, ← c96566] at e_1577_13
  have hr := e_1577_13
  simp only [e_1577_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15535)) := by
  have c96567 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96568 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96569 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96570 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96571 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96572 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_14
  simp only [← c96567, ← c96568, ← c96569] at e_1577_14
  have e_1577_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_15
  simp only [← c96570, ← c96571, k2, ← c96572] at e_1577_15
  have hr := e_1577_15
  simp only [e_1577_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15536)) := by
  have c96573 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96574 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96575 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96576 := (publicBatchWrapper86_copies3018 a h).1
  have c96577 := (publicBatchWrapper86_copies3018 a h).2.1
  have c96578 := (publicBatchWrapper86_copies3018 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_16
  simp only [← c96573, ← c96574, ← c96575] at e_1577_16
  have e_1577_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_17
  simp only [← c96576, ← c96577, k2, ← c96578] at e_1577_17
  have hr := e_1577_17
  simp only [e_1577_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15537)) := by
  have c96579 := (publicBatchWrapper86_copies3018 a h).2.2.2.1
  have c96580 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.1
  have c96581 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.1
  have c96582 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.1
  have c96583 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.1
  have c96584 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_18
  simp only [← c96579, ← c96580, ← c96581] at e_1577_18
  have e_1577_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_19
  simp only [← c96582, ← c96583, k2, ← c96584] at e_1577_19
  have hr := e_1577_19
  simp only [e_1577_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15538)) := by
  have c96585 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.1
  have c96586 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96587 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96588 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96589 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96590 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_0
  simp only [← c96585, ← c96586, ← c96587] at e_1578_0
  have e_1578_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_1
  simp only [← c96588, ← c96589, k2, ← c96590] at e_1578_1
  have hr := e_1578_1
  simp only [e_1578_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15539)) := by
  have c96591 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96592 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96593 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96594 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96595 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96596 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_2
  simp only [← c96591, ← c96592, ← c96593] at e_1578_2
  have e_1578_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_3
  simp only [← c96594, ← c96595, k2, ← c96596] at e_1578_3
  have hr := e_1578_3
  simp only [e_1578_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15540)) := by
  have c96597 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96598 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96599 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96600 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96601 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96602 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_4
  simp only [← c96597, ← c96598, ← c96599] at e_1578_4
  have e_1578_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_5
  simp only [← c96600, ← c96601, k2, ← c96602] at e_1578_5
  have hr := e_1578_5
  simp only [e_1578_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15541)) := by
  have c96603 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96604 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96605 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96606 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96607 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96608 := (publicBatchWrapper86_copies3019 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_6
  simp only [← c96603, ← c96604, ← c96605] at e_1578_6
  have e_1578_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_7
  simp only [← c96606, ← c96607, k2, ← c96608] at e_1578_7
  have hr := e_1578_7
  simp only [e_1578_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15542)) := by
  have c96609 := (publicBatchWrapper86_copies3019 a h).2.1
  have c96610 := (publicBatchWrapper86_copies3019 a h).2.2.1
  have c96611 := (publicBatchWrapper86_copies3019 a h).2.2.2.1
  have c96612 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.1
  have c96613 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.1
  have c96614 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_8
  simp only [← c96609, ← c96610, ← c96611] at e_1578_8
  have e_1578_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_9
  simp only [← c96612, ← c96613, k2, ← c96614] at e_1578_9
  have hr := e_1578_9
  simp only [e_1578_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15543)) := by
  have c96615 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.1
  have c96616 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.1
  have c96617 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.1
  have c96618 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96619 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96620 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_10
  simp only [← c96615, ← c96616, ← c96617] at e_1578_10
  have e_1578_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_11
  simp only [← c96618, ← c96619, k2, ← c96620] at e_1578_11
  have hr := e_1578_11
  simp only [e_1578_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15544)) := by
  have c96621 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96622 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96623 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96624 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96625 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96626 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_12
  simp only [← c96621, ← c96622, ← c96623] at e_1578_12
  have e_1578_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_13
  simp only [← c96624, ← c96625, k2, ← c96626] at e_1578_13
  have hr := e_1578_13
  simp only [e_1578_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15545)) := by
  have c96627 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96628 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96629 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96630 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96631 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96632 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_14
  simp only [← c96627, ← c96628, ← c96629] at e_1578_14
  have e_1578_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_15
  simp only [← c96630, ← c96631, k2, ← c96632] at e_1578_15
  have hr := e_1578_15
  simp only [e_1578_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15546)) := by
  have c96633 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96634 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96635 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96636 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96637 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96638 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_16
  simp only [← c96633, ← c96634, ← c96635] at e_1578_16
  have e_1578_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_17
  simp only [← c96636, ← c96637, k2, ← c96638] at e_1578_17
  have hr := e_1578_17
  simp only [e_1578_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15547)) := by
  have c96639 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96640 := (publicBatchWrapper86_copies3020 a h).1
  have c96641 := (publicBatchWrapper86_copies3020 a h).2.1
  have c96642 := (publicBatchWrapper86_copies3020 a h).2.2.1
  have c96643 := (publicBatchWrapper86_copies3020 a h).2.2.2.1
  have c96644 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_18
  simp only [← c96639, ← c96640, ← c96641] at e_1578_18
  have e_1578_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_19
  simp only [← c96642, ← c96643, k2, ← c96644] at e_1578_19
  have hr := e_1578_19
  simp only [e_1578_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15548)) := by
  have c96645 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.1
  have c96646 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.1
  have c96647 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.1
  have c96648 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.1
  have c96649 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.1
  have c96650 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_0
  simp only [← c96645, ← c96646, ← c96647] at e_1579_0
  have e_1579_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_1
  simp only [← c96648, ← c96649, k2, ← c96650] at e_1579_1
  have hr := e_1579_1
  simp only [e_1579_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15549)) := by
  have c96651 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96652 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96653 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96654 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96655 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96656 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_2
  simp only [← c96651, ← c96652, ← c96653] at e_1579_2
  have e_1579_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_3
  simp only [← c96654, ← c96655, k2, ← c96656] at e_1579_3
  have hr := e_1579_3
  simp only [e_1579_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15550)) := by
  have c96657 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96658 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96659 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96660 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96661 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96662 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_4
  simp only [← c96657, ← c96658, ← c96659] at e_1579_4
  have e_1579_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_5
  simp only [← c96660, ← c96661, k2, ← c96662] at e_1579_5
  have hr := e_1579_5
  simp only [e_1579_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15551)) := by
  have c96663 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96664 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96665 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96666 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96667 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96668 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_6
  simp only [← c96663, ← c96664, ← c96665] at e_1579_6
  have e_1579_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_7
  simp only [← c96666, ← c96667, k2, ← c96668] at e_1579_7
  have hr := e_1579_7
  simp only [e_1579_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15552)) := by
  have c96669 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96670 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96671 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96672 := (publicBatchWrapper86_copies3021 a h).1
  have c96673 := (publicBatchWrapper86_copies3021 a h).2.1
  have c96674 := (publicBatchWrapper86_copies3021 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_8
  simp only [← c96669, ← c96670, ← c96671] at e_1579_8
  have e_1579_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_9
  simp only [← c96672, ← c96673, k2, ← c96674] at e_1579_9
  have hr := e_1579_9
  simp only [e_1579_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15553)) := by
  have c96675 := (publicBatchWrapper86_copies3021 a h).2.2.2.1
  have c96676 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.1
  have c96677 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.1
  have c96678 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.1
  have c96679 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.1
  have c96680 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_10
  simp only [← c96675, ← c96676, ← c96677] at e_1579_10
  have e_1579_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_11
  simp only [← c96678, ← c96679, k2, ← c96680] at e_1579_11
  have hr := e_1579_11
  simp only [e_1579_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15554)) := by
  have c96681 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.1
  have c96682 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96683 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96684 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96685 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96686 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_12
  simp only [← c96681, ← c96682, ← c96683] at e_1579_12
  have e_1579_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_13
  simp only [← c96684, ← c96685, k2, ← c96686] at e_1579_13
  have hr := e_1579_13
  simp only [e_1579_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15555)) := by
  have c96687 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96688 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96689 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96690 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96691 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96692 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_14
  simp only [← c96687, ← c96688, ← c96689] at e_1579_14
  have e_1579_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_15
  simp only [← c96690, ← c96691, k2, ← c96692] at e_1579_15
  have hr := e_1579_15
  simp only [e_1579_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15556)) := by
  have c96693 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96694 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96695 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96696 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96697 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96698 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_16
  simp only [← c96693, ← c96694, ← c96695] at e_1579_16
  have e_1579_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_17
  simp only [← c96696, ← c96697, k2, ← c96698] at e_1579_17
  have hr := e_1579_17
  simp only [e_1579_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15557)) := by
  have c96699 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96700 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96701 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96702 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96703 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96704 := (publicBatchWrapper86_copies3022 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_18
  simp only [← c96699, ← c96700, ← c96701] at e_1579_18
  have e_1579_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_19
  simp only [← c96702, ← c96703, k2, ← c96704] at e_1579_19
  have hr := e_1579_19
  simp only [e_1579_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15558)) := by
  have c96705 := (publicBatchWrapper86_copies3022 a h).2.1
  have c96706 := (publicBatchWrapper86_copies3022 a h).2.2.1
  have c96707 := (publicBatchWrapper86_copies3022 a h).2.2.2.1
  have c96708 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.1
  have c96709 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.1
  have c96710 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_0
  simp only [← c96705, ← c96706, ← c96707] at e_1580_0
  have e_1580_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_1
  simp only [← c96708, ← c96709, k2, ← c96710] at e_1580_1
  have hr := e_1580_1
  simp only [e_1580_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15559)) := by
  have c96711 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.1
  have c96712 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.1
  have c96713 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.1
  have c96714 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96715 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96716 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_2
  simp only [← c96711, ← c96712, ← c96713] at e_1580_2
  have e_1580_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_3
  simp only [← c96714, ← c96715, k2, ← c96716] at e_1580_3
  have hr := e_1580_3
  simp only [e_1580_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15560)) := by
  have c96717 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96718 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96719 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96720 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96721 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96722 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_4
  simp only [← c96717, ← c96718, ← c96719] at e_1580_4
  have e_1580_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_5
  simp only [← c96720, ← c96721, k2, ← c96722] at e_1580_5
  have hr := e_1580_5
  simp only [e_1580_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15561)) := by
  have c96723 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96724 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96725 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96726 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96727 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96728 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_6
  simp only [← c96723, ← c96724, ← c96725] at e_1580_6
  have e_1580_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_7
  simp only [← c96726, ← c96727, k2, ← c96728] at e_1580_7
  have hr := e_1580_7
  simp only [e_1580_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15562)) := by
  have c96729 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96730 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96731 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96732 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96733 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96734 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_8
  simp only [← c96729, ← c96730, ← c96731] at e_1580_8
  have e_1580_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_9
  simp only [← c96732, ← c96733, k2, ← c96734] at e_1580_9
  have hr := e_1580_9
  simp only [e_1580_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15563)) := by
  have c96735 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96736 := (publicBatchWrapper86_copies3023 a h).1
  have c96737 := (publicBatchWrapper86_copies3023 a h).2.1
  have c96738 := (publicBatchWrapper86_copies3023 a h).2.2.1
  have c96739 := (publicBatchWrapper86_copies3023 a h).2.2.2.1
  have c96740 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_10
  simp only [← c96735, ← c96736, ← c96737] at e_1580_10
  have e_1580_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_11
  simp only [← c96738, ← c96739, k2, ← c96740] at e_1580_11
  have hr := e_1580_11
  simp only [e_1580_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15564)) := by
  have c96741 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.1
  have c96742 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.1
  have c96743 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.1
  have c96744 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.1
  have c96745 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.1
  have c96746 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_12
  simp only [← c96741, ← c96742, ← c96743] at e_1580_12
  have e_1580_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_13
  simp only [← c96744, ← c96745, k2, ← c96746] at e_1580_13
  have hr := e_1580_13
  simp only [e_1580_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15565)) := by
  have c96747 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96748 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96749 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96750 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96751 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96752 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_14
  simp only [← c96747, ← c96748, ← c96749] at e_1580_14
  have e_1580_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_15
  simp only [← c96750, ← c96751, k2, ← c96752] at e_1580_15
  have hr := e_1580_15
  simp only [e_1580_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15566)) := by
  have c96753 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96754 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96755 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96756 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96757 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96758 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_16
  simp only [← c96753, ← c96754, ← c96755] at e_1580_16
  have e_1580_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_17
  simp only [← c96756, ← c96757, k2, ← c96758] at e_1580_17
  have hr := e_1580_17
  simp only [e_1580_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15567)) := by
  have c96759 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96760 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96761 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96762 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96763 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96764 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_18
  simp only [← c96759, ← c96760, ← c96761] at e_1580_18
  have e_1580_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_19
  simp only [← c96762, ← c96763, k2, ← c96764] at e_1580_19
  have hr := e_1580_19
  simp only [e_1580_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15568)) := by
  have c96765 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96766 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96767 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96768 := (publicBatchWrapper86_copies3024 a h).1
  have c96769 := (publicBatchWrapper86_copies3024 a h).2.1
  have c96770 := (publicBatchWrapper86_copies3024 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_0
  simp only [← c96765, ← c96766, ← c96767] at e_1581_0
  have e_1581_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_1
  simp only [← c96768, ← c96769, k2, ← c96770] at e_1581_1
  have hr := e_1581_1
  simp only [e_1581_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15569)) := by
  have c96771 := (publicBatchWrapper86_copies3024 a h).2.2.2.1
  have c96772 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.1
  have c96773 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.1
  have c96774 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.1
  have c96775 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.1
  have c96776 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_2
  simp only [← c96771, ← c96772, ← c96773] at e_1581_2
  have e_1581_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_3
  simp only [← c96774, ← c96775, k2, ← c96776] at e_1581_3
  have hr := e_1581_3
  simp only [e_1581_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15610)) := by
  have c96777 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.1
  have c96778 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96779 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96780 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96781 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96782 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_4
  simp only [← c96777, ← c96778, ← c96779] at e_1581_4
  have e_1581_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_5
  simp only [← c96780, ← c96781, k2, ← c96782] at e_1581_5
  have hr := e_1581_5
  simp only [e_1581_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15611)) := by
  have c96783 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96784 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96785 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96786 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96787 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96788 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_6
  simp only [← c96783, ← c96784, ← c96785] at e_1581_6
  have e_1581_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_7
  simp only [← c96786, ← c96787, k2, ← c96788] at e_1581_7
  have hr := e_1581_7
  simp only [e_1581_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15612)) := by
  have c96789 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96790 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96791 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96792 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96793 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96794 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_8
  simp only [← c96789, ← c96790, ← c96791] at e_1581_8
  have e_1581_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_9
  simp only [← c96792, ← c96793, k2, ← c96794] at e_1581_9
  have hr := e_1581_9
  simp only [e_1581_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15613)) := by
  have c96795 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96796 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96797 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96798 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96799 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96800 := (publicBatchWrapper86_copies3025 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_10
  simp only [← c96795, ← c96796, ← c96797] at e_1581_10
  have e_1581_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_11
  simp only [← c96798, ← c96799, k2, ← c96800] at e_1581_11
  have hr := e_1581_11
  simp only [e_1581_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15614)) := by
  have c96801 := (publicBatchWrapper86_copies3025 a h).2.1
  have c96802 := (publicBatchWrapper86_copies3025 a h).2.2.1
  have c96803 := (publicBatchWrapper86_copies3025 a h).2.2.2.1
  have c96804 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.1
  have c96805 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.1
  have c96806 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_12
  simp only [← c96801, ← c96802, ← c96803] at e_1581_12
  have e_1581_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_13
  simp only [← c96804, ← c96805, k2, ← c96806] at e_1581_13
  have hr := e_1581_13
  simp only [e_1581_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15615)) := by
  have c96807 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.1
  have c96808 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.1
  have c96809 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.1
  have c96810 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96811 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96812 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_14
  simp only [← c96807, ← c96808, ← c96809] at e_1581_14
  have e_1581_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_15
  simp only [← c96810, ← c96811, k2, ← c96812] at e_1581_15
  have hr := e_1581_15
  simp only [e_1581_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15616)) := by
  have c96813 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96814 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96815 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96816 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96817 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96818 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_16
  simp only [← c96813, ← c96814, ← c96815] at e_1581_16
  have e_1581_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_17
  simp only [← c96816, ← c96817, k2, ← c96818] at e_1581_17
  have hr := e_1581_17
  simp only [e_1581_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15617)) := by
  have c96819 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96820 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96821 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96822 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96823 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96824 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_18
  simp only [← c96819, ← c96820, ← c96821] at e_1581_18
  have e_1581_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_19
  simp only [← c96822, ← c96823, k2, ← c96824] at e_1581_19
  have hr := e_1581_19
  simp only [e_1581_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15618)) := by
  have c96825 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96826 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96827 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96828 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96829 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96830 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_0
  simp only [← c96825, ← c96826, ← c96827] at e_1582_0
  have e_1582_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_1
  simp only [← c96828, ← c96829, k2, ← c96830] at e_1582_1
  have hr := e_1582_1
  simp only [e_1582_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15619)) := by
  have c96831 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96832 := (publicBatchWrapper86_copies3026 a h).1
  have c96833 := (publicBatchWrapper86_copies3026 a h).2.1
  have c96834 := (publicBatchWrapper86_copies3026 a h).2.2.1
  have c96835 := (publicBatchWrapper86_copies3026 a h).2.2.2.1
  have c96836 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_2
  simp only [← c96831, ← c96832, ← c96833] at e_1582_2
  have e_1582_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_3
  simp only [← c96834, ← c96835, k2, ← c96836] at e_1582_3
  have hr := e_1582_3
  simp only [e_1582_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15620)) := by
  have c96837 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.1
  have c96838 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.1
  have c96839 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.1
  have c96840 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.1
  have c96841 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.1
  have c96842 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_4
  simp only [← c96837, ← c96838, ← c96839] at e_1582_4
  have e_1582_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_5
  simp only [← c96840, ← c96841, k2, ← c96842] at e_1582_5
  have hr := e_1582_5
  simp only [e_1582_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15621)) := by
  have c96843 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96844 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96845 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96846 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96847 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96848 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_6
  simp only [← c96843, ← c96844, ← c96845] at e_1582_6
  have e_1582_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_7
  simp only [← c96846, ← c96847, k2, ← c96848] at e_1582_7
  have hr := e_1582_7
  simp only [e_1582_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15622)) := by
  have c96849 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96850 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96851 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96852 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96853 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96854 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_8
  simp only [← c96849, ← c96850, ← c96851] at e_1582_8
  have e_1582_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_9
  simp only [← c96852, ← c96853, k2, ← c96854] at e_1582_9
  have hr := e_1582_9
  simp only [e_1582_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15623)) := by
  have c96855 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96856 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96857 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96858 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96859 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96860 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_10
  simp only [← c96855, ← c96856, ← c96857] at e_1582_10
  have e_1582_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_11
  simp only [← c96858, ← c96859, k2, ← c96860] at e_1582_11
  have hr := e_1582_11
  simp only [e_1582_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15624)) := by
  have c96861 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96862 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96863 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96864 := (publicBatchWrapper86_copies3027 a h).1
  have c96865 := (publicBatchWrapper86_copies3027 a h).2.1
  have c96866 := (publicBatchWrapper86_copies3027 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_12
  simp only [← c96861, ← c96862, ← c96863] at e_1582_12
  have e_1582_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_13
  simp only [← c96864, ← c96865, k2, ← c96866] at e_1582_13
  have hr := e_1582_13
  simp only [e_1582_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15625)) := by
  have c96867 := (publicBatchWrapper86_copies3027 a h).2.2.2.1
  have c96868 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.1
  have c96869 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.1
  have c96870 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.1
  have c96871 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.1
  have c96872 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_14
  simp only [← c96867, ← c96868, ← c96869] at e_1582_14
  have e_1582_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_15
  simp only [← c96870, ← c96871, k2, ← c96872] at e_1582_15
  have hr := e_1582_15
  simp only [e_1582_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15626)) := by
  have c96873 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.1
  have c96874 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96875 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96876 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96877 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96878 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_16
  simp only [← c96873, ← c96874, ← c96875] at e_1582_16
  have e_1582_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_17
  simp only [← c96876, ← c96877, k2, ← c96878] at e_1582_17
  have hr := e_1582_17
  simp only [e_1582_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15627)) := by
  have c96879 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96880 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96881 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96882 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96883 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96884 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_18
  simp only [← c96879, ← c96880, ← c96881] at e_1582_18
  have e_1582_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_19
  simp only [← c96882, ← c96883, k2, ← c96884] at e_1582_19
  have hr := e_1582_19
  simp only [e_1582_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15628)) := by
  have c96885 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96886 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96887 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96888 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96889 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96890 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_0
  simp only [← c96885, ← c96886, ← c96887] at e_1583_0
  have e_1583_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_1
  simp only [← c96888, ← c96889, k2, ← c96890] at e_1583_1
  have hr := e_1583_1
  simp only [e_1583_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15629)) := by
  have c96891 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96892 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96893 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96894 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96895 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96896 := (publicBatchWrapper86_copies3028 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_2
  simp only [← c96891, ← c96892, ← c96893] at e_1583_2
  have e_1583_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_3
  simp only [← c96894, ← c96895, k2, ← c96896] at e_1583_3
  have hr := e_1583_3
  simp only [e_1583_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15630)) := by
  have c96897 := (publicBatchWrapper86_copies3028 a h).2.1
  have c96898 := (publicBatchWrapper86_copies3028 a h).2.2.1
  have c96899 := (publicBatchWrapper86_copies3028 a h).2.2.2.1
  have c96900 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.1
  have c96901 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.1
  have c96902 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_4
  simp only [← c96897, ← c96898, ← c96899] at e_1583_4
  have e_1583_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_5
  simp only [← c96900, ← c96901, k2, ← c96902] at e_1583_5
  have hr := e_1583_5
  simp only [e_1583_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15631)) := by
  have c96903 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.1
  have c96904 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.1
  have c96905 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.1
  have c96906 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96907 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96908 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_6
  simp only [← c96903, ← c96904, ← c96905] at e_1583_6
  have e_1583_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_7
  simp only [← c96906, ← c96907, k2, ← c96908] at e_1583_7
  have hr := e_1583_7
  simp only [e_1583_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15632)) := by
  have c96909 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96910 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96911 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96912 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96913 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96914 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_8
  simp only [← c96909, ← c96910, ← c96911] at e_1583_8
  have e_1583_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_9
  simp only [← c96912, ← c96913, k2, ← c96914] at e_1583_9
  have hr := e_1583_9
  simp only [e_1583_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15633)) := by
  have c96915 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96916 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96917 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96918 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96919 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96920 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_10
  simp only [← c96915, ← c96916, ← c96917] at e_1583_10
  have e_1583_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_11
  simp only [← c96918, ← c96919, k2, ← c96920] at e_1583_11
  have hr := e_1583_11
  simp only [e_1583_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15634)) := by
  have c96921 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96922 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96923 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96924 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96925 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96926 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_12
  simp only [← c96921, ← c96922, ← c96923] at e_1583_12
  have e_1583_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_13
  simp only [← c96924, ← c96925, k2, ← c96926] at e_1583_13
  have hr := e_1583_13
  simp only [e_1583_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15635)) := by
  have c96927 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96928 := (publicBatchWrapper86_copies3029 a h).1
  have c96929 := (publicBatchWrapper86_copies3029 a h).2.1
  have c96930 := (publicBatchWrapper86_copies3029 a h).2.2.1
  have c96931 := (publicBatchWrapper86_copies3029 a h).2.2.2.1
  have c96932 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_14
  simp only [← c96927, ← c96928, ← c96929] at e_1583_14
  have e_1583_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_15
  simp only [← c96930, ← c96931, k2, ← c96932] at e_1583_15
  have hr := e_1583_15
  simp only [e_1583_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15636)) := by
  have c96933 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.1
  have c96934 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.1
  have c96935 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.1
  have c96936 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.1
  have c96937 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.1
  have c96938 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_16
  simp only [← c96933, ← c96934, ← c96935] at e_1583_16
  have e_1583_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_17
  simp only [← c96936, ← c96937, k2, ← c96938] at e_1583_17
  have hr := e_1583_17
  simp only [e_1583_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15637)) := by
  have c96939 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96940 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96941 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96942 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96943 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96944 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_18
  simp only [← c96939, ← c96940, ← c96941] at e_1583_18
  have e_1583_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_19
  simp only [← c96942, ← c96943, k2, ← c96944] at e_1583_19
  have hr := e_1583_19
  simp only [e_1583_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15638)) := by
  have c96945 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96946 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96947 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96948 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96949 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96950 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_0
  simp only [← c96945, ← c96946, ← c96947] at e_1584_0
  have e_1584_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_1
  simp only [← c96948, ← c96949, k2, ← c96950] at e_1584_1
  have hr := e_1584_1
  simp only [e_1584_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15639)) := by
  have c96951 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96952 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96953 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96954 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96955 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96956 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_2
  simp only [← c96951, ← c96952, ← c96953] at e_1584_2
  have e_1584_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_3
  simp only [← c96954, ← c96955, k2, ← c96956] at e_1584_3
  have hr := e_1584_3
  simp only [e_1584_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15640)) := by
  have c96957 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96958 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96959 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96960 := (publicBatchWrapper86_copies3030 a h).1
  have c96961 := (publicBatchWrapper86_copies3030 a h).2.1
  have c96962 := (publicBatchWrapper86_copies3030 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_4
  simp only [← c96957, ← c96958, ← c96959] at e_1584_4
  have e_1584_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_5
  simp only [← c96960, ← c96961, k2, ← c96962] at e_1584_5
  have hr := e_1584_5
  simp only [e_1584_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15641)) := by
  have c96963 := (publicBatchWrapper86_copies3030 a h).2.2.2.1
  have c96964 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.1
  have c96965 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.1
  have c96966 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.1
  have c96967 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.1
  have c96968 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_6
  simp only [← c96963, ← c96964, ← c96965] at e_1584_6
  have e_1584_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_7
  simp only [← c96966, ← c96967, k2, ← c96968] at e_1584_7
  have hr := e_1584_7
  simp only [e_1584_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15642)) := by
  have c96969 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.1
  have c96970 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96971 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96972 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96973 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96974 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_8
  simp only [← c96969, ← c96970, ← c96971] at e_1584_8
  have e_1584_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_9
  simp only [← c96972, ← c96973, k2, ← c96974] at e_1584_9
  have hr := e_1584_9
  simp only [e_1584_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15643)) := by
  have c96975 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96976 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96977 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96978 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96979 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96980 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_10
  simp only [← c96975, ← c96976, ← c96977] at e_1584_10
  have e_1584_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_11
  simp only [← c96978, ← c96979, k2, ← c96980] at e_1584_11
  have hr := e_1584_11
  simp only [e_1584_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15644)) := by
  have c96981 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96982 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96983 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96984 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96985 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96986 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_12
  simp only [← c96981, ← c96982, ← c96983] at e_1584_12
  have e_1584_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_13
  simp only [← c96984, ← c96985, k2, ← c96986] at e_1584_13
  have hr := e_1584_13
  simp only [e_1584_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15645)) := by
  have c96987 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96988 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96989 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96990 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96991 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96992 := (publicBatchWrapper86_copies3031 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_14
  simp only [← c96987, ← c96988, ← c96989] at e_1584_14
  have e_1584_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_15
  simp only [← c96990, ← c96991, k2, ← c96992] at e_1584_15
  have hr := e_1584_15
  simp only [e_1584_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15646)) := by
  have c96993 := (publicBatchWrapper86_copies3031 a h).2.1
  have c96994 := (publicBatchWrapper86_copies3031 a h).2.2.1
  have c96995 := (publicBatchWrapper86_copies3031 a h).2.2.2.1
  have c96996 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.1
  have c96997 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.1
  have c96998 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_16
  simp only [← c96993, ← c96994, ← c96995] at e_1584_16
  have e_1584_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_17
  simp only [← c96996, ← c96997, k2, ← c96998] at e_1584_17
  have hr := e_1584_17
  simp only [e_1584_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15647)) := by
  have c96999 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.1
  have c97000 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.1
  have c97001 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.1
  have c97002 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97003 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97004 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_18
  simp only [← c96999, ← c97000, ← c97001] at e_1584_18
  have e_1584_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_19
  simp only [← c97002, ← c97003, k2, ← c97004] at e_1584_19
  have hr := e_1584_19
  simp only [e_1584_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15648)) := by
  have c97005 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97006 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97007 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97008 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97009 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97010 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_0
  simp only [← c97005, ← c97006, ← c97007] at e_1585_0
  have e_1585_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_1
  simp only [← c97008, ← c97009, k2, ← c97010] at e_1585_1
  have hr := e_1585_1
  simp only [e_1585_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15649)) := by
  have c97011 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97012 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97013 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97014 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97015 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97016 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_2
  simp only [← c97011, ← c97012, ← c97013] at e_1585_2
  have e_1585_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_3
  simp only [← c97014, ← c97015, k2, ← c97016] at e_1585_3
  have hr := e_1585_3
  simp only [e_1585_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15650)) := by
  have c97017 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97018 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97019 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97020 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97021 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97022 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_4
  simp only [← c97017, ← c97018, ← c97019] at e_1585_4
  have e_1585_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_5
  simp only [← c97020, ← c97021, k2, ← c97022] at e_1585_5
  have hr := e_1585_5
  simp only [e_1585_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15651)) := by
  have c97023 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97024 := (publicBatchWrapper86_copies3032 a h).1
  have c97025 := (publicBatchWrapper86_copies3032 a h).2.1
  have c97026 := (publicBatchWrapper86_copies3032 a h).2.2.1
  have c97027 := (publicBatchWrapper86_copies3032 a h).2.2.2.1
  have c97028 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_6
  simp only [← c97023, ← c97024, ← c97025] at e_1585_6
  have e_1585_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_7
  simp only [← c97026, ← c97027, k2, ← c97028] at e_1585_7
  have hr := e_1585_7
  simp only [e_1585_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15652)) := by
  have c97029 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.1
  have c97030 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.1
  have c97031 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.1
  have c97032 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.1
  have c97033 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.1
  have c97034 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_8
  simp only [← c97029, ← c97030, ← c97031] at e_1585_8
  have e_1585_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_9
  simp only [← c97032, ← c97033, k2, ← c97034] at e_1585_9
  have hr := e_1585_9
  simp only [e_1585_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15653)) := by
  have c97035 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97036 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97037 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97038 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97039 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97040 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_10
  simp only [← c97035, ← c97036, ← c97037] at e_1585_10
  have e_1585_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_11
  simp only [← c97038, ← c97039, k2, ← c97040] at e_1585_11
  have hr := e_1585_11
  simp only [e_1585_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15654)) := by
  have c97041 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97042 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97043 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97044 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97045 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97046 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_12
  simp only [← c97041, ← c97042, ← c97043] at e_1585_12
  have e_1585_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_13
  simp only [← c97044, ← c97045, k2, ← c97046] at e_1585_13
  have hr := e_1585_13
  simp only [e_1585_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15655)) := by
  have c97047 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97048 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97049 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97050 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97051 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97052 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_14
  simp only [← c97047, ← c97048, ← c97049] at e_1585_14
  have e_1585_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1585 (by decide)
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
  have e_1585_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_16
  simp only [← c97053, ← c97054, ← c97055] at e_1585_16
  have e_1585_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1585 (by decide)
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
  have e_1585_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_18
  simp only [← c97059, ← c97060, ← c97061] at e_1585_18
  have e_1585_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1585 (by decide)
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
  have e_1586_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_0
  simp only [← c97065, ← c97066, ← c97067] at e_1586_0
  have e_1586_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1586 (by decide)
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
  have e_1586_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_2
  simp only [← c97071, ← c97072, ← c97073] at e_1586_2
  have e_1586_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1586 (by decide)
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
  have e_1586_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_4
  simp only [← c97077, ← c97078, ← c97079] at e_1586_4
  have e_1586_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1586 (by decide)
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
  have e_1586_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_6
  simp only [← c97083, ← c97084, ← c97085] at e_1586_6
  have e_1586_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1586 (by decide)
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
  have e_1586_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_8
  simp only [← c97089, ← c97090, ← c97091] at e_1586_8
  have e_1586_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1586 (by decide)
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
  have e_1586_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_10
  simp only [← c97095, ← c97096, ← c97097] at e_1586_10
  have e_1586_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1586 (by decide)
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
  have e_1586_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_12
  simp only [← c97101, ← c97102, ← c97103] at e_1586_12
  have e_1586_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1586 (by decide)
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
  have e_1586_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_14
  simp only [← c97107, ← c97108, ← c97109] at e_1586_14
  have e_1586_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1586 (by decide)
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
  have e_1586_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_16
  simp only [← c97113, ← c97114, ← c97115] at e_1586_16
  have e_1586_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1586 (by decide)
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
  have e_1586_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_18
  simp only [← c97119, ← c97120, ← c97121] at e_1586_18
  have e_1586_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1586 (by decide)
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
  have e_1587_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_0
  simp only [← c97125, ← c97126, ← c97127] at e_1587_0
  have e_1587_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1587 (by decide)
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
  have e_1587_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_2
  simp only [← c97131, ← c97132, ← c97133] at e_1587_2
  have e_1587_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1587 (by decide)
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
  have e_1587_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_4
  simp only [← c97137, ← c97138, ← c97139] at e_1587_4
  have e_1587_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1587 (by decide)
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
  have e_1587_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_6
  simp only [← c97143, ← c97144, ← c97145] at e_1587_6
  have e_1587_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1587 (by decide)
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
  have e_1587_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_8
  simp only [← c97149, ← c97150, ← c97151] at e_1587_8
  have e_1587_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1587 (by decide)
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
  have e_1587_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_10
  simp only [← c97155, ← c97156, ← c97157] at e_1587_10
  have e_1587_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1587 (by decide)
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
  have e_1587_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_12
  simp only [← c97161, ← c97162, ← c97163] at e_1587_12
  have e_1587_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1587 (by decide)
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
  have e_1587_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_14
  simp only [← c97167, ← c97168, ← c97169] at e_1587_14
  have e_1587_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1587 (by decide)
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
  have e_1587_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_16
  simp only [← c97173, ← c97174, ← c97175] at e_1587_16
  have e_1587_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1587 (by decide)
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
  have e_1587_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_18
  simp only [← c97179, ← c97180, ← c97181] at e_1587_18
  have e_1587_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1587 (by decide)
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
  have e_1588_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_0
  simp only [← c97185, ← c97186, ← c97187] at e_1588_0
  have e_1588_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1588 (by decide)
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
  have e_1588_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_2
  simp only [← c97191, ← c97192, ← c97193] at e_1588_2
  have e_1588_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1588 (by decide)
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
  have e_1588_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_4
  simp only [← c97197, ← c97198, ← c97199] at e_1588_4
  have e_1588_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1588 (by decide)
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
  have e_1588_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_6
  simp only [← c97203, ← c97204, ← c97205] at e_1588_6
  have e_1588_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1588 (by decide)
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
  have e_1588_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_8
  simp only [← c97209, ← c97210, ← c97211] at e_1588_8
  have e_1588_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1588 (by decide)
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
  have e_1588_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_10
  simp only [← c97215, ← c97216, ← c97217] at e_1588_10
  have e_1588_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1588 (by decide)
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
  have e_1588_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_12
  simp only [← c97221, ← c97222, ← c97223] at e_1588_12
  have e_1588_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1588 (by decide)
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
  have e_1588_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_14
  simp only [← c97227, ← c97228, ← c97229] at e_1588_14
  have e_1588_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1588 (by decide)
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
  have e_1588_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_16
  simp only [← c97233, ← c97234, ← c97235] at e_1588_16
  have e_1588_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1588 (by decide)
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
  have e_1588_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_18
  simp only [← c97239, ← c97240, ← c97241] at e_1588_18
  have e_1588_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1588 (by decide)
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
  have e_1589_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_0
  simp only [← c97245, ← c97246, ← c97247] at e_1589_0
  have e_1589_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1589 (by decide)
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
  have e_1589_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_2
  simp only [← c97251, ← c97252, ← c97253] at e_1589_2
  have e_1589_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1589 (by decide)
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
  have e_1589_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_4
  simp only [← c97257, ← c97258, ← c97259] at e_1589_4
  have e_1589_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1589 (by decide)
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
  have e_1589_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_6
  simp only [← c97263, ← c97264, ← c97265] at e_1589_6
  have e_1589_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1589 (by decide)
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
  have e_1589_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_8
  simp only [← c97269, ← c97270, ← c97271] at e_1589_8
  have e_1589_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1589 (by decide)
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
  have e_1589_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_10
  simp only [← c97275, ← c97276, ← c97277] at e_1589_10
  have e_1589_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1589 (by decide)
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
  have e_1589_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_12
  simp only [← c97281, ← c97282, ← c97283] at e_1589_12
  have e_1589_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1589 (by decide)
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
  have e_1589_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_14
  simp only [← c97287, ← c97288, ← c97289] at e_1589_14
  have e_1589_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1589 (by decide)
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
  have e_1589_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_16
  simp only [← c97293, ← c97294, ← c97295] at e_1589_16
  have e_1589_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1589 (by decide)
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
  have e_1589_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_18
  simp only [← c97299, ← c97300, ← c97301] at e_1589_18
  have e_1589_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1589 (by decide)
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
  have e_1590_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_0
  simp only [← c97305, ← c97306, ← c97307] at e_1590_0
  have e_1590_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1590 (by decide)
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
  have e_1590_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_2
  simp only [← c97311, ← c97312, ← c97313] at e_1590_2
  have e_1590_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1590 (by decide)
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
  have e_1590_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_4
  simp only [← c97317, ← c97318, ← c97319] at e_1590_4
  have e_1590_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1590 (by decide)
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
  have e_1590_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_6
  simp only [← c97323, ← c97324, ← c97325] at e_1590_6
  have e_1590_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1590 (by decide)
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
  have e_1590_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_8
  simp only [← c97329, ← c97330, ← c97331] at e_1590_8
  have e_1590_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1590 (by decide)
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
  have e_1590_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_10
  simp only [← c97335, ← c97336, ← c97337] at e_1590_10
  have e_1590_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1590 (by decide)
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
  have e_1590_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_12
  simp only [← c97341, ← c97342, ← c97343] at e_1590_12
  have e_1590_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1590 (by decide)
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
  have e_1590_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_14
  simp only [← c97347, ← c97348, ← c97349] at e_1590_14
  have e_1590_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1590 (by decide)
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
  have e_1590_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_16
  simp only [← c97353, ← c97354, ← c97355] at e_1590_16
  have e_1590_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1590 (by decide)
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
  have e_1590_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_18
  simp only [← c97359, ← c97360, ← c97361] at e_1590_18
  have e_1590_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1590 (by decide)
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
  have e_1591_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_0
  simp only [← c97365, ← c97366, ← c97367] at e_1591_0
  have e_1591_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1591 (by decide)
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
  have e_1591_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_2
  simp only [← c97371, ← c97372, ← c97373] at e_1591_2
  have e_1591_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1591 (by decide)
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
  have e_1591_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_4
  simp only [← c97377, ← c97378, ← c97379] at e_1591_4
  have e_1591_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1591 (by decide)
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
  have e_1591_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_6
  simp only [← c97383, ← c97384, ← c97385] at e_1591_6
  have e_1591_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1591 (by decide)
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
  have e_1591_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_8
  simp only [← c97389, ← c97390, ← c97391] at e_1591_8
  have e_1591_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1591 (by decide)
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
  have e_1591_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_10
  simp only [← c97395, ← c97396, ← c97397] at e_1591_10
  have e_1591_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1591 (by decide)
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
  have e_1591_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_12
  simp only [← c97401, ← c97402, ← c97403] at e_1591_12
  have e_1591_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1591 (by decide)
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
  have e_1591_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_14
  simp only [← c97407, ← c97408, ← c97409] at e_1591_14
  have e_1591_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1591 (by decide)
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
  have e_1591_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_16
  simp only [← c97413, ← c97414, ← c97415] at e_1591_16
  have e_1591_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1591 (by decide)
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
  have e_1591_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_18
  simp only [← c97419, ← c97420, ← c97421] at e_1591_18
  have e_1591_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1591 (by decide)
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
  have e_1592_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_0
  simp only [← c97425, ← c97426, ← c97427] at e_1592_0
  have e_1592_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1592 (by decide)
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
  have e_1592_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_2
  simp only [← c97431, ← c97432, ← c97433] at e_1592_2
  have e_1592_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1592 (by decide)
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
  have e_1592_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_4
  simp only [← c97437, ← c97438, ← c97439] at e_1592_4
  have e_1592_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1592 (by decide)
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
  have e_1592_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_6
  simp only [← c97443, ← c97444, ← c97445] at e_1592_6
  have e_1592_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1592 (by decide)
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
  have e_1592_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_8
  simp only [← c97449, ← c97450, ← c97451] at e_1592_8
  have e_1592_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1592 (by decide)
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
  have e_1592_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_10
  simp only [← c97455, ← c97456, ← c97457] at e_1592_10
  have e_1592_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1592 (by decide)
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
  have e_1592_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_12
  simp only [← c97461, ← c97462, ← c97463] at e_1592_12
  have e_1592_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1592 (by decide)
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
  have e_1592_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_14
  simp only [← c97467, ← c97468, ← c97469] at e_1592_14
  have e_1592_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1592 (by decide)
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
  have e_1592_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_16
  simp only [← c97473, ← c97474, ← c97475] at e_1592_16
  have e_1592_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1592 (by decide)
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
  have e_1592_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_18
  simp only [← c97479, ← c97480, ← c97481] at e_1592_18
  have e_1592_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1592 (by decide)
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
  have e_1593_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_0
  simp only [← c97485, ← c97486, ← c97487] at e_1593_0
  have e_1593_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1593 (by decide)
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
  have e_1593_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_2
  simp only [← c97491, ← c97492, ← c97493] at e_1593_2
  have e_1593_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1593 (by decide)
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
  have e_1593_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_4
  simp only [← c97497, ← c97498, ← c97499] at e_1593_4
  have e_1593_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1593 (by decide)
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
  have e_1593_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_6
  simp only [← c97503, ← c97504, ← c97505] at e_1593_6
  have e_1593_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1593 (by decide)
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
  have e_1593_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_8
  simp only [← c97509, ← c97510, ← c97511] at e_1593_8
  have e_1593_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1593 (by decide)
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
  have e_1593_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_10
  simp only [← c97515, ← c97516, ← c97517] at e_1593_10
  have e_1593_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1593 (by decide)
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
  have e_1593_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_12
  simp only [← c97521, ← c97522, ← c97523] at e_1593_12
  have e_1593_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1593 (by decide)
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
  have e_1593_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_14
  simp only [← c97527, ← c97528, ← c97529] at e_1593_14
  have e_1593_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1593 (by decide)
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
  have e_1593_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_16
  simp only [← c97533, ← c97534, ← c97535] at e_1593_16
  have e_1593_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1593 (by decide)
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
  have e_1593_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_18
  simp only [← c97539, ← c97540, ← c97541] at e_1593_18
  have e_1593_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1593 (by decide)
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
  have e_1594_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_0
  simp only [← c97545, ← c97546, ← c97547] at e_1594_0
  have e_1594_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1594 (by decide)
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
  have e_1594_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_2
  simp only [← c97551, ← c97552, ← c97553] at e_1594_2
  have e_1594_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1594 (by decide)
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
  have e_1594_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_4
  simp only [← c97557, ← c97558, ← c97559] at e_1594_4
  have e_1594_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1594 (by decide)
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
  have e_1594_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_6
  simp only [← c97563, ← c97564, ← c97565] at e_1594_6
  have e_1594_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1594 (by decide)
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
  have e_1594_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_8
  simp only [← c97569, ← c97570, ← c97571] at e_1594_8
  have e_1594_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1594 (by decide)
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
  have e_1594_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_10
  simp only [← c97575, ← c97576, ← c97577] at e_1594_10
  have e_1594_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1594 (by decide)
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
  have e_1594_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_12
  simp only [← c97581, ← c97582, ← c97583] at e_1594_12
  have e_1594_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1594 (by decide)
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
  have e_1594_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_14
  simp only [← c97587, ← c97588, ← c97589] at e_1594_14
  have e_1594_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1594 (by decide)
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
  have e_1594_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_16
  simp only [← c97593, ← c97594, ← c97595] at e_1594_16
  have e_1594_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1594 (by decide)
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
  have e_1594_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_18
  simp only [← c97599, ← c97600, ← c97601] at e_1594_18
  have e_1594_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1594 (by decide)
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
  have e_1595_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_0
  simp only [← c97605, ← c97606, ← c97607] at e_1595_0
  have e_1595_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1595 (by decide)
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
  have e_1595_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_2
  simp only [← c97611, ← c97612, ← c97613] at e_1595_2
  have e_1595_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1595 (by decide)
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
  have e_1595_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_4
  simp only [← c97617, ← c97618, ← c97619] at e_1595_4
  have e_1595_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1595 (by decide)
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
  have e_1595_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_6
  simp only [← c97623, ← c97624, ← c97625] at e_1595_6
  have e_1595_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1595 (by decide)
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
  have e_1595_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_8
  simp only [← c97629, ← c97630, ← c97631] at e_1595_8
  have e_1595_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1595 (by decide)
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
  have e_1595_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_10
  simp only [← c97635, ← c97636, ← c97637] at e_1595_10
  have e_1595_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1595 (by decide)
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
  have e_1595_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_12
  simp only [← c97641, ← c97642, ← c97643] at e_1595_12
  have e_1595_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1595 (by decide)
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
  have e_1595_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_14
  simp only [← c97647, ← c97648, ← c97649] at e_1595_14
  have e_1595_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1595 (by decide)
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
  have e_1595_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_16
  simp only [← c97653, ← c97654, ← c97655] at e_1595_16
  have e_1595_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1595 (by decide)
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
  have e_1595_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_18
  simp only [← c97659, ← c97660, ← c97661] at e_1595_18
  have e_1595_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1595 (by decide)
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
  have e_1596_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_0
  simp only [← c97665, ← c97666, ← c97667] at e_1596_0
  have e_1596_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1596 (by decide)
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
  have e_1596_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_2
  simp only [← c97671, ← c97672, ← c97673] at e_1596_2
  have e_1596_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1596 (by decide)
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
  have e_1596_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_4
  simp only [← c97677, ← c97678, ← c97679] at e_1596_4
  have e_1596_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1596 (by decide)
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
  have e_1596_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_6
  simp only [← c97683, ← c97684, ← c97685] at e_1596_6
  have e_1596_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1596 (by decide)
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
  have e_1596_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_8
  simp only [← c97689, ← c97690, ← c97691] at e_1596_8
  have e_1596_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1596 (by decide)
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
  have e_1596_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_10
  simp only [← c97695, ← c97696, ← c97697] at e_1596_10
  have e_1596_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1596 (by decide)
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
  have e_1596_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_12
  simp only [← c97701, ← c97702, ← c97703] at e_1596_12
  have e_1596_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1596 (by decide)
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
  have e_1596_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_14
  simp only [← c97707, ← c97708, ← c97709] at e_1596_14
  have e_1596_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1596 (by decide)
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
  have e_1596_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_16
  simp only [← c97713, ← c97714, ← c97715] at e_1596_16
  have e_1596_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1596 (by decide)
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
  have e_1596_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_18
  simp only [← c97719, ← c97720, ← c97721] at e_1596_18
  have e_1596_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1596 (by decide)
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
  have e_1597_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_0
  simp only [← c97725, ← c97726, ← c97727] at e_1597_0
  have e_1597_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1597 (by decide)
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
  have e_1597_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_2
  simp only [← c97731, ← c97732, ← c97733] at e_1597_2
  have e_1597_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1597 (by decide)
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
  have e_1597_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_4
  simp only [← c97737, ← c97738, ← c97739] at e_1597_4
  have e_1597_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1597 (by decide)
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
  have e_1597_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_6
  simp only [← c97743, ← c97744, ← c97745] at e_1597_6
  have e_1597_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1597 (by decide)
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
  have e_1597_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_8
  simp only [← c97749, ← c97750, ← c97751] at e_1597_8
  have e_1597_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1597 (by decide)
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
  have e_1597_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_10
  simp only [← c97755, ← c97756, ← c97757] at e_1597_10
  have e_1597_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1597 (by decide)
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
  have e_1597_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_12
  simp only [← c97761, ← c97762, ← c97763] at e_1597_12
  have e_1597_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1597 (by decide)
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
  have e_1597_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_14
  simp only [← c97767, ← c97768, ← c97769] at e_1597_14
  have e_1597_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1597 (by decide)
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
  have e_1597_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_16
  simp only [← c97773, ← c97774, ← c97775] at e_1597_16
  have e_1597_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1597 (by decide)
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
  have e_1597_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_18
  simp only [← c97779, ← c97780, ← c97781] at e_1597_18
  have e_1597_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1597 (by decide)
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
  have e_1598_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_0
  simp only [← c97785, ← c97786, ← c97787] at e_1598_0
  have e_1598_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1598 (by decide)
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
  have e_1598_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_2
  simp only [← c97791, ← c97792, ← c97793] at e_1598_2
  have e_1598_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1598 (by decide)
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
  have e_1598_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_4
  simp only [← c97797, ← c97798, ← c97799] at e_1598_4
  have e_1598_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1598 (by decide)
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
  have e_1598_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_6
  simp only [← c97803, ← c97804, ← c97805] at e_1598_6
  have e_1598_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1598 (by decide)
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
  have e_1598_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_8
  simp only [← c97809, ← c97810, ← c97811] at e_1598_8
  have e_1598_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1598 (by decide)
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
  have e_1598_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_10
  simp only [← c97815, ← c97816, ← c97817] at e_1598_10
  have e_1598_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1598 (by decide)
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
  have e_1598_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_12
  simp only [← c97821, ← c97822, ← c97823] at e_1598_12
  have e_1598_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1598 (by decide)
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
  have e_1598_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_14
  simp only [← c97827, ← c97828, ← c97829] at e_1598_14
  have e_1598_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1598 (by decide)
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
  have e_1598_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_16
  simp only [← c97833, ← c97834, ← c97835] at e_1598_16
  have e_1598_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1598 (by decide)
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
  have e_1598_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_18
  simp only [← c97839, ← c97840, ← c97841] at e_1598_18
  have e_1598_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1598 (by decide)
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
  have e_1599_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_0
  simp only [← c97845, ← c97846, ← c97847] at e_1599_0
  have e_1599_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1599 (by decide)
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
  have e_1599_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_2
  simp only [← c97851, ← c97852, ← c97853] at e_1599_2
  have e_1599_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1599 (by decide)
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
  have e_1599_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_4
  simp only [← c97857, ← c97858, ← c97859] at e_1599_4
  have e_1599_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1599 (by decide)
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
  have e_1599_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_6
  simp only [← c97863, ← c97864, ← c97865] at e_1599_6
  have e_1599_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1599 (by decide)
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
  have e_1599_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_8
  simp only [← c97869, ← c97870, ← c97871] at e_1599_8
  have e_1599_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1599 (by decide)
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
  have e_1599_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_10
  simp only [← c97875, ← c97876, ← c97877] at e_1599_10
  have e_1599_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1599 (by decide)
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
  have e_1599_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_12
  simp only [← c97881, ← c97882, ← c97883] at e_1599_12
  have e_1599_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1599 (by decide)
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
  have e_1599_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_14
  simp only [← c97887, ← c97888, ← c97889] at e_1599_14
  have e_1599_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1599 (by decide)
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
  have e_1599_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_16
  simp only [← c97893, ← c97894, ← c97895] at e_1599_16
  have e_1599_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1599 (by decide)
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
  have e_1599_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_18
  simp only [← c97899, ← c97900, ← c97901] at e_1599_18
  have e_1599_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1599 (by decide)
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
  have e_1600_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_0
  simp only [← c97905, ← c97906, ← c97907] at e_1600_0
  have e_1600_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1600 (by decide)
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
  have e_1600_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_2
  simp only [← c97911, ← c97912, ← c97913] at e_1600_2
  have e_1600_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1600 (by decide)
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
  have e_1600_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_4
  simp only [← c97917, ← c97918, ← c97919] at e_1600_4
  have e_1600_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1600 (by decide)
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
  have e_1600_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_6
  simp only [← c97923, ← c97924, ← c97925] at e_1600_6
  have e_1600_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1600 (by decide)
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
  have e_1600_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_8
  simp only [← c97929, ← c97930, ← c97931] at e_1600_8
  have e_1600_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1600 (by decide)
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
  have e_1600_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_10
  simp only [← c97935, ← c97936, ← c97937] at e_1600_10
  have e_1600_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1600 (by decide)
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
  have e_1600_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_12
  simp only [← c97941, ← c97942, ← c97943] at e_1600_12
  have e_1600_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1600 (by decide)
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
  have e_1600_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_14
  simp only [← c97947, ← c97948, ← c97949] at e_1600_14
  have e_1600_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1600 (by decide)
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
  have e_1600_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_16
  simp only [← c97953, ← c97954, ← c97955] at e_1600_16
  have e_1600_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1600 (by decide)
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
  have e_1600_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_18
  simp only [← c97959, ← c97960, ← c97961] at e_1600_18
  have e_1600_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1600 (by decide)
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
  have e_1601_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_0
  simp only [← c97965, ← c97966, ← c97967] at e_1601_0
  have e_1601_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1601 (by decide)
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
  have e_1601_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_2
  simp only [← c97971, ← c97972, ← c97973] at e_1601_2
  have e_1601_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1601 (by decide)
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
  have e_1601_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_4
  simp only [← c97977, ← c97978, ← c97979] at e_1601_4
  have e_1601_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1601 (by decide)
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
  have e_1601_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_6
  simp only [← c97983, ← c97984, ← c97985] at e_1601_6
  have e_1601_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1601 (by decide)
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
  have e_1601_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_8
  simp only [← c97989, ← c97990, ← c97991] at e_1601_8
  have e_1601_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1601 (by decide)
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
  have e_1601_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_10
  simp only [← c97995, ← c97996, ← c97997] at e_1601_10
  have e_1601_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1601 (by decide)
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
  have e_1601_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_12
  simp only [← c98001, ← c98002, ← c98003] at e_1601_12
  have e_1601_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1601 (by decide)
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
  have e_1601_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_14
  simp only [← c98007, ← c98008, ← c98009] at e_1601_14
  have e_1601_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1601 (by decide)
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
  have e_1601_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_16
  simp only [← c98013, ← c98014, ← c98015] at e_1601_16
  have e_1601_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1601 (by decide)
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
  have e_1601_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_18
  simp only [← c98019, ← c98020, ← c98021] at e_1601_18
  have e_1601_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1601 (by decide)
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
  have e_1602_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_0
  simp only [← c98025, ← c98026, ← c98027] at e_1602_0
  have e_1602_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1602 (by decide)
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
  have e_1602_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_2
  simp only [← c98031, ← c98032, ← c98033] at e_1602_2
  have e_1602_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1602 (by decide)
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
  have e_1602_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_4
  simp only [← c98037, ← c98038, ← c98039] at e_1602_4
  have e_1602_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1602 (by decide)
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
  have e_1602_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_6
  simp only [← c98043, ← c98044, ← c98045] at e_1602_6
  have e_1602_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1602 (by decide)
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
  have e_1602_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_8
  simp only [← c98049, ← c98050, ← c98051] at e_1602_8
  have e_1602_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1602 (by decide)
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
  have e_1602_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_10
  simp only [← c98055, ← c98056, ← c98057] at e_1602_10
  have e_1602_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1602 (by decide)
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
  have e_1602_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_12
  simp only [← c98061, ← c98062, ← c98063] at e_1602_12
  have e_1602_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1602 (by decide)
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
  have e_1602_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_14
  simp only [← c98067, ← c98068, ← c98069] at e_1602_14
  have e_1602_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1602 (by decide)
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
  have e_1602_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_16
  simp only [← c98073, ← c98074, ← c98075] at e_1602_16
  have e_1602_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1602 (by decide)
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
  have e_1602_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_18
  simp only [← c98079, ← c98080, ← c98081] at e_1602_18
  have e_1602_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1602 (by decide)
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
  have e_1603_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_0
  simp only [← c98085, ← c98086, ← c98087] at e_1603_0
  have e_1603_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1603 (by decide)
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
  have e_1603_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_2
  simp only [← c98091, ← c98092, ← c98093] at e_1603_2
  have e_1603_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1603 (by decide)
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
  have e_1603_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_4
  simp only [← c98097, ← c98098, ← c98099] at e_1603_4
  have e_1603_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1603 (by decide)
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
  have e_1603_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_6
  simp only [← c98103, ← c98104, ← c98105] at e_1603_6
  have e_1603_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1603 (by decide)
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
  have e_1603_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_8
  simp only [← c98109, ← c98110, ← c98111] at e_1603_8
  have e_1603_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1603 (by decide)
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
  have e_1603_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_10
  simp only [← c98115, ← c98116, ← c98117] at e_1603_10
  have e_1603_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1603 (by decide)
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
  have e_1603_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_12
  simp only [← c98121, ← c98122, ← c98123] at e_1603_12
  have e_1603_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1603 (by decide)
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
  have e_1603_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_14
  simp only [← c98127, ← c98128, ← c98129] at e_1603_14
  have e_1603_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1603 (by decide)
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
  have e_1603_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_16
  simp only [← c98133, ← c98134, ← c98135] at e_1603_16
  have e_1603_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1603 (by decide)
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
  have e_1603_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_18
  simp only [← c98139, ← c98140, ← c98141] at e_1603_18
  have e_1603_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1603 (by decide)
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
  have e_1604_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_0
  simp only [← c98145, ← c98146, ← c98147] at e_1604_0
  have e_1604_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1604 (by decide)
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
  have e_1604_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_2
  simp only [← c98151, ← c98152, ← c98153] at e_1604_2
  have e_1604_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1604 (by decide)
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
  have e_1604_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_4
  simp only [← c98157, ← c98158, ← c98159] at e_1604_4
  have e_1604_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1604 (by decide)
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
  have e_1604_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_6
  simp only [← c98163, ← c98164, ← c98165] at e_1604_6
  have e_1604_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1604 (by decide)
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
  have e_1604_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_8
  simp only [← c98169, ← c98170, ← c98171] at e_1604_8
  have e_1604_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1604 (by decide)
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
  have e_1604_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_10
  simp only [← c98175, ← c98176, ← c98177] at e_1604_10
  have e_1604_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1604 (by decide)
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
  have e_1604_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_12
  simp only [← c98181, ← c98182, ← c98183] at e_1604_12
  have e_1604_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1604 (by decide)
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
  have e_1604_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_14
  simp only [← c98187, ← c98188, ← c98189] at e_1604_14
  have e_1604_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1604 (by decide)
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
  have e_1604_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_16
  simp only [← c98193, ← c98194, ← c98195] at e_1604_16
  have e_1604_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1604 (by decide)
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
  have e_1604_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_18
  simp only [← c98199, ← c98200, ← c98201] at e_1604_18
  have e_1604_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1604 (by decide)
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
  have e_1605_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_0
  simp only [← c98205, ← c98206, ← c98207] at e_1605_0
  have e_1605_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1605 (by decide)
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
  have e_1605_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_2
  simp only [← c98211, ← c98212, ← c98213] at e_1605_2
  have e_1605_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1605 (by decide)
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
  have e_1605_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_4
  simp only [← c98217, ← c98218, ← c98219] at e_1605_4
  have e_1605_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1605 (by decide)
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
  have e_1605_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_6
  simp only [← c98223, ← c98224, ← c98225] at e_1605_6
  have e_1605_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1605 (by decide)
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
  have e_1605_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_8
  simp only [← c98229, ← c98230, ← c98231] at e_1605_8
  have e_1605_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1605 (by decide)
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
  have e_1605_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_10
  simp only [← c98235, ← c98236, ← c98237] at e_1605_10
  have e_1605_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1605 (by decide)
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
  have e_1605_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_12
  simp only [← c98241, ← c98242, ← c98243] at e_1605_12
  have e_1605_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1605 (by decide)
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
  have e_1605_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_14
  simp only [← c98247, ← c98248, ← c98249] at e_1605_14
  have e_1605_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1605 (by decide)
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
  have e_1605_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_16
  simp only [← c98253, ← c98254, ← c98255] at e_1605_16
  have e_1605_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1605 (by decide)
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
  have e_1605_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_18
  simp only [← c98259, ← c98260, ← c98261] at e_1605_18
  have e_1605_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1605 (by decide)
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
  have e_1606_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_0
  simp only [← c98265, ← c98266, ← c98267] at e_1606_0
  have e_1606_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1606 (by decide)
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
  have e_1606_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_2
  simp only [← c98271, ← c98272, ← c98273] at e_1606_2
  have e_1606_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1606 (by decide)
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
  have e_1606_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_4
  simp only [← c98277, ← c98278, ← c98279] at e_1606_4
  have e_1606_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1606 (by decide)
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
  have e_1606_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_6
  simp only [← c98283, ← c98284, ← c98285] at e_1606_6
  have e_1606_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1606 (by decide)
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
  have e_1606_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_8
  simp only [← c98289, ← c98290, ← c98291] at e_1606_8
  have e_1606_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1606 (by decide)
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
  have e_1606_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_10
  simp only [← c98295, ← c98296, ← c98297] at e_1606_10
  have e_1606_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1606 (by decide)
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
  have e_1606_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_12
  simp only [← c98301, ← c98302, ← c98303] at e_1606_12
  have e_1606_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1606 (by decide)
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
  have e_1606_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_14
  simp only [← c98307, ← c98308, ← c98309] at e_1606_14
  have e_1606_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1606 (by decide)
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
  have e_1606_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_16
  simp only [← c98313, ← c98314, ← c98315] at e_1606_16
  have e_1606_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1606 (by decide)
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
  have e_1606_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_18
  simp only [← c98319, ← c98320, ← c98321] at e_1606_18
  have e_1606_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1606 (by decide)
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
  have e_1607_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_0
  simp only [← c98325, ← c98326, ← c98327] at e_1607_0
  have e_1607_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1607 (by decide)
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
  have e_1607_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_2
  simp only [← c98331, ← c98332, ← c98333] at e_1607_2
  have e_1607_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1607 (by decide)
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
  have e_1607_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_4
  simp only [← c98337, ← c98338, ← c98339] at e_1607_4
  have e_1607_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1607 (by decide)
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
  have e_1607_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_6
  simp only [← c98343, ← c98344, ← c98345] at e_1607_6
  have e_1607_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1607 (by decide)
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
  have e_1607_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_8
  simp only [← c98349, ← c98350, ← c98351] at e_1607_8
  have e_1607_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1607 (by decide)
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
  have e_1607_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_10
  simp only [← c98355, ← c98356, ← c98357] at e_1607_10
  have e_1607_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1607 (by decide)
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
  have e_1607_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_12
  simp only [← c98361, ← c98362, ← c98363] at e_1607_12
  have e_1607_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1607 (by decide)
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
  have e_1607_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_14
  simp only [← c98367, ← c98368, ← c98369] at e_1607_14
  have e_1607_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1607 (by decide)
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
  have e_1607_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_16
  simp only [← c98373, ← c98374, ← c98375] at e_1607_16
  have e_1607_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1607 (by decide)
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
  have e_1607_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_18
  simp only [← c98379, ← c98380, ← c98381] at e_1607_18
  have e_1607_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1607 (by decide)
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
  have e_1608_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_0
  simp only [← c98385, ← c98386, ← c98387] at e_1608_0
  have e_1608_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1608 (by decide)
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
  have e_1608_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_2
  simp only [← c98391, ← c98392, ← c98393] at e_1608_2
  have e_1608_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1608 (by decide)
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
  have e_1608_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_4
  simp only [← c98397, ← c98398, ← c98399] at e_1608_4
  have e_1608_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1608 (by decide)
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
  have e_1608_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_6
  simp only [← c98403, ← c98404, ← c98405] at e_1608_6
  have e_1608_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1608 (by decide)
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
  have e_1608_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_8
  simp only [← c98409, ← c98410, ← c98411] at e_1608_8
  have e_1608_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1608 (by decide)
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
  have e_1608_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_10
  simp only [← c98415, ← c98416, ← c98417] at e_1608_10
  have e_1608_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1608 (by decide)
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
  have e_1608_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_12
  simp only [← c98421, ← c98422, ← c98423] at e_1608_12
  have e_1608_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1608 (by decide)
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
  have e_1608_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_14
  simp only [← c98427, ← c98428, ← c98429] at e_1608_14
  have e_1608_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1608 (by decide)
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
  have e_1608_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_16
  simp only [← c98433, ← c98434, ← c98435] at e_1608_16
  have e_1608_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1608 (by decide)
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
  have e_1608_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_18
  simp only [← c98439, ← c98440, ← c98441] at e_1608_18
  have e_1608_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1608 (by decide)
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
  have e_1609_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_0
  simp only [← c98445, ← c98446, ← c98447] at e_1609_0
  have e_1609_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1609 (by decide)
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
  have e_1609_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_2
  simp only [← c98451, ← c98452, ← c98453] at e_1609_2
  have e_1609_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1609 (by decide)
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
  have e_1609_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_4
  simp only [← c98457, ← c98458, ← c98459] at e_1609_4
  have e_1609_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1609 (by decide)
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
  have e_1609_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_6
  simp only [← c98463, ← c98464, ← c98465] at e_1609_6
  have e_1609_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1609 (by decide)
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
  have e_1609_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_8
  simp only [← c98469, ← c98470, ← c98471] at e_1609_8
  have e_1609_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1609 (by decide)
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
  have e_1609_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_10
  simp only [← c98475, ← c98476, ← c98477] at e_1609_10
  have e_1609_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1609 (by decide)
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
  have e_1609_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_12
  simp only [← c98481, ← c98482, ← c98483] at e_1609_12
  have e_1609_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1609 (by decide)
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
  have e_1609_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_14
  simp only [← c98487, ← c98488, ← c98489] at e_1609_14
  have e_1609_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1609 (by decide)
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
  have e_1609_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_16
  simp only [← c98493, ← c98494, ← c98495] at e_1609_16
  have e_1609_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1609 (by decide)
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
  have e_1609_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_18
  simp only [← c98499, ← c98500, ← c98501] at e_1609_18
  have e_1609_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1609 (by decide)
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
  have e_1610_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_0
  simp only [← c98505, ← c98506, ← c98507] at e_1610_0
  have e_1610_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1610 (by decide)
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
  have e_1610_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_2
  simp only [← c98511, ← c98512, ← c98513] at e_1610_2
  have e_1610_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1610 (by decide)
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
  have e_1610_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_4
  simp only [← c98517, ← c98518, ← c98519] at e_1610_4
  have e_1610_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1610 (by decide)
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
  have e_1610_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_6
  simp only [← c98523, ← c98524, ← c98525] at e_1610_6
  have e_1610_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1610 (by decide)
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
  have e_1610_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_8
  simp only [← c98529, ← c98530, ← c98531] at e_1610_8
  have e_1610_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1610 (by decide)
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
  have e_1610_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_10
  simp only [← c98535, ← c98536, ← c98537] at e_1610_10
  have e_1610_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1610 (by decide)
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
  have e_1610_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_12
  simp only [← c98541, ← c98542, ← c98543] at e_1610_12
  have e_1610_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1610 (by decide)
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
  have e_1610_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_14
  simp only [← c98547, ← c98548, ← c98549] at e_1610_14
  have e_1610_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1610 (by decide)
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
  have e_1610_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_16
  simp only [← c98553, ← c98554, ← c98555] at e_1610_16
  have e_1610_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1610 (by decide)
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
  have e_1610_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_18
  simp only [← c98559, ← c98560, ← c98561] at e_1610_18
  have e_1610_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1610 (by decide)
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
  have e_1611_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_0
  simp only [← c98565, ← c98566, ← c98567] at e_1611_0
  have e_1611_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1611 (by decide)
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
  have e_1611_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_2
  simp only [← c98571, ← c98572, ← c98573] at e_1611_2
  have e_1611_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1611 (by decide)
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
  have e_1611_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_4
  simp only [← c98577, ← c98578, ← c98579] at e_1611_4
  have e_1611_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1611 (by decide)
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
  have e_1611_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_6
  simp only [← c98583, ← c98584, ← c98585] at e_1611_6
  have e_1611_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1611 (by decide)
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
  have e_1611_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_8
  simp only [← c98589, ← c98590, ← c98591] at e_1611_8
  have e_1611_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1611 (by decide)
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
  have e_1611_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_10
  simp only [← c98595, ← c98596, ← c98597] at e_1611_10
  have e_1611_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1611 (by decide)
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
  have e_1611_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_12
  simp only [← c98601, ← c98602, ← c98603] at e_1611_12
  have e_1611_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1611 (by decide)
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
  have e_1611_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_14
  simp only [← c98607, ← c98608, ← c98609] at e_1611_14
  have e_1611_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1611 (by decide)
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
  have e_1611_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_16
  simp only [← c98613, ← c98614, ← c98615] at e_1611_16
  have e_1611_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1611 (by decide)
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
  have e_1611_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_18
  simp only [← c98619, ← c98620, ← c98621] at e_1611_18
  have e_1611_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1611 (by decide)
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
  have e_1612_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_0
  simp only [← c98625, ← c98626, ← c98627] at e_1612_0
  have e_1612_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1612 (by decide)
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
  have e_1612_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_2
  simp only [← c98631, ← c98632, ← c98633] at e_1612_2
  have e_1612_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1612 (by decide)
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
  have e_1612_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_4
  simp only [← c98637, ← c98638, ← c98639] at e_1612_4
  have e_1612_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1612 (by decide)
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
  have e_1612_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_6
  simp only [← c98643, ← c98644, ← c98645] at e_1612_6
  have e_1612_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1612 (by decide)
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
  have e_1612_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_8
  simp only [← c98649, ← c98650, ← c98651] at e_1612_8
  have e_1612_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1612 (by decide)
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
  have e_1612_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_10
  simp only [← c98655, ← c98656, ← c98657] at e_1612_10
  have e_1612_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1612 (by decide)
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
  have e_1612_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_12
  simp only [← c98661, ← c98662, ← c98663] at e_1612_12
  have e_1612_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1612 (by decide)
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
  have e_1612_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_14
  simp only [← c98667, ← c98668, ← c98669] at e_1612_14
  have e_1612_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1612 (by decide)
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
  have e_1612_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_16
  simp only [← c98673, ← c98674, ← c98675] at e_1612_16
  have e_1612_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1612 (by decide)
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
  have e_1612_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_18
  simp only [← c98679, ← c98680, ← c98681] at e_1612_18
  have e_1612_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1612 (by decide)
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
  have e_1613_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_0
  simp only [← c98685, ← c98686, ← c98687] at e_1613_0
  have e_1613_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1613 (by decide)
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
  have e_1613_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_2
  simp only [← c98691, ← c98692, ← c98693] at e_1613_2
  have e_1613_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1613 (by decide)
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
  have e_1613_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_4
  simp only [← c98697, ← c98698, ← c98699] at e_1613_4
  have e_1613_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1613 (by decide)
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
  have e_1613_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_6
  simp only [← c98703, ← c98704, ← c98705] at e_1613_6
  have e_1613_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1613 (by decide)
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
  have e_1613_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_8
  simp only [← c98709, ← c98710, ← c98711] at e_1613_8
  have e_1613_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1613 (by decide)
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
  have e_1613_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_10
  simp only [← c98715, ← c98716, ← c98717] at e_1613_10
  have e_1613_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1613 (by decide)
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
  have e_1613_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_12
  simp only [← c98721, ← c98722, ← c98723] at e_1613_12
  have e_1613_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1613 (by decide)
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
  have e_1613_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_14
  simp only [← c98727, ← c98728, ← c98729] at e_1613_14
  have e_1613_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1613 (by decide)
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
  have e_1613_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_16
  simp only [← c98733, ← c98734, ← c98735] at e_1613_16
  have e_1613_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1613 (by decide)
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
  have e_1613_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_18
  simp only [← c98739, ← c98740, ← c98741] at e_1613_18
  have e_1613_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1613 (by decide)
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
  have e_1614_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_0
  simp only [← c98745, ← c98746, ← c98747] at e_1614_0
  have e_1614_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1614 (by decide)
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
  have e_1614_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_2
  simp only [← c98751, ← c98752, ← c98753] at e_1614_2
  have e_1614_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1614 (by decide)
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
  have e_1614_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_4
  simp only [← c98757, ← c98758, ← c98759] at e_1614_4
  have e_1614_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1614 (by decide)
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
  have e_1614_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_6
  simp only [← c98763, ← c98764, ← c98765] at e_1614_6
  have e_1614_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1614 (by decide)
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
  have e_1614_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_8
  simp only [← c98769, ← c98770, ← c98771] at e_1614_8
  have e_1614_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1614 (by decide)
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
  have e_1614_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_10
  simp only [← c98775, ← c98776, ← c98777] at e_1614_10
  have e_1614_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1614 (by decide)
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
  have e_1614_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_12
  simp only [← c98781, ← c98782, ← c98783] at e_1614_12
  have e_1614_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1614 (by decide)
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
  have e_1614_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_14
  simp only [← c98787, ← c98788, ← c98789] at e_1614_14
  have e_1614_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1614 (by decide)
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
  have e_1614_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_16
  simp only [← c98793, ← c98794, ← c98795] at e_1614_16
  have e_1614_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1614 (by decide)
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
  have e_1614_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_18
  simp only [← c98799, ← c98800, ← c98801] at e_1614_18
  have e_1614_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1614 (by decide)
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
  have e_1615_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_0
  simp only [← c98805, ← c98806, ← c98807] at e_1615_0
  have e_1615_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1615 (by decide)
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
  have e_1615_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_2
  simp only [← c98811, ← c98812, ← c98813] at e_1615_2
  have e_1615_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1615 (by decide)
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
  have e_1615_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_4
  simp only [← c98817, ← c98818, ← c98819] at e_1615_4
  have e_1615_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1615 (by decide)
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
  have e_1615_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_6
  simp only [← c98823, ← c98824, ← c98825] at e_1615_6
  have e_1615_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1615 (by decide)
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
  have e_1615_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_8
  simp only [← c98829, ← c98830, ← c98831] at e_1615_8
  have e_1615_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1615 (by decide)
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
  have e_1615_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_10
  simp only [← c98835, ← c98836, ← c98837] at e_1615_10
  have e_1615_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1615 (by decide)
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
  have e_1615_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_12
  simp only [← c98841, ← c98842, ← c98843] at e_1615_12
  have e_1615_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1615 (by decide)
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
  have e_1615_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_14
  simp only [← c98847, ← c98848, ← c98849] at e_1615_14
  have e_1615_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1615 (by decide)
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
  have e_1615_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_16
  simp only [← c98853, ← c98854, ← c98855] at e_1615_16
  have e_1615_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1615 (by decide)
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
  have e_1615_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_18
  simp only [← c98859, ← c98860, ← c98861] at e_1615_18
  have e_1615_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1615 (by decide)
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
  have e_1616_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_0
  simp only [← c98865, ← c98866, ← c98867] at e_1616_0
  have e_1616_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1616 (by decide)
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
  have e_1616_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_2
  simp only [← c98871, ← c98872, ← c98873] at e_1616_2
  have e_1616_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1616 (by decide)
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
  have e_1616_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_4
  simp only [← c98877, ← c98878, ← c98879] at e_1616_4
  have e_1616_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1616 (by decide)
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
  have e_1616_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_6
  simp only [← c98883, ← c98884, ← c98885] at e_1616_6
  have e_1616_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1616 (by decide)
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
  have e_1616_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_8
  simp only [← c98889, ← c98890, ← c98891] at e_1616_8
  have e_1616_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1616 (by decide)
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
  have e_1616_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_10
  simp only [← c98895, ← c98896, ← c98897] at e_1616_10
  have e_1616_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1616 (by decide)
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
  have e_1616_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_12
  simp only [← c98901, ← c98902, ← c98903] at e_1616_12
  have e_1616_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1616 (by decide)
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
  have e_1616_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_14
  simp only [← c98907, ← c98908, ← c98909] at e_1616_14
  have e_1616_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1616 (by decide)
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
  have e_1616_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_16
  simp only [← c98913, ← c98914, ← c98915] at e_1616_16
  have e_1616_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1616 (by decide)
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
  have e_1616_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_18
  simp only [← c98919, ← c98920, ← c98921] at e_1616_18
  have e_1616_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1616 (by decide)
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
  have e_1617_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_0
  simp only [← c98925, ← c98926, ← c98927] at e_1617_0
  have e_1617_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1617 (by decide)
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
  have e_1617_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_2
  simp only [← c98931, ← c98932, ← c98933] at e_1617_2
  have e_1617_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1617 (by decide)
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
  have e_1617_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_4
  simp only [← c98937, ← c98938, ← c98939] at e_1617_4
  have e_1617_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1617 (by decide)
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
  have e_1617_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_6
  simp only [← c98943, ← c98944, ← c98945] at e_1617_6
  have e_1617_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1617 (by decide)
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
  have e_1617_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_8
  simp only [← c98949, ← c98950, ← c98951] at e_1617_8
  have e_1617_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1617 (by decide)
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
  have e_1617_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_10
  simp only [← c98955, ← c98956, ← c98957] at e_1617_10
  have e_1617_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1617 (by decide)
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
  have e_1617_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_12
  simp only [← c98961, ← c98962, ← c98963] at e_1617_12
  have e_1617_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1617 (by decide)
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
  have e_1617_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_14
  simp only [← c98967, ← c98968, ← c98969] at e_1617_14
  have e_1617_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1617 (by decide)
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
  have e_1617_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_16
  simp only [← c98973, ← c98974, ← c98975] at e_1617_16
  have e_1617_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1617 (by decide)
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
  have e_1617_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_18
  simp only [← c98979, ← c98980, ← c98981] at e_1617_18
  have e_1617_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1617 (by decide)
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
  have e_1618_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_0
  simp only [← c98985, ← c98986, ← c98987] at e_1618_0
  have e_1618_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1618 (by decide)
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
  have e_1618_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_2
  simp only [← c98991, ← c98992, ← c98993] at e_1618_2
  have e_1618_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1618 (by decide)
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
  have e_1618_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_4
  simp only [← c98997, ← c98998, ← c98999] at e_1618_4
  have e_1618_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1618 (by decide)
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
  have e_1618_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_6
  simp only [← c99003, ← c99004, ← c99005] at e_1618_6
  have e_1618_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1618 (by decide)
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
  have e_1618_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_8
  simp only [← c99009, ← c99010, ← c99011] at e_1618_8
  have e_1618_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1618 (by decide)
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
  have e_1618_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_10
  simp only [← c99015, ← c99016, ← c99017] at e_1618_10
  have e_1618_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1618 (by decide)
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
  have e_1618_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_12
  simp only [← c99021, ← c99022, ← c99023] at e_1618_12
  have e_1618_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1618 (by decide)
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
  have e_1618_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_14
  simp only [← c99027, ← c99028, ← c99029] at e_1618_14
  have e_1618_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1618 (by decide)
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
  have e_1618_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_16
  simp only [← c99033, ← c99034, ← c99035] at e_1618_16
  have e_1618_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1618 (by decide)
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
  have e_1618_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_18
  simp only [← c99039, ← c99040, ← c99041] at e_1618_18
  have e_1618_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1618 (by decide)
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
  have e_1619_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_0
  simp only [← c99045, ← c99046, ← c99047] at e_1619_0
  have e_1619_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1619 (by decide)
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
  have e_1619_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_2
  simp only [← c99051, ← c99052, ← c99053] at e_1619_2
  have e_1619_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1619 (by decide)
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
  have e_1619_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_4
  simp only [← c99057, ← c99058, ← c99059] at e_1619_4
  have e_1619_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1619 (by decide)
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
  have e_1619_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_6
  simp only [← c99063, ← c99064, ← c99065] at e_1619_6
  have e_1619_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1619 (by decide)
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
  have e_1619_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_8
  simp only [← c99069, ← c99070, ← c99071] at e_1619_8
  have e_1619_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1619 (by decide)
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
  have e_1619_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_10
  simp only [← c99075, ← c99076, ← c99077] at e_1619_10
  have e_1619_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1619 (by decide)
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
  have e_1619_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_12
  simp only [← c99081, ← c99082, ← c99083] at e_1619_12
  have e_1619_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1619 (by decide)
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
  have e_1619_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_14
  simp only [← c99087, ← c99088, ← c99089] at e_1619_14
  have e_1619_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1619 (by decide)
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
  have e_1619_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_16
  simp only [← c99093, ← c99094, ← c99095] at e_1619_16
  have e_1619_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1619 (by decide)
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
  have e_1619_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_18
  simp only [← c99099, ← c99100, ← c99101] at e_1619_18
  have e_1619_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1619 (by decide)
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
  have e_1620_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_0
  simp only [← c99105, ← c99106, ← c99107] at e_1620_0
  have e_1620_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1620 (by decide)
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
  have e_1620_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_2
  simp only [← c99111, ← c99112, ← c99113] at e_1620_2
  have e_1620_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1620 (by decide)
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
  have e_1620_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_4
  simp only [← c99117, ← c99118, ← c99119] at e_1620_4
  have e_1620_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1620 (by decide)
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
  have e_1620_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_6
  simp only [← c99123, ← c99124, ← c99125] at e_1620_6
  have e_1620_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1620 (by decide)
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
  have e_1620_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_8
  simp only [← c99129, ← c99130, ← c99131] at e_1620_8
  have e_1620_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1620 (by decide)
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
  have e_1620_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_10
  simp only [← c99135, ← c99136, ← c99137] at e_1620_10
  have e_1620_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1620 (by decide)
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
  have e_1620_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_12
  simp only [← c99141, ← c99142, ← c99143] at e_1620_12
  have e_1620_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1620 (by decide)
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
  have e_1620_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_14
  simp only [← c99147, ← c99148, ← c99149] at e_1620_14
  have e_1620_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1620 (by decide)
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
  have e_1620_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_16
  simp only [← c99153, ← c99154, ← c99155] at e_1620_16
  have e_1620_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1620 (by decide)
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
  have e_1620_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_18
  simp only [← c99159, ← c99160, ← c99161] at e_1620_18
  have e_1620_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1620 (by decide)
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
  have e_1621_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_0
  simp only [← c99165, ← c99166, ← c99167] at e_1621_0
  have e_1621_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1621 (by decide)
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
  have e_1621_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_2
  simp only [← c99171, ← c99172, ← c99173] at e_1621_2
  have e_1621_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1621 (by decide)
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
  have e_1621_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_4
  simp only [← c99177, ← c99178, ← c99179] at e_1621_4
  have e_1621_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1621 (by decide)
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
  have e_1621_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_6
  simp only [← c99183, ← c99184, ← c99185] at e_1621_6
  have e_1621_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1621 (by decide)
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
  have e_1621_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_8
  simp only [← c99189, ← c99190, ← c99191] at e_1621_8
  have e_1621_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1621 (by decide)
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
  have e_1621_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_10
  simp only [← c99195, ← c99196, ← c99197] at e_1621_10
  have e_1621_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1621 (by decide)
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
  have e_1621_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_12
  simp only [← c99201, ← c99202, ← c99203] at e_1621_12
  have e_1621_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1621 (by decide)
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
  have e_1621_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_14
  simp only [← c99207, ← c99208, ← c99209] at e_1621_14
  have e_1621_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1621 (by decide)
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
  have e_1621_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_16
  simp only [← c99213, ← c99214, ← c99215] at e_1621_16
  have e_1621_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1621 (by decide)
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
  have e_1621_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_18
  simp only [← c99219, ← c99220, ← c99221] at e_1621_18
  have e_1621_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1621 (by decide)
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
  have e_1622_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_0
  simp only [← c99225, ← c99226, ← c99227] at e_1622_0
  have e_1622_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1622 (by decide)
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
  have e_1622_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_2
  simp only [← c99231, ← c99232, ← c99233] at e_1622_2
  have e_1622_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1622 (by decide)
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
  have e_1622_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_4
  simp only [← c99237, ← c99238, ← c99239] at e_1622_4
  have e_1622_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1622 (by decide)
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
  have e_1622_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_6
  simp only [← c99243, ← c99244, ← c99245] at e_1622_6
  have e_1622_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1622 (by decide)
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
  have e_1622_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_8
  simp only [← c99249, ← c99250, ← c99251] at e_1622_8
  have e_1622_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1622 (by decide)
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
  have e_1622_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_10
  simp only [← c99255, ← c99256, ← c99257] at e_1622_10
  have e_1622_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1622 (by decide)
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
  have e_1622_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_12
  simp only [← c99261, ← c99262, ← c99263] at e_1622_12
  have e_1622_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1622 (by decide)
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
  have e_1622_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_14
  simp only [← c99267, ← c99268, ← c99269] at e_1622_14
  have e_1622_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1622 (by decide)
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
  have e_1622_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_16
  simp only [← c99273, ← c99274, ← c99275] at e_1622_16
  have e_1622_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1622 (by decide)
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
  have e_1622_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_18
  simp only [← c99279, ← c99280, ← c99281] at e_1622_18
  have e_1622_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1622 (by decide)
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
  have e_1623_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_0
  simp only [← c99285, ← c99286, ← c99287] at e_1623_0
  have e_1623_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1623 (by decide)
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
  have e_1623_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_2
  simp only [← c99291, ← c99292, ← c99293] at e_1623_2
  have e_1623_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1623 (by decide)
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
  have e_1623_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_4
  simp only [← c99297, ← c99298, ← c99299] at e_1623_4
  have e_1623_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1623 (by decide)
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
  have e_1623_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_6
  simp only [← c99303, ← c99304, ← c99305] at e_1623_6
  have e_1623_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1623 (by decide)
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
  have e_1623_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_8
  simp only [← c99309, ← c99310, ← c99311] at e_1623_8
  have e_1623_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1623 (by decide)
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
  have e_1623_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_10
  simp only [← c99315, ← c99316, ← c99317] at e_1623_10
  have e_1623_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1623 (by decide)
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
  have e_1623_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_12
  simp only [← c99321, ← c99322, ← c99323] at e_1623_12
  have e_1623_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1623 (by decide)
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
  have e_1623_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_14
  simp only [← c99327, ← c99328, ← c99329] at e_1623_14
  have e_1623_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1623 (by decide)
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
  have e_1623_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_16
  simp only [← c99333, ← c99334, ← c99335] at e_1623_16
  have e_1623_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1623 (by decide)
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
  have e_1623_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_18
  simp only [← c99339, ← c99340, ← c99341] at e_1623_18
  have e_1623_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1623 (by decide)
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
  have e_1624_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_0
  simp only [← c99345, ← c99346, ← c99347] at e_1624_0
  have e_1624_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1624 (by decide)
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
  have e_1624_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_2
  simp only [← c99351, ← c99352, ← c99353] at e_1624_2
  have e_1624_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1624 (by decide)
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
  have e_1624_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_4
  simp only [← c99357, ← c99358, ← c99359] at e_1624_4
  have e_1624_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1624 (by decide)
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
  have e_1624_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_6
  simp only [← c99363, ← c99364, ← c99365] at e_1624_6
  have e_1624_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1624 (by decide)
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
  have e_1624_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_8
  simp only [← c99369, ← c99370, ← c99371] at e_1624_8
  have e_1624_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1624 (by decide)
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
  have e_1624_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_10
  simp only [← c99375, ← c99376, ← c99377] at e_1624_10
  have e_1624_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1624 (by decide)
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
  have e_1624_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_12
  simp only [← c99381, ← c99382, ← c99383] at e_1624_12
  have e_1624_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1624 (by decide)
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
  have e_1624_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_14
  simp only [← c99387, ← c99388, ← c99389] at e_1624_14
  have e_1624_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1624 (by decide)
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
  have e_1624_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_16
  simp only [← c99393, ← c99394, ← c99395] at e_1624_16
  have e_1624_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1624 (by decide)
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
  have e_1624_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_18
  simp only [← c99399, ← c99400, ← c99401] at e_1624_18
  have e_1624_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1624 (by decide)
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
  have e_1625_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_0
  simp only [← c99405, ← c99406, ← c99407] at e_1625_0
  have e_1625_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1625 (by decide)
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
  have e_1625_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_2
  simp only [← c99411, ← c99412, ← c99413] at e_1625_2
  have e_1625_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1625 (by decide)
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
  have e_1625_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_4
  simp only [← c99417, ← c99418, ← c99419] at e_1625_4
  have e_1625_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1625 (by decide)
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
  have e_1625_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_6
  simp only [← c99423, ← c99424, ← c99425] at e_1625_6
  have e_1625_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1625 (by decide)
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
  have e_1625_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_8
  simp only [← c99429, ← c99430, ← c99431] at e_1625_8
  have e_1625_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1625 (by decide)
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
  have e_1625_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_10
  simp only [← c99435, ← c99436, ← c99437] at e_1625_10
  have e_1625_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1625 (by decide)
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
  have e_1625_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_12
  simp only [← c99441, ← c99442, ← c99443] at e_1625_12
  have e_1625_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1625 (by decide)
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
  have e_1625_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_14
  simp only [← c99447, ← c99448, ← c99449] at e_1625_14
  have e_1625_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1625 (by decide)
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
  have e_1625_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_16
  simp only [← c99453, ← c99454, ← c99455] at e_1625_16
  have e_1625_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1625 (by decide)
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
  have e_1625_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_18
  simp only [← c99459, ← c99460, ← c99461] at e_1625_18
  have e_1625_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1625 (by decide)
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
  have e_1626_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_0
  simp only [← c99465, ← c99466, ← c99467] at e_1626_0
  have e_1626_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1626 (by decide)
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
  have e_1626_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_2
  simp only [← c99471, ← c99472, ← c99473] at e_1626_2
  have e_1626_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1626 (by decide)
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
  have e_1626_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_4
  simp only [← c99477, ← c99478, ← c99479] at e_1626_4
  have e_1626_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1626 (by decide)
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
  have e_1626_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_6
  simp only [← c99483, ← c99484, ← c99485] at e_1626_6
  have e_1626_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1626 (by decide)
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
  have e_1626_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_8
  simp only [← c99489, ← c99490, ← c99491] at e_1626_8
  have e_1626_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1626 (by decide)
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
  have e_1626_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_10
  simp only [← c99495, ← c99496, ← c99497] at e_1626_10
  have e_1626_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1626 (by decide)
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
  have e_1626_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_12
  simp only [← c99501, ← c99502, ← c99503] at e_1626_12
  have e_1626_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1626 (by decide)
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
  have e_1626_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_14
  simp only [← c99507, ← c99508, ← c99509] at e_1626_14
  have e_1626_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1626 (by decide)
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
  have e_1626_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_16
  simp only [← c99513, ← c99514, ← c99515] at e_1626_16
  have e_1626_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1626 (by decide)
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
  have e_1626_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_18
  simp only [← c99519, ← c99520, ← c99521] at e_1626_18
  have e_1626_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1626 (by decide)
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
  have e_1627_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_0
  simp only [← c99525, ← c99526, ← c99527] at e_1627_0
  have e_1627_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1627 (by decide)
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
  have e_1627_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_2
  simp only [← c99531, ← c99532, ← c99533] at e_1627_2
  have e_1627_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1627 (by decide)
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
  have e_1627_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_4
  simp only [← c99537, ← c99538, ← c99539] at e_1627_4
  have e_1627_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1627 (by decide)
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
  have e_1627_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_6
  simp only [← c99543, ← c99544, ← c99545] at e_1627_6
  have e_1627_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1627 (by decide)
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
  have e_1627_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_8
  simp only [← c99549, ← c99550, ← c99551] at e_1627_8
  have e_1627_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1627 (by decide)
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
  have e_1627_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_10
  simp only [← c99555, ← c99556, ← c99557] at e_1627_10
  have e_1627_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1627 (by decide)
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
  have e_1627_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_12
  simp only [← c99561, ← c99562, ← c99563] at e_1627_12
  have e_1627_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1627 (by decide)
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
  have e_1627_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_14
  simp only [← c99567, ← c99568, ← c99569] at e_1627_14
  have e_1627_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1627 (by decide)
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
  have e_1627_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_16
  simp only [← c99573, ← c99574, ← c99575] at e_1627_16
  have e_1627_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1627 (by decide)
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
  have e_1627_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_18
  simp only [← c99579, ← c99580, ← c99581] at e_1627_18
  have e_1627_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1627 (by decide)
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
  have e_1628_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_0
  simp only [← c99585, ← c99586, ← c99587] at e_1628_0
  have e_1628_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1628 (by decide)
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
  have e_1628_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_2
  simp only [← c99591, ← c99592, ← c99593] at e_1628_2
  have e_1628_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1628 (by decide)
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
  have e_1628_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_4
  simp only [← c99597, ← c99598, ← c99599] at e_1628_4
  have e_1628_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1628 (by decide)
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
  have e_1628_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_6
  simp only [← c99603, ← c99604, ← c99605] at e_1628_6
  have e_1628_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1628 (by decide)
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
  have e_1628_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_8
  simp only [← c99609, ← c99610, ← c99611] at e_1628_8
  have e_1628_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1628 (by decide)
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
  have e_1628_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_10
  simp only [← c99615, ← c99616, ← c99617] at e_1628_10
  have e_1628_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1628 (by decide)
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
  have e_1628_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_12
  simp only [← c99621, ← c99622, ← c99623] at e_1628_12
  have e_1628_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1628 (by decide)
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
  have e_1628_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_14
  simp only [← c99627, ← c99628, ← c99629] at e_1628_14
  have e_1628_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1628 (by decide)
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
  have e_1628_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_16
  simp only [← c99633, ← c99634, ← c99635] at e_1628_16
  have e_1628_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1628 (by decide)
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
  have e_1628_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_18
  simp only [← c99639, ← c99640, ← c99641] at e_1628_18
  have e_1628_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1628 (by decide)
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
  have e_1629_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_0
  simp only [← c99645, ← c99646, ← c99647] at e_1629_0
  have e_1629_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1629 (by decide)
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
  have e_1629_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_2
  simp only [← c99651, ← c99652, ← c99653] at e_1629_2
  have e_1629_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1629 (by decide)
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
  have e_1629_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_4
  simp only [← c99657, ← c99658, ← c99659] at e_1629_4
  have e_1629_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1629 (by decide)
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
  have e_1629_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_6
  simp only [← c99663, ← c99664, ← c99665] at e_1629_6
  have e_1629_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1629 (by decide)
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
  have e_1629_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_8
  simp only [← c99669, ← c99670, ← c99671] at e_1629_8
  have e_1629_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1629 (by decide)
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
  have e_1629_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_10
  simp only [← c99675, ← c99676, ← c99677] at e_1629_10
  have e_1629_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1629 (by decide)
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
  have e_1629_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_12
  simp only [← c99681, ← c99682, ← c99683] at e_1629_12
  have e_1629_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1629 (by decide)
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
  have e_1629_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_14
  simp only [← c99687, ← c99688, ← c99689] at e_1629_14
  have e_1629_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1629 (by decide)
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
  have e_1629_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_16
  simp only [← c99693, ← c99694, ← c99695] at e_1629_16
  have e_1629_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1629 (by decide)
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
  have e_1629_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_18
  simp only [← c99699, ← c99700, ← c99701] at e_1629_18
  have e_1629_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1629 (by decide)
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
  have e_1630_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_0
  simp only [← c99705, ← c99706, ← c99707] at e_1630_0
  have e_1630_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1630 (by decide)
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
  have e_1630_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_2
  simp only [← c99711, ← c99712, ← c99713] at e_1630_2
  have e_1630_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1630 (by decide)
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
  have e_1630_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_4
  simp only [← c99717, ← c99718, ← c99719] at e_1630_4
  have e_1630_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1630 (by decide)
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
  have e_1630_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_6
  simp only [← c99723, ← c99724, ← c99725] at e_1630_6
  have e_1630_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1630 (by decide)
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
  have e_1630_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_8
  simp only [← c99729, ← c99730, ← c99731] at e_1630_8
  have e_1630_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1630 (by decide)
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
  have e_1630_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_10
  simp only [← c99735, ← c99736, ← c99737] at e_1630_10
  have e_1630_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1630 (by decide)
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
  have e_1630_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_12
  simp only [← c99741, ← c99742, ← c99743] at e_1630_12
  have e_1630_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1630 (by decide)
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
  have e_1630_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_14
  simp only [← c99747, ← c99748, ← c99749] at e_1630_14
  have e_1630_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1630 (by decide)
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
  have e_1630_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_16
  simp only [← c99753, ← c99754, ← c99755] at e_1630_16
  have e_1630_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1630 (by decide)
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
  have e_1630_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_18
  simp only [← c99759, ← c99760, ← c99761] at e_1630_18
  have e_1630_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1630 (by decide)
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
  have e_1631_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_0
  simp only [← c99765, ← c99766, ← c99767] at e_1631_0
  have e_1631_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1631 (by decide)
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
  have e_1631_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_2
  simp only [← c99771, ← c99772, ← c99773] at e_1631_2
  have e_1631_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1631 (by decide)
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
  have e_1631_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_4
  simp only [← c99777, ← c99778, ← c99779] at e_1631_4
  have e_1631_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1631 (by decide)
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
  have e_1631_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_6
  simp only [← c99783, ← c99784, ← c99785] at e_1631_6
  have e_1631_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1631 (by decide)
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
  have e_1631_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_8
  simp only [← c99789, ← c99790, ← c99791] at e_1631_8
  have e_1631_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1631 (by decide)
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
  have e_1631_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_10
  simp only [← c99795, ← c99796, ← c99797] at e_1631_10
  have e_1631_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1631 (by decide)
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
  have e_1631_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_12
  simp only [← c99801, ← c99802, ← c99803] at e_1631_12
  have e_1631_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1631 (by decide)
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
  have e_1631_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_14
  simp only [← c99807, ← c99808, ← c99809] at e_1631_14
  have e_1631_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1631 (by decide)
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
  have e_1631_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_16
  simp only [← c99813, ← c99814, ← c99815] at e_1631_16
  have e_1631_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1631 (by decide)
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
  have e_1631_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_18
  simp only [← c99819, ← c99820, ← c99821] at e_1631_18
  have e_1631_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1631 (by decide)
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
  have e_1632_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_0
  simp only [← c99825, ← c99826, ← c99827] at e_1632_0
  have e_1632_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1632 (by decide)
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
  have e_1632_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_2
  simp only [← c99831, ← c99832, ← c99833] at e_1632_2
  have e_1632_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1632 (by decide)
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
  have e_1632_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_4
  simp only [← c99837, ← c99838, ← c99839] at e_1632_4
  have e_1632_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1632 (by decide)
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
  have e_1632_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_6
  simp only [← c99843, ← c99844, ← c99845] at e_1632_6
  have e_1632_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1632 (by decide)
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
  have e_1632_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_8
  simp only [← c99849, ← c99850, ← c99851] at e_1632_8
  have e_1632_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1632 (by decide)
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
  have e_1632_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_10
  simp only [← c99855, ← c99856, ← c99857] at e_1632_10
  have e_1632_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1632 (by decide)
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
  have e_1632_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_12
  simp only [← c99861, ← c99862, ← c99863] at e_1632_12
  have e_1632_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1632 (by decide)
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
  have e_1632_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_14
  simp only [← c99867, ← c99868, ← c99869] at e_1632_14
  have e_1632_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1632 (by decide)
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
  have e_1632_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_16
  simp only [← c99873, ← c99874, ← c99875] at e_1632_16
  have e_1632_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1632 (by decide)
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
  have e_1632_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_18
  simp only [← c99879, ← c99880, ← c99881] at e_1632_18
  have e_1632_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1632 (by decide)
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
  have e_1633_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_0
  simp only [← c99885, ← c99886, ← c99887] at e_1633_0
  have e_1633_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1633 (by decide)
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
  have e_1633_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_2
  simp only [← c99891, ← c99892, ← c99893] at e_1633_2
  have e_1633_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1633 (by decide)
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
  have e_1633_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_4
  simp only [← c99897, ← c99898, ← c99899] at e_1633_4
  have e_1633_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1633 (by decide)
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
  have e_1633_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_6
  simp only [← c99903, ← c99904, ← c99905] at e_1633_6
  have e_1633_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1633 (by decide)
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
  have e_1633_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_8
  simp only [← c99909, ← c99910, ← c99911] at e_1633_8
  have e_1633_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1633 (by decide)
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
  have e_1633_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_10
  simp only [← c99915, ← c99916, ← c99917] at e_1633_10
  have e_1633_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1633 (by decide)
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
  have e_1633_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_12
  simp only [← c99921, ← c99922, ← c99923] at e_1633_12
  have e_1633_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1633 (by decide)
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
  have e_1633_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_14
  simp only [← c99927, ← c99928, ← c99929] at e_1633_14
  have e_1633_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1633 (by decide)
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
  have e_1633_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_16
  simp only [← c99933, ← c99934, ← c99935] at e_1633_16
  have e_1633_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1633 (by decide)
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
  have e_1633_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_18
  simp only [← c99939, ← c99940, ← c99941] at e_1633_18
  have e_1633_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1633 (by decide)
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
  have e_1634_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_0
  simp only [← c99945, ← c99946, ← c99947] at e_1634_0
  have e_1634_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1634 (by decide)
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
  have e_1634_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_2
  simp only [← c99951, ← c99952, ← c99953] at e_1634_2
  have e_1634_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1634 (by decide)
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
  have e_1634_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_4
  simp only [← c99957, ← c99958, ← c99959] at e_1634_4
  have e_1634_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1634 (by decide)
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
  have e_1634_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_6
  simp only [← c99963, ← c99964, ← c99965] at e_1634_6
  have e_1634_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1634 (by decide)
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
  have e_1634_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_8
  simp only [← c99969, ← c99970, ← c99971] at e_1634_8
  have e_1634_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1634 (by decide)
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
  have e_1634_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_10
  simp only [← c99975, ← c99976, ← c99977] at e_1634_10
  have e_1634_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1634 (by decide)
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
  have e_1634_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_12
  simp only [← c99981, ← c99982, ← c99983] at e_1634_12
  have e_1634_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1634 (by decide)
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
  have e_1634_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_14
  simp only [← c99987, ← c99988, ← c99989] at e_1634_14
  have e_1634_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1634 (by decide)
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
  have e_1634_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_16
  simp only [← c99993, ← c99994, ← c99995] at e_1634_16
  have e_1634_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1634 (by decide)
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
  have e_1634_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_18
  simp only [← c99999, ← c100000, ← c100001] at e_1634_18
  have e_1634_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1634 (by decide)
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
  have e_1635_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_0
  simp only [← c100005, ← c100006, ← c100007] at e_1635_0
  have e_1635_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1635 (by decide)
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
  have e_1635_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_2
  simp only [← c100011, ← c100012, ← c100013] at e_1635_2
  have e_1635_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1635 (by decide)
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
  have e_1635_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_4
  simp only [← c100017, ← c100018, ← c100019] at e_1635_4
  have e_1635_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1635 (by decide)
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
  have e_1635_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_6
  simp only [← c100023, ← c100024, ← c100025] at e_1635_6
  have e_1635_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1635 (by decide)
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
  have e_1635_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_8
  simp only [← c100029, ← c100030, ← c100031] at e_1635_8
  have e_1635_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1635 (by decide)
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
  have e_1635_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_10
  simp only [← c100035, ← c100036, ← c100037] at e_1635_10
  have e_1635_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1635 (by decide)
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
  have e_1635_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_12
  simp only [← c100041, ← c100042, ← c100043] at e_1635_12
  have e_1635_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1635 (by decide)
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
  have e_1635_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_14
  simp only [← c100047, ← c100048, ← c100049] at e_1635_14
  have e_1635_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1635 (by decide)
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
  have e_1635_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_16
  simp only [← c100053, ← c100054, ← c100055] at e_1635_16
  have e_1635_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1635 (by decide)
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
  have e_1635_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_18
  simp only [← c100059, ← c100060, ← c100061] at e_1635_18
  have e_1635_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1635 (by decide)
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
  have e_1636_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_0
  simp only [← c100065, ← c100066, ← c100067] at e_1636_0
  have e_1636_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_1
  simp only [← c100068, ← c100069, k2, ← c100070] at e_1636_1
  have hr := e_1636_1
  simp only [e_1636_0] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
