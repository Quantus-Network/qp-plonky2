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

theorem publicBatchWrapper89_f18991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10799)) := by
  have c118452 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118453 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118454 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118455 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118456 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118457 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_0
  simp only [← c118452, ← c118453, ← c118454] at e_1941_0
  have e_1941_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_1
  simp only [← c118455, ← c118456, k2, ← c118457] at e_1941_1
  have hr := e_1941_1
  simp only [e_1941_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10800)) := by
  have c118458 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118459 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118460 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118461 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118462 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118463 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_2
  simp only [← c118458, ← c118459, ← c118460] at e_1941_2
  have e_1941_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_3
  simp only [← c118461, ← c118462, k2, ← c118463] at e_1941_3
  have hr := e_1941_3
  simp only [e_1941_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10801)) := by
  have c118464 := (publicBatchWrapper89_copies3702 a h).1
  have c118465 := (publicBatchWrapper89_copies3702 a h).2.1
  have c118466 := (publicBatchWrapper89_copies3702 a h).2.2.1
  have c118467 := (publicBatchWrapper89_copies3702 a h).2.2.2.1
  have c118468 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.1
  have c118469 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_4
  simp only [← c118464, ← c118465, ← c118466] at e_1941_4
  have e_1941_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_5
  simp only [← c118467, ← c118468, k2, ← c118469] at e_1941_5
  have hr := e_1941_5
  simp only [e_1941_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10970)) := by
  have c118470 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.1
  have c118471 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.1
  have c118472 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.1
  have c118473 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.1
  have c118474 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118475 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_6
  simp only [← c118470, ← c118471, ← c118472] at e_1941_6
  have e_1941_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_7
  simp only [← c118473, ← c118474, k2, ← c118475] at e_1941_7
  have hr := e_1941_7
  simp only [e_1941_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10971)) := by
  have c118476 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118477 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118478 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118479 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118480 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118481 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_8
  simp only [← c118476, ← c118477, ← c118478] at e_1941_8
  have e_1941_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_9
  simp only [← c118479, ← c118480, k2, ← c118481] at e_1941_9
  have hr := e_1941_9
  simp only [e_1941_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10972)) := by
  have c118482 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118483 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118484 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118485 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118486 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118487 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_10
  simp only [← c118482, ← c118483, ← c118484] at e_1941_10
  have e_1941_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_11
  simp only [← c118485, ← c118486, k2, ← c118487] at e_1941_11
  have hr := e_1941_11
  simp only [e_1941_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10973)) := by
  have c118488 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118489 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118490 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118491 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118492 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118493 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_12
  simp only [← c118488, ← c118489, ← c118490] at e_1941_12
  have e_1941_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_13
  simp only [← c118491, ← c118492, k2, ← c118493] at e_1941_13
  have hr := e_1941_13
  simp only [e_1941_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10974)) := by
  have c118494 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118495 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118496 := (publicBatchWrapper89_copies3703 a h).1
  have c118497 := (publicBatchWrapper89_copies3703 a h).2.1
  have c118498 := (publicBatchWrapper89_copies3703 a h).2.2.1
  have c118499 := (publicBatchWrapper89_copies3703 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_14
  simp only [← c118494, ← c118495, ← c118496] at e_1941_14
  have e_1941_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_15
  simp only [← c118497, ← c118498, k2, ← c118499] at e_1941_15
  have hr := e_1941_15
  simp only [e_1941_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10975)) := by
  have c118500 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.1
  have c118501 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.1
  have c118502 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.1
  have c118503 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.1
  have c118504 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.1
  have c118505 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_16
  simp only [← c118500, ← c118501, ← c118502] at e_1941_16
  have e_1941_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_17
  simp only [← c118503, ← c118504, k2, ← c118505] at e_1941_17
  have hr := e_1941_17
  simp only [e_1941_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10976)) := by
  have c118506 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118507 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118508 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118509 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118510 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118511 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_18
  simp only [← c118506, ← c118507, ← c118508] at e_1941_18
  have e_1941_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_19
  simp only [← c118509, ← c118510, k2, ← c118511] at e_1941_19
  have hr := e_1941_19
  simp only [e_1941_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10977)) := by
  have c118512 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118513 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118514 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118515 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118516 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118517 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_0
  simp only [← c118512, ← c118513, ← c118514] at e_1942_0
  have e_1942_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_1
  simp only [← c118515, ← c118516, k2, ← c118517] at e_1942_1
  have hr := e_1942_1
  simp only [e_1942_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10978)) := by
  have c118518 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118519 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118520 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118521 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118522 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118523 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_2
  simp only [← c118518, ← c118519, ← c118520] at e_1942_2
  have e_1942_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_3
  simp only [← c118521, ← c118522, k2, ← c118523] at e_1942_3
  have hr := e_1942_3
  simp only [e_1942_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10979)) := by
  have c118524 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118525 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118526 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118527 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118528 := (publicBatchWrapper89_copies3704 a h).1
  have c118529 := (publicBatchWrapper89_copies3704 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_4
  simp only [← c118524, ← c118525, ← c118526] at e_1942_4
  have e_1942_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_5
  simp only [← c118527, ← c118528, k2, ← c118529] at e_1942_5
  have hr := e_1942_5
  simp only [e_1942_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10980)) := by
  have c118530 := (publicBatchWrapper89_copies3704 a h).2.2.1
  have c118531 := (publicBatchWrapper89_copies3704 a h).2.2.2.1
  have c118532 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.1
  have c118533 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.1
  have c118534 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.1
  have c118535 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_6
  simp only [← c118530, ← c118531, ← c118532] at e_1942_6
  have e_1942_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_7
  simp only [← c118533, ← c118534, k2, ← c118535] at e_1942_7
  have hr := e_1942_7
  simp only [e_1942_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10981)) := by
  have c118536 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.1
  have c118537 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.1
  have c118538 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118539 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118540 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118541 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_8
  simp only [← c118536, ← c118537, ← c118538] at e_1942_8
  have e_1942_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_9
  simp only [← c118539, ← c118540, k2, ← c118541] at e_1942_9
  have hr := e_1942_9
  simp only [e_1942_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10982)) := by
  have c118542 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118543 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118544 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118545 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118546 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118547 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_10
  simp only [← c118542, ← c118543, ← c118544] at e_1942_10
  have e_1942_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_11
  simp only [← c118545, ← c118546, k2, ← c118547] at e_1942_11
  have hr := e_1942_11
  simp only [e_1942_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10983)) := by
  have c118548 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118549 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118550 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118551 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118552 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118553 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_12
  simp only [← c118548, ← c118549, ← c118550] at e_1942_12
  have e_1942_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_13
  simp only [← c118551, ← c118552, k2, ← c118553] at e_1942_13
  have hr := e_1942_13
  simp only [e_1942_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10984)) := by
  have c118554 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118555 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118556 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118557 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118558 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118559 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_14
  simp only [← c118554, ← c118555, ← c118556] at e_1942_14
  have e_1942_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_15
  simp only [← c118557, ← c118558, k2, ← c118559] at e_1942_15
  have hr := e_1942_15
  simp only [e_1942_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10985)) := by
  have c118560 := (publicBatchWrapper89_copies3705 a h).1
  have c118561 := (publicBatchWrapper89_copies3705 a h).2.1
  have c118562 := (publicBatchWrapper89_copies3705 a h).2.2.1
  have c118563 := (publicBatchWrapper89_copies3705 a h).2.2.2.1
  have c118564 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.1
  have c118565 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_16
  simp only [← c118560, ← c118561, ← c118562] at e_1942_16
  have e_1942_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_17
  simp only [← c118563, ← c118564, k2, ← c118565] at e_1942_17
  have hr := e_1942_17
  simp only [e_1942_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10986)) := by
  have c118566 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.1
  have c118567 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.1
  have c118568 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.1
  have c118569 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.1
  have c118570 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118571 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_18
  simp only [← c118566, ← c118567, ← c118568] at e_1942_18
  have e_1942_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_19
  simp only [← c118569, ← c118570, k2, ← c118571] at e_1942_19
  have hr := e_1942_19
  simp only [e_1942_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10987)) := by
  have c118572 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118573 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118574 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118575 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118576 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118577 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_0
  simp only [← c118572, ← c118573, ← c118574] at e_1943_0
  have e_1943_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_1
  simp only [← c118575, ← c118576, k2, ← c118577] at e_1943_1
  have hr := e_1943_1
  simp only [e_1943_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10988)) := by
  have c118578 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118579 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118580 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118581 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118582 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118583 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_2
  simp only [← c118578, ← c118579, ← c118580] at e_1943_2
  have e_1943_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_3
  simp only [← c118581, ← c118582, k2, ← c118583] at e_1943_3
  have hr := e_1943_3
  simp only [e_1943_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10989)) := by
  have c118584 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118585 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118586 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118587 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118588 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118589 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_4
  simp only [← c118584, ← c118585, ← c118586] at e_1943_4
  have e_1943_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_5
  simp only [← c118587, ← c118588, k2, ← c118589] at e_1943_5
  have hr := e_1943_5
  simp only [e_1943_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10990)) := by
  have c118590 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118591 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118592 := (publicBatchWrapper89_copies3706 a h).1
  have c118593 := (publicBatchWrapper89_copies3706 a h).2.1
  have c118594 := (publicBatchWrapper89_copies3706 a h).2.2.1
  have c118595 := (publicBatchWrapper89_copies3706 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_6
  simp only [← c118590, ← c118591, ← c118592] at e_1943_6
  have e_1943_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_7
  simp only [← c118593, ← c118594, k2, ← c118595] at e_1943_7
  have hr := e_1943_7
  simp only [e_1943_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10991)) := by
  have c118596 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.1
  have c118597 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.1
  have c118598 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.1
  have c118599 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.1
  have c118600 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.1
  have c118601 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_8
  simp only [← c118596, ← c118597, ← c118598] at e_1943_8
  have e_1943_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_9
  simp only [← c118599, ← c118600, k2, ← c118601] at e_1943_9
  have hr := e_1943_9
  simp only [e_1943_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10992)) := by
  have c118602 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118603 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118604 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118605 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118606 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118607 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_10
  simp only [← c118602, ← c118603, ← c118604] at e_1943_10
  have e_1943_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_11
  simp only [← c118605, ← c118606, k2, ← c118607] at e_1943_11
  have hr := e_1943_11
  simp only [e_1943_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10993)) := by
  have c118608 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118609 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118610 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118611 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118612 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118613 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_12
  simp only [← c118608, ← c118609, ← c118610] at e_1943_12
  have e_1943_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_13
  simp only [← c118611, ← c118612, k2, ← c118613] at e_1943_13
  have hr := e_1943_13
  simp only [e_1943_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10994)) := by
  have c118614 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118615 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118616 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118617 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118618 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118619 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_14
  simp only [← c118614, ← c118615, ← c118616] at e_1943_14
  have e_1943_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_15
  simp only [← c118617, ← c118618, k2, ← c118619] at e_1943_15
  have hr := e_1943_15
  simp only [e_1943_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10995)) := by
  have c118620 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118621 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118622 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118623 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118624 := (publicBatchWrapper89_copies3707 a h).1
  have c118625 := (publicBatchWrapper89_copies3707 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_16
  simp only [← c118620, ← c118621, ← c118622] at e_1943_16
  have e_1943_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_17
  simp only [← c118623, ← c118624, k2, ← c118625] at e_1943_17
  have hr := e_1943_17
  simp only [e_1943_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10996)) := by
  have c118626 := (publicBatchWrapper89_copies3707 a h).2.2.1
  have c118627 := (publicBatchWrapper89_copies3707 a h).2.2.2.1
  have c118628 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.1
  have c118629 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.1
  have c118630 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.1
  have c118631 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_18
  simp only [← c118626, ← c118627, ← c118628] at e_1943_18
  have e_1943_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_19
  simp only [← c118629, ← c118630, k2, ← c118631] at e_1943_19
  have hr := e_1943_19
  simp only [e_1943_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10997)) := by
  have c118632 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.1
  have c118633 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.1
  have c118634 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118635 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118636 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118637 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_0
  simp only [← c118632, ← c118633, ← c118634] at e_1944_0
  have e_1944_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_1
  simp only [← c118635, ← c118636, k2, ← c118637] at e_1944_1
  have hr := e_1944_1
  simp only [e_1944_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10998)) := by
  have c118638 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118639 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118640 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118641 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118642 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118643 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_2
  simp only [← c118638, ← c118639, ← c118640] at e_1944_2
  have e_1944_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_3
  simp only [← c118641, ← c118642, k2, ← c118643] at e_1944_3
  have hr := e_1944_3
  simp only [e_1944_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10999)) := by
  have c118644 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118645 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118646 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118647 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118648 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118649 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_4
  simp only [← c118644, ← c118645, ← c118646] at e_1944_4
  have e_1944_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_5
  simp only [← c118647, ← c118648, k2, ← c118649] at e_1944_5
  have hr := e_1944_5
  simp only [e_1944_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 11000)) := by
  have c118650 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118651 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118652 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118653 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118654 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118655 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_6
  simp only [← c118650, ← c118651, ← c118652] at e_1944_6
  have e_1944_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_7
  simp only [← c118653, ← c118654, k2, ← c118655] at e_1944_7
  have hr := e_1944_7
  simp only [e_1944_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 11001)) := by
  have c118656 := (publicBatchWrapper89_copies3708 a h).1
  have c118657 := (publicBatchWrapper89_copies3708 a h).2.1
  have c118658 := (publicBatchWrapper89_copies3708 a h).2.2.1
  have c118659 := (publicBatchWrapper89_copies3708 a h).2.2.2.1
  have c118660 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.1
  have c118661 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_8
  simp only [← c118656, ← c118657, ← c118658] at e_1944_8
  have e_1944_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_9
  simp only [← c118659, ← c118660, k2, ← c118661] at e_1944_9
  have hr := e_1944_9
  simp only [e_1944_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11170)) := by
  have c118662 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.1
  have c118663 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.1
  have c118664 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.1
  have c118665 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.1
  have c118666 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118667 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_10
  simp only [← c118662, ← c118663, ← c118664] at e_1944_10
  have e_1944_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_11
  simp only [← c118665, ← c118666, k2, ← c118667] at e_1944_11
  have hr := e_1944_11
  simp only [e_1944_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11171)) := by
  have c118668 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118669 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118670 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118671 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118672 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118673 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_12
  simp only [← c118668, ← c118669, ← c118670] at e_1944_12
  have e_1944_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_13
  simp only [← c118671, ← c118672, k2, ← c118673] at e_1944_13
  have hr := e_1944_13
  simp only [e_1944_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11172)) := by
  have c118674 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118675 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118676 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118677 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118678 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118679 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_14
  simp only [← c118674, ← c118675, ← c118676] at e_1944_14
  have e_1944_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_15
  simp only [← c118677, ← c118678, k2, ← c118679] at e_1944_15
  have hr := e_1944_15
  simp only [e_1944_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11173)) := by
  have c118680 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118681 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118682 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118683 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118684 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118685 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_16
  simp only [← c118680, ← c118681, ← c118682] at e_1944_16
  have e_1944_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_17
  simp only [← c118683, ← c118684, k2, ← c118685] at e_1944_17
  have hr := e_1944_17
  simp only [e_1944_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11174)) := by
  have c118686 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118687 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118688 := (publicBatchWrapper89_copies3709 a h).1
  have c118689 := (publicBatchWrapper89_copies3709 a h).2.1
  have c118690 := (publicBatchWrapper89_copies3709 a h).2.2.1
  have c118691 := (publicBatchWrapper89_copies3709 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_18
  simp only [← c118686, ← c118687, ← c118688] at e_1944_18
  have e_1944_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_19
  simp only [← c118689, ← c118690, k2, ← c118691] at e_1944_19
  have hr := e_1944_19
  simp only [e_1944_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11175)) := by
  have c118692 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.1
  have c118693 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.1
  have c118694 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.1
  have c118695 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.1
  have c118696 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.1
  have c118697 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_0
  simp only [← c118692, ← c118693, ← c118694] at e_1945_0
  have e_1945_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_1
  simp only [← c118695, ← c118696, k2, ← c118697] at e_1945_1
  have hr := e_1945_1
  simp only [e_1945_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11176)) := by
  have c118698 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118699 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118700 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118701 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118702 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118703 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_2
  simp only [← c118698, ← c118699, ← c118700] at e_1945_2
  have e_1945_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_3
  simp only [← c118701, ← c118702, k2, ← c118703] at e_1945_3
  have hr := e_1945_3
  simp only [e_1945_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11177)) := by
  have c118704 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118705 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118706 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118707 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118708 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118709 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_4
  simp only [← c118704, ← c118705, ← c118706] at e_1945_4
  have e_1945_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_5
  simp only [← c118707, ← c118708, k2, ← c118709] at e_1945_5
  have hr := e_1945_5
  simp only [e_1945_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11178)) := by
  have c118710 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118711 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118712 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118713 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118714 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118715 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_6
  simp only [← c118710, ← c118711, ← c118712] at e_1945_6
  have e_1945_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_7
  simp only [← c118713, ← c118714, k2, ← c118715] at e_1945_7
  have hr := e_1945_7
  simp only [e_1945_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11179)) := by
  have c118716 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118717 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118718 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118719 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118720 := (publicBatchWrapper89_copies3710 a h).1
  have c118721 := (publicBatchWrapper89_copies3710 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_8
  simp only [← c118716, ← c118717, ← c118718] at e_1945_8
  have e_1945_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_9
  simp only [← c118719, ← c118720, k2, ← c118721] at e_1945_9
  have hr := e_1945_9
  simp only [e_1945_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11180)) := by
  have c118722 := (publicBatchWrapper89_copies3710 a h).2.2.1
  have c118723 := (publicBatchWrapper89_copies3710 a h).2.2.2.1
  have c118724 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.1
  have c118725 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.1
  have c118726 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.1
  have c118727 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_10
  simp only [← c118722, ← c118723, ← c118724] at e_1945_10
  have e_1945_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_11
  simp only [← c118725, ← c118726, k2, ← c118727] at e_1945_11
  have hr := e_1945_11
  simp only [e_1945_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11181)) := by
  have c118728 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.1
  have c118729 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.1
  have c118730 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118731 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118732 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118733 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_12
  simp only [← c118728, ← c118729, ← c118730] at e_1945_12
  have e_1945_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_13
  simp only [← c118731, ← c118732, k2, ← c118733] at e_1945_13
  have hr := e_1945_13
  simp only [e_1945_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11182)) := by
  have c118734 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118735 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118736 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118737 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118738 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118739 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_14
  simp only [← c118734, ← c118735, ← c118736] at e_1945_14
  have e_1945_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_15
  simp only [← c118737, ← c118738, k2, ← c118739] at e_1945_15
  have hr := e_1945_15
  simp only [e_1945_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11183)) := by
  have c118740 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118741 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118742 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118743 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118744 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118745 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_16
  simp only [← c118740, ← c118741, ← c118742] at e_1945_16
  have e_1945_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_17
  simp only [← c118743, ← c118744, k2, ← c118745] at e_1945_17
  have hr := e_1945_17
  simp only [e_1945_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11184)) := by
  have c118746 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118747 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118748 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118749 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118750 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118751 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_18
  simp only [← c118746, ← c118747, ← c118748] at e_1945_18
  have e_1945_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_19
  simp only [← c118749, ← c118750, k2, ← c118751] at e_1945_19
  have hr := e_1945_19
  simp only [e_1945_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11185)) := by
  have c118752 := (publicBatchWrapper89_copies3711 a h).1
  have c118753 := (publicBatchWrapper89_copies3711 a h).2.1
  have c118754 := (publicBatchWrapper89_copies3711 a h).2.2.1
  have c118755 := (publicBatchWrapper89_copies3711 a h).2.2.2.1
  have c118756 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.1
  have c118757 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_0
  simp only [← c118752, ← c118753, ← c118754] at e_1946_0
  have e_1946_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_1
  simp only [← c118755, ← c118756, k2, ← c118757] at e_1946_1
  have hr := e_1946_1
  simp only [e_1946_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11186)) := by
  have c118758 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.1
  have c118759 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.1
  have c118760 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.1
  have c118761 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.1
  have c118762 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118763 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_2
  simp only [← c118758, ← c118759, ← c118760] at e_1946_2
  have e_1946_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_3
  simp only [← c118761, ← c118762, k2, ← c118763] at e_1946_3
  have hr := e_1946_3
  simp only [e_1946_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11187)) := by
  have c118764 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118765 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118766 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118767 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118768 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118769 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_4
  simp only [← c118764, ← c118765, ← c118766] at e_1946_4
  have e_1946_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_5
  simp only [← c118767, ← c118768, k2, ← c118769] at e_1946_5
  have hr := e_1946_5
  simp only [e_1946_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11188)) := by
  have c118770 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118771 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118772 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118773 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118774 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118775 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_6
  simp only [← c118770, ← c118771, ← c118772] at e_1946_6
  have e_1946_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_7
  simp only [← c118773, ← c118774, k2, ← c118775] at e_1946_7
  have hr := e_1946_7
  simp only [e_1946_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11189)) := by
  have c118776 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118777 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118778 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118779 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118780 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118781 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_8
  simp only [← c118776, ← c118777, ← c118778] at e_1946_8
  have e_1946_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_9
  simp only [← c118779, ← c118780, k2, ← c118781] at e_1946_9
  have hr := e_1946_9
  simp only [e_1946_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11190)) := by
  have c118782 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118783 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118784 := (publicBatchWrapper89_copies3712 a h).1
  have c118785 := (publicBatchWrapper89_copies3712 a h).2.1
  have c118786 := (publicBatchWrapper89_copies3712 a h).2.2.1
  have c118787 := (publicBatchWrapper89_copies3712 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_10
  simp only [← c118782, ← c118783, ← c118784] at e_1946_10
  have e_1946_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_11
  simp only [← c118785, ← c118786, k2, ← c118787] at e_1946_11
  have hr := e_1946_11
  simp only [e_1946_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11191)) := by
  have c118788 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.1
  have c118789 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.1
  have c118790 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.1
  have c118791 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.1
  have c118792 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.1
  have c118793 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_12
  simp only [← c118788, ← c118789, ← c118790] at e_1946_12
  have e_1946_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_13
  simp only [← c118791, ← c118792, k2, ← c118793] at e_1946_13
  have hr := e_1946_13
  simp only [e_1946_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11192)) := by
  have c118794 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118795 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118796 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118797 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118798 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118799 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_14
  simp only [← c118794, ← c118795, ← c118796] at e_1946_14
  have e_1946_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_15
  simp only [← c118797, ← c118798, k2, ← c118799] at e_1946_15
  have hr := e_1946_15
  simp only [e_1946_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11193)) := by
  have c118800 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118801 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118802 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118803 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118804 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118805 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_16
  simp only [← c118800, ← c118801, ← c118802] at e_1946_16
  have e_1946_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_17
  simp only [← c118803, ← c118804, k2, ← c118805] at e_1946_17
  have hr := e_1946_17
  simp only [e_1946_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11194)) := by
  have c118806 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118807 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118808 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118809 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118810 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118811 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_18
  simp only [← c118806, ← c118807, ← c118808] at e_1946_18
  have e_1946_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_19
  simp only [← c118809, ← c118810, k2, ← c118811] at e_1946_19
  have hr := e_1946_19
  simp only [e_1946_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11195)) := by
  have c118812 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118813 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118814 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118815 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118816 := (publicBatchWrapper89_copies3713 a h).1
  have c118817 := (publicBatchWrapper89_copies3713 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_0
  simp only [← c118812, ← c118813, ← c118814] at e_1947_0
  have e_1947_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_1
  simp only [← c118815, ← c118816, k2, ← c118817] at e_1947_1
  have hr := e_1947_1
  simp only [e_1947_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11196)) := by
  have c118818 := (publicBatchWrapper89_copies3713 a h).2.2.1
  have c118819 := (publicBatchWrapper89_copies3713 a h).2.2.2.1
  have c118820 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.1
  have c118821 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.1
  have c118822 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.1
  have c118823 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_2
  simp only [← c118818, ← c118819, ← c118820] at e_1947_2
  have e_1947_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_3
  simp only [← c118821, ← c118822, k2, ← c118823] at e_1947_3
  have hr := e_1947_3
  simp only [e_1947_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11197)) := by
  have c118824 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.1
  have c118825 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.1
  have c118826 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118827 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118828 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118829 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_4
  simp only [← c118824, ← c118825, ← c118826] at e_1947_4
  have e_1947_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_5
  simp only [← c118827, ← c118828, k2, ← c118829] at e_1947_5
  have hr := e_1947_5
  simp only [e_1947_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11198)) := by
  have c118830 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118831 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118832 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118833 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118834 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118835 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_6
  simp only [← c118830, ← c118831, ← c118832] at e_1947_6
  have e_1947_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_7
  simp only [← c118833, ← c118834, k2, ← c118835] at e_1947_7
  have hr := e_1947_7
  simp only [e_1947_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11199)) := by
  have c118836 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118837 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118838 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118839 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118840 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118841 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_8
  simp only [← c118836, ← c118837, ← c118838] at e_1947_8
  have e_1947_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_9
  simp only [← c118839, ← c118840, k2, ← c118841] at e_1947_9
  have hr := e_1947_9
  simp only [e_1947_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11200)) := by
  have c118842 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118843 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118844 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118845 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118846 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118847 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_10
  simp only [← c118842, ← c118843, ← c118844] at e_1947_10
  have e_1947_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_11
  simp only [← c118845, ← c118846, k2, ← c118847] at e_1947_11
  have hr := e_1947_11
  simp only [e_1947_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11201)) := by
  have c118848 := (publicBatchWrapper89_copies3714 a h).1
  have c118849 := (publicBatchWrapper89_copies3714 a h).2.1
  have c118850 := (publicBatchWrapper89_copies3714 a h).2.2.1
  have c118851 := (publicBatchWrapper89_copies3714 a h).2.2.2.1
  have c118852 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.1
  have c118853 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_12
  simp only [← c118848, ← c118849, ← c118850] at e_1947_12
  have e_1947_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_13
  simp only [← c118851, ← c118852, k2, ← c118853] at e_1947_13
  have hr := e_1947_13
  simp only [e_1947_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11370)) := by
  have c118854 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.1
  have c118855 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.1
  have c118856 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.1
  have c118857 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.1
  have c118858 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118859 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_14
  simp only [← c118854, ← c118855, ← c118856] at e_1947_14
  have e_1947_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_15
  simp only [← c118857, ← c118858, k2, ← c118859] at e_1947_15
  have hr := e_1947_15
  simp only [e_1947_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11371)) := by
  have c118860 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118861 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118862 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118863 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118864 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118865 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_16
  simp only [← c118860, ← c118861, ← c118862] at e_1947_16
  have e_1947_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_17
  simp only [← c118863, ← c118864, k2, ← c118865] at e_1947_17
  have hr := e_1947_17
  simp only [e_1947_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11372)) := by
  have c118866 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118867 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118868 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118869 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118870 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118871 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_18
  simp only [← c118866, ← c118867, ← c118868] at e_1947_18
  have e_1947_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_19
  simp only [← c118869, ← c118870, k2, ← c118871] at e_1947_19
  have hr := e_1947_19
  simp only [e_1947_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11373)) := by
  have c118872 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118873 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118874 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118875 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118876 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118877 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_0
  simp only [← c118872, ← c118873, ← c118874] at e_1948_0
  have e_1948_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_1
  simp only [← c118875, ← c118876, k2, ← c118877] at e_1948_1
  have hr := e_1948_1
  simp only [e_1948_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11374)) := by
  have c118878 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118879 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118880 := (publicBatchWrapper89_copies3715 a h).1
  have c118881 := (publicBatchWrapper89_copies3715 a h).2.1
  have c118882 := (publicBatchWrapper89_copies3715 a h).2.2.1
  have c118883 := (publicBatchWrapper89_copies3715 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_2
  simp only [← c118878, ← c118879, ← c118880] at e_1948_2
  have e_1948_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_3
  simp only [← c118881, ← c118882, k2, ← c118883] at e_1948_3
  have hr := e_1948_3
  simp only [e_1948_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11375)) := by
  have c118884 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.1
  have c118885 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.1
  have c118886 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.1
  have c118887 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.1
  have c118888 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.1
  have c118889 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_4
  simp only [← c118884, ← c118885, ← c118886] at e_1948_4
  have e_1948_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_5
  simp only [← c118887, ← c118888, k2, ← c118889] at e_1948_5
  have hr := e_1948_5
  simp only [e_1948_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11376)) := by
  have c118890 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118891 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118892 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118893 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118894 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118895 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_6
  simp only [← c118890, ← c118891, ← c118892] at e_1948_6
  have e_1948_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_7
  simp only [← c118893, ← c118894, k2, ← c118895] at e_1948_7
  have hr := e_1948_7
  simp only [e_1948_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11377)) := by
  have c118896 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118897 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118898 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118899 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118900 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118901 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_8
  simp only [← c118896, ← c118897, ← c118898] at e_1948_8
  have e_1948_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_9
  simp only [← c118899, ← c118900, k2, ← c118901] at e_1948_9
  have hr := e_1948_9
  simp only [e_1948_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11378)) := by
  have c118902 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118903 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118904 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118905 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118906 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118907 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_10
  simp only [← c118902, ← c118903, ← c118904] at e_1948_10
  have e_1948_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_11
  simp only [← c118905, ← c118906, k2, ← c118907] at e_1948_11
  have hr := e_1948_11
  simp only [e_1948_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11379)) := by
  have c118908 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118909 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118910 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118911 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118912 := (publicBatchWrapper89_copies3716 a h).1
  have c118913 := (publicBatchWrapper89_copies3716 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_12
  simp only [← c118908, ← c118909, ← c118910] at e_1948_12
  have e_1948_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_13
  simp only [← c118911, ← c118912, k2, ← c118913] at e_1948_13
  have hr := e_1948_13
  simp only [e_1948_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11380)) := by
  have c118914 := (publicBatchWrapper89_copies3716 a h).2.2.1
  have c118915 := (publicBatchWrapper89_copies3716 a h).2.2.2.1
  have c118916 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.1
  have c118917 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.1
  have c118918 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.1
  have c118919 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_14
  simp only [← c118914, ← c118915, ← c118916] at e_1948_14
  have e_1948_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_15
  simp only [← c118917, ← c118918, k2, ← c118919] at e_1948_15
  have hr := e_1948_15
  simp only [e_1948_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11381)) := by
  have c118920 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.1
  have c118921 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.1
  have c118922 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118923 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118924 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118925 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_16
  simp only [← c118920, ← c118921, ← c118922] at e_1948_16
  have e_1948_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_17
  simp only [← c118923, ← c118924, k2, ← c118925] at e_1948_17
  have hr := e_1948_17
  simp only [e_1948_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11382)) := by
  have c118926 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118927 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118928 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118929 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118930 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118931 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_18
  simp only [← c118926, ← c118927, ← c118928] at e_1948_18
  have e_1948_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_19
  simp only [← c118929, ← c118930, k2, ← c118931] at e_1948_19
  have hr := e_1948_19
  simp only [e_1948_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11383)) := by
  have c118932 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118933 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118934 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118935 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118936 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118937 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_0
  simp only [← c118932, ← c118933, ← c118934] at e_1949_0
  have e_1949_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_1
  simp only [← c118935, ← c118936, k2, ← c118937] at e_1949_1
  have hr := e_1949_1
  simp only [e_1949_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11384)) := by
  have c118938 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118939 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118940 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118941 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118942 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118943 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_2
  simp only [← c118938, ← c118939, ← c118940] at e_1949_2
  have e_1949_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_3
  simp only [← c118941, ← c118942, k2, ← c118943] at e_1949_3
  have hr := e_1949_3
  simp only [e_1949_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11385)) := by
  have c118944 := (publicBatchWrapper89_copies3717 a h).1
  have c118945 := (publicBatchWrapper89_copies3717 a h).2.1
  have c118946 := (publicBatchWrapper89_copies3717 a h).2.2.1
  have c118947 := (publicBatchWrapper89_copies3717 a h).2.2.2.1
  have c118948 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.1
  have c118949 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_4
  simp only [← c118944, ← c118945, ← c118946] at e_1949_4
  have e_1949_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_5
  simp only [← c118947, ← c118948, k2, ← c118949] at e_1949_5
  have hr := e_1949_5
  simp only [e_1949_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11386)) := by
  have c118950 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.1
  have c118951 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.1
  have c118952 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.1
  have c118953 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.1
  have c118954 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118955 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_6
  simp only [← c118950, ← c118951, ← c118952] at e_1949_6
  have e_1949_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_7
  simp only [← c118953, ← c118954, k2, ← c118955] at e_1949_7
  have hr := e_1949_7
  simp only [e_1949_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11387)) := by
  have c118956 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118957 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118958 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118959 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118960 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118961 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_8
  simp only [← c118956, ← c118957, ← c118958] at e_1949_8
  have e_1949_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_9
  simp only [← c118959, ← c118960, k2, ← c118961] at e_1949_9
  have hr := e_1949_9
  simp only [e_1949_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11388)) := by
  have c118962 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118963 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118964 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118965 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118966 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118967 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_10
  simp only [← c118962, ← c118963, ← c118964] at e_1949_10
  have e_1949_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_11
  simp only [← c118965, ← c118966, k2, ← c118967] at e_1949_11
  have hr := e_1949_11
  simp only [e_1949_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11389)) := by
  have c118968 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118969 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118970 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118971 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118972 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118973 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_12
  simp only [← c118968, ← c118969, ← c118970] at e_1949_12
  have e_1949_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_13
  simp only [← c118971, ← c118972, k2, ← c118973] at e_1949_13
  have hr := e_1949_13
  simp only [e_1949_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11390)) := by
  have c118974 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118975 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118976 := (publicBatchWrapper89_copies3718 a h).1
  have c118977 := (publicBatchWrapper89_copies3718 a h).2.1
  have c118978 := (publicBatchWrapper89_copies3718 a h).2.2.1
  have c118979 := (publicBatchWrapper89_copies3718 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_14
  simp only [← c118974, ← c118975, ← c118976] at e_1949_14
  have e_1949_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_15
  simp only [← c118977, ← c118978, k2, ← c118979] at e_1949_15
  have hr := e_1949_15
  simp only [e_1949_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11391)) := by
  have c118980 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.1
  have c118981 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.1
  have c118982 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.1
  have c118983 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.1
  have c118984 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.1
  have c118985 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_16
  simp only [← c118980, ← c118981, ← c118982] at e_1949_16
  have e_1949_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_17
  simp only [← c118983, ← c118984, k2, ← c118985] at e_1949_17
  have hr := e_1949_17
  simp only [e_1949_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11392)) := by
  have c118986 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118987 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118988 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118989 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118990 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118991 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_18
  simp only [← c118986, ← c118987, ← c118988] at e_1949_18
  have e_1949_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_19
  simp only [← c118989, ← c118990, k2, ← c118991] at e_1949_19
  have hr := e_1949_19
  simp only [e_1949_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11393)) := by
  have c118992 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118993 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118994 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118995 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118996 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118997 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_0
  simp only [← c118992, ← c118993, ← c118994] at e_1950_0
  have e_1950_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_1
  simp only [← c118995, ← c118996, k2, ← c118997] at e_1950_1
  have hr := e_1950_1
  simp only [e_1950_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11394)) := by
  have c118998 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118999 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119000 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119001 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119002 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119003 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_2
  simp only [← c118998, ← c118999, ← c119000] at e_1950_2
  have e_1950_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_3
  simp only [← c119001, ← c119002, k2, ← c119003] at e_1950_3
  have hr := e_1950_3
  simp only [e_1950_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11395)) := by
  have c119004 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119005 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119006 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119007 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119008 := (publicBatchWrapper89_copies3719 a h).1
  have c119009 := (publicBatchWrapper89_copies3719 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_4
  simp only [← c119004, ← c119005, ← c119006] at e_1950_4
  have e_1950_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_5
  simp only [← c119007, ← c119008, k2, ← c119009] at e_1950_5
  have hr := e_1950_5
  simp only [e_1950_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11396)) := by
  have c119010 := (publicBatchWrapper89_copies3719 a h).2.2.1
  have c119011 := (publicBatchWrapper89_copies3719 a h).2.2.2.1
  have c119012 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.1
  have c119013 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.1
  have c119014 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.1
  have c119015 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_6
  simp only [← c119010, ← c119011, ← c119012] at e_1950_6
  have e_1950_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_7
  simp only [← c119013, ← c119014, k2, ← c119015] at e_1950_7
  have hr := e_1950_7
  simp only [e_1950_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11397)) := by
  have c119016 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.1
  have c119017 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.1
  have c119018 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119019 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119020 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119021 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_8
  simp only [← c119016, ← c119017, ← c119018] at e_1950_8
  have e_1950_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_9
  simp only [← c119019, ← c119020, k2, ← c119021] at e_1950_9
  have hr := e_1950_9
  simp only [e_1950_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11398)) := by
  have c119022 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119023 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119024 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119025 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119026 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119027 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_10
  simp only [← c119022, ← c119023, ← c119024] at e_1950_10
  have e_1950_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_11
  simp only [← c119025, ← c119026, k2, ← c119027] at e_1950_11
  have hr := e_1950_11
  simp only [e_1950_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11399)) := by
  have c119028 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119029 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119030 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119031 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119032 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119033 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_12
  simp only [← c119028, ← c119029, ← c119030] at e_1950_12
  have e_1950_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_13
  simp only [← c119031, ← c119032, k2, ← c119033] at e_1950_13
  have hr := e_1950_13
  simp only [e_1950_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11400)) := by
  have c119034 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119035 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119036 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119037 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119038 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119039 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_14
  simp only [← c119034, ← c119035, ← c119036] at e_1950_14
  have e_1950_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_15
  simp only [← c119037, ← c119038, k2, ← c119039] at e_1950_15
  have hr := e_1950_15
  simp only [e_1950_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11401)) := by
  have c119040 := (publicBatchWrapper89_copies3720 a h).1
  have c119041 := (publicBatchWrapper89_copies3720 a h).2.1
  have c119042 := (publicBatchWrapper89_copies3720 a h).2.2.1
  have c119043 := (publicBatchWrapper89_copies3720 a h).2.2.2.1
  have c119044 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.1
  have c119045 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_16
  simp only [← c119040, ← c119041, ← c119042] at e_1950_16
  have e_1950_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_17
  simp only [← c119043, ← c119044, k2, ← c119045] at e_1950_17
  have hr := e_1950_17
  simp only [e_1950_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11570)) := by
  have c119046 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.1
  have c119047 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.1
  have c119048 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.1
  have c119049 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.1
  have c119050 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119051 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_18
  simp only [← c119046, ← c119047, ← c119048] at e_1950_18
  have e_1950_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_19
  simp only [← c119049, ← c119050, k2, ← c119051] at e_1950_19
  have hr := e_1950_19
  simp only [e_1950_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11571)) := by
  have c119052 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119053 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119054 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119055 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119056 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119057 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_0
  simp only [← c119052, ← c119053, ← c119054] at e_1951_0
  have e_1951_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_1
  simp only [← c119055, ← c119056, k2, ← c119057] at e_1951_1
  have hr := e_1951_1
  simp only [e_1951_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11572)) := by
  have c119058 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119059 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119060 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119061 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119062 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119063 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_2
  simp only [← c119058, ← c119059, ← c119060] at e_1951_2
  have e_1951_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_3
  simp only [← c119061, ← c119062, k2, ← c119063] at e_1951_3
  have hr := e_1951_3
  simp only [e_1951_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11573)) := by
  have c119064 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119065 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119066 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119067 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119068 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119069 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_4
  simp only [← c119064, ← c119065, ← c119066] at e_1951_4
  have e_1951_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_5
  simp only [← c119067, ← c119068, k2, ← c119069] at e_1951_5
  have hr := e_1951_5
  simp only [e_1951_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11574)) := by
  have c119070 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119071 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119072 := (publicBatchWrapper89_copies3721 a h).1
  have c119073 := (publicBatchWrapper89_copies3721 a h).2.1
  have c119074 := (publicBatchWrapper89_copies3721 a h).2.2.1
  have c119075 := (publicBatchWrapper89_copies3721 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_6
  simp only [← c119070, ← c119071, ← c119072] at e_1951_6
  have e_1951_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_7
  simp only [← c119073, ← c119074, k2, ← c119075] at e_1951_7
  have hr := e_1951_7
  simp only [e_1951_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11575)) := by
  have c119076 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.1
  have c119077 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.1
  have c119078 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.1
  have c119079 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.1
  have c119080 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.1
  have c119081 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_8
  simp only [← c119076, ← c119077, ← c119078] at e_1951_8
  have e_1951_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_9
  simp only [← c119079, ← c119080, k2, ← c119081] at e_1951_9
  have hr := e_1951_9
  simp only [e_1951_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11576)) := by
  have c119082 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119083 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119084 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119085 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119086 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119087 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_10
  simp only [← c119082, ← c119083, ← c119084] at e_1951_10
  have e_1951_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_11
  simp only [← c119085, ← c119086, k2, ← c119087] at e_1951_11
  have hr := e_1951_11
  simp only [e_1951_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11577)) := by
  have c119088 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119089 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119090 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119091 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119092 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119093 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_12
  simp only [← c119088, ← c119089, ← c119090] at e_1951_12
  have e_1951_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_13
  simp only [← c119091, ← c119092, k2, ← c119093] at e_1951_13
  have hr := e_1951_13
  simp only [e_1951_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11578)) := by
  have c119094 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119095 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119096 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119097 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119098 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119099 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_14
  simp only [← c119094, ← c119095, ← c119096] at e_1951_14
  have e_1951_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_15
  simp only [← c119097, ← c119098, k2, ← c119099] at e_1951_15
  have hr := e_1951_15
  simp only [e_1951_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11579)) := by
  have c119100 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119101 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119102 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119103 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119104 := (publicBatchWrapper89_copies3722 a h).1
  have c119105 := (publicBatchWrapper89_copies3722 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_16
  simp only [← c119100, ← c119101, ← c119102] at e_1951_16
  have e_1951_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_17
  simp only [← c119103, ← c119104, k2, ← c119105] at e_1951_17
  have hr := e_1951_17
  simp only [e_1951_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11580)) := by
  have c119106 := (publicBatchWrapper89_copies3722 a h).2.2.1
  have c119107 := (publicBatchWrapper89_copies3722 a h).2.2.2.1
  have c119108 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.1
  have c119109 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.1
  have c119110 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.1
  have c119111 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_18
  simp only [← c119106, ← c119107, ← c119108] at e_1951_18
  have e_1951_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_19
  simp only [← c119109, ← c119110, k2, ← c119111] at e_1951_19
  have hr := e_1951_19
  simp only [e_1951_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11581)) := by
  have c119112 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.1
  have c119113 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.1
  have c119114 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119115 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119116 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119117 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_0
  simp only [← c119112, ← c119113, ← c119114] at e_1952_0
  have e_1952_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_1
  simp only [← c119115, ← c119116, k2, ← c119117] at e_1952_1
  have hr := e_1952_1
  simp only [e_1952_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11582)) := by
  have c119118 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119119 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119120 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119121 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119122 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119123 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_2
  simp only [← c119118, ← c119119, ← c119120] at e_1952_2
  have e_1952_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_3
  simp only [← c119121, ← c119122, k2, ← c119123] at e_1952_3
  have hr := e_1952_3
  simp only [e_1952_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11583)) := by
  have c119124 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119125 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119126 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119127 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119128 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119129 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_4
  simp only [← c119124, ← c119125, ← c119126] at e_1952_4
  have e_1952_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_5
  simp only [← c119127, ← c119128, k2, ← c119129] at e_1952_5
  have hr := e_1952_5
  simp only [e_1952_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11584)) := by
  have c119130 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119131 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119132 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119133 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119134 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119135 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_6
  simp only [← c119130, ← c119131, ← c119132] at e_1952_6
  have e_1952_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_7
  simp only [← c119133, ← c119134, k2, ← c119135] at e_1952_7
  have hr := e_1952_7
  simp only [e_1952_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11585)) := by
  have c119136 := (publicBatchWrapper89_copies3723 a h).1
  have c119137 := (publicBatchWrapper89_copies3723 a h).2.1
  have c119138 := (publicBatchWrapper89_copies3723 a h).2.2.1
  have c119139 := (publicBatchWrapper89_copies3723 a h).2.2.2.1
  have c119140 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.1
  have c119141 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_8
  simp only [← c119136, ← c119137, ← c119138] at e_1952_8
  have e_1952_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_9
  simp only [← c119139, ← c119140, k2, ← c119141] at e_1952_9
  have hr := e_1952_9
  simp only [e_1952_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11586)) := by
  have c119142 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.1
  have c119143 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.1
  have c119144 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.1
  have c119145 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.1
  have c119146 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119147 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_10
  simp only [← c119142, ← c119143, ← c119144] at e_1952_10
  have e_1952_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_11
  simp only [← c119145, ← c119146, k2, ← c119147] at e_1952_11
  have hr := e_1952_11
  simp only [e_1952_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11587)) := by
  have c119148 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119149 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119150 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119151 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119152 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119153 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_12
  simp only [← c119148, ← c119149, ← c119150] at e_1952_12
  have e_1952_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_13
  simp only [← c119151, ← c119152, k2, ← c119153] at e_1952_13
  have hr := e_1952_13
  simp only [e_1952_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11588)) := by
  have c119154 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119155 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119156 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119157 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119158 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119159 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_14
  simp only [← c119154, ← c119155, ← c119156] at e_1952_14
  have e_1952_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_15
  simp only [← c119157, ← c119158, k2, ← c119159] at e_1952_15
  have hr := e_1952_15
  simp only [e_1952_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11589)) := by
  have c119160 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119161 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119162 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119163 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119164 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119165 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_16
  simp only [← c119160, ← c119161, ← c119162] at e_1952_16
  have e_1952_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_17
  simp only [← c119163, ← c119164, k2, ← c119165] at e_1952_17
  have hr := e_1952_17
  simp only [e_1952_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11590)) := by
  have c119166 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119167 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119168 := (publicBatchWrapper89_copies3724 a h).1
  have c119169 := (publicBatchWrapper89_copies3724 a h).2.1
  have c119170 := (publicBatchWrapper89_copies3724 a h).2.2.1
  have c119171 := (publicBatchWrapper89_copies3724 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_18
  simp only [← c119166, ← c119167, ← c119168] at e_1952_18
  have e_1952_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_19
  simp only [← c119169, ← c119170, k2, ← c119171] at e_1952_19
  have hr := e_1952_19
  simp only [e_1952_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11591)) := by
  have c119172 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.1
  have c119173 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.1
  have c119174 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.1
  have c119175 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.1
  have c119176 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.1
  have c119177 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_0
  simp only [← c119172, ← c119173, ← c119174] at e_1953_0
  have e_1953_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_1
  simp only [← c119175, ← c119176, k2, ← c119177] at e_1953_1
  have hr := e_1953_1
  simp only [e_1953_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11592)) := by
  have c119178 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119179 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119180 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119181 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119182 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119183 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_2
  simp only [← c119178, ← c119179, ← c119180] at e_1953_2
  have e_1953_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_3
  simp only [← c119181, ← c119182, k2, ← c119183] at e_1953_3
  have hr := e_1953_3
  simp only [e_1953_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11593)) := by
  have c119184 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119185 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119186 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119187 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119188 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119189 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_4
  simp only [← c119184, ← c119185, ← c119186] at e_1953_4
  have e_1953_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_5
  simp only [← c119187, ← c119188, k2, ← c119189] at e_1953_5
  have hr := e_1953_5
  simp only [e_1953_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11594)) := by
  have c119190 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119191 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119192 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119193 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119194 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119195 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_6
  simp only [← c119190, ← c119191, ← c119192] at e_1953_6
  have e_1953_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_7
  simp only [← c119193, ← c119194, k2, ← c119195] at e_1953_7
  have hr := e_1953_7
  simp only [e_1953_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11595)) := by
  have c119196 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119197 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119198 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119199 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119200 := (publicBatchWrapper89_copies3725 a h).1
  have c119201 := (publicBatchWrapper89_copies3725 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_8
  simp only [← c119196, ← c119197, ← c119198] at e_1953_8
  have e_1953_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_9
  simp only [← c119199, ← c119200, k2, ← c119201] at e_1953_9
  have hr := e_1953_9
  simp only [e_1953_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11596)) := by
  have c119202 := (publicBatchWrapper89_copies3725 a h).2.2.1
  have c119203 := (publicBatchWrapper89_copies3725 a h).2.2.2.1
  have c119204 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.1
  have c119205 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.1
  have c119206 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.1
  have c119207 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_10
  simp only [← c119202, ← c119203, ← c119204] at e_1953_10
  have e_1953_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_11
  simp only [← c119205, ← c119206, k2, ← c119207] at e_1953_11
  have hr := e_1953_11
  simp only [e_1953_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11597)) := by
  have c119208 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.1
  have c119209 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.1
  have c119210 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119211 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119212 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119213 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_12
  simp only [← c119208, ← c119209, ← c119210] at e_1953_12
  have e_1953_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_13
  simp only [← c119211, ← c119212, k2, ← c119213] at e_1953_13
  have hr := e_1953_13
  simp only [e_1953_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11598)) := by
  have c119214 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119215 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119216 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119217 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119218 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119219 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_14
  simp only [← c119214, ← c119215, ← c119216] at e_1953_14
  have e_1953_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_15
  simp only [← c119217, ← c119218, k2, ← c119219] at e_1953_15
  have hr := e_1953_15
  simp only [e_1953_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11599)) := by
  have c119220 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119221 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119222 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119223 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119224 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119225 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_16
  simp only [← c119220, ← c119221, ← c119222] at e_1953_16
  have e_1953_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_17
  simp only [← c119223, ← c119224, k2, ← c119225] at e_1953_17
  have hr := e_1953_17
  simp only [e_1953_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11600)) := by
  have c119226 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119227 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119228 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119229 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119230 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119231 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_18
  simp only [← c119226, ← c119227, ← c119228] at e_1953_18
  have e_1953_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_19
  simp only [← c119229, ← c119230, k2, ← c119231] at e_1953_19
  have hr := e_1953_19
  simp only [e_1953_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11601)) := by
  have c119232 := (publicBatchWrapper89_copies3726 a h).1
  have c119233 := (publicBatchWrapper89_copies3726 a h).2.1
  have c119234 := (publicBatchWrapper89_copies3726 a h).2.2.1
  have c119235 := (publicBatchWrapper89_copies3726 a h).2.2.2.1
  have c119236 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.1
  have c119237 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_0
  simp only [← c119232, ← c119233, ← c119234] at e_1954_0
  have e_1954_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_1
  simp only [← c119235, ← c119236, k2, ← c119237] at e_1954_1
  have hr := e_1954_1
  simp only [e_1954_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11770)) := by
  have c119238 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.1
  have c119239 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.1
  have c119240 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.1
  have c119241 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.1
  have c119242 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119243 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_2
  simp only [← c119238, ← c119239, ← c119240] at e_1954_2
  have e_1954_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_3
  simp only [← c119241, ← c119242, k2, ← c119243] at e_1954_3
  have hr := e_1954_3
  simp only [e_1954_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11771)) := by
  have c119244 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119245 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119246 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119247 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119248 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119249 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_4
  simp only [← c119244, ← c119245, ← c119246] at e_1954_4
  have e_1954_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_5
  simp only [← c119247, ← c119248, k2, ← c119249] at e_1954_5
  have hr := e_1954_5
  simp only [e_1954_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11772)) := by
  have c119250 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119251 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119252 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119253 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119254 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119255 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_6
  simp only [← c119250, ← c119251, ← c119252] at e_1954_6
  have e_1954_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_7
  simp only [← c119253, ← c119254, k2, ← c119255] at e_1954_7
  have hr := e_1954_7
  simp only [e_1954_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11773)) := by
  have c119256 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119257 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119258 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119259 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119260 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119261 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_8
  simp only [← c119256, ← c119257, ← c119258] at e_1954_8
  have e_1954_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_9
  simp only [← c119259, ← c119260, k2, ← c119261] at e_1954_9
  have hr := e_1954_9
  simp only [e_1954_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11774)) := by
  have c119262 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119263 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119264 := (publicBatchWrapper89_copies3727 a h).1
  have c119265 := (publicBatchWrapper89_copies3727 a h).2.1
  have c119266 := (publicBatchWrapper89_copies3727 a h).2.2.1
  have c119267 := (publicBatchWrapper89_copies3727 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_10
  simp only [← c119262, ← c119263, ← c119264] at e_1954_10
  have e_1954_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_11
  simp only [← c119265, ← c119266, k2, ← c119267] at e_1954_11
  have hr := e_1954_11
  simp only [e_1954_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11775)) := by
  have c119268 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.1
  have c119269 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.1
  have c119270 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.1
  have c119271 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.1
  have c119272 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.1
  have c119273 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_12
  simp only [← c119268, ← c119269, ← c119270] at e_1954_12
  have e_1954_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_13
  simp only [← c119271, ← c119272, k2, ← c119273] at e_1954_13
  have hr := e_1954_13
  simp only [e_1954_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11776)) := by
  have c119274 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119275 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119276 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119277 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119278 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119279 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_14
  simp only [← c119274, ← c119275, ← c119276] at e_1954_14
  have e_1954_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_15
  simp only [← c119277, ← c119278, k2, ← c119279] at e_1954_15
  have hr := e_1954_15
  simp only [e_1954_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11777)) := by
  have c119280 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119281 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119282 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119283 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119284 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119285 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_16
  simp only [← c119280, ← c119281, ← c119282] at e_1954_16
  have e_1954_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_17
  simp only [← c119283, ← c119284, k2, ← c119285] at e_1954_17
  have hr := e_1954_17
  simp only [e_1954_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11778)) := by
  have c119286 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119287 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119288 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119289 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119290 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119291 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_18
  simp only [← c119286, ← c119287, ← c119288] at e_1954_18
  have e_1954_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_19
  simp only [← c119289, ← c119290, k2, ← c119291] at e_1954_19
  have hr := e_1954_19
  simp only [e_1954_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11779)) := by
  have c119292 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119293 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119294 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119295 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119296 := (publicBatchWrapper89_copies3728 a h).1
  have c119297 := (publicBatchWrapper89_copies3728 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_0
  simp only [← c119292, ← c119293, ← c119294] at e_1955_0
  have e_1955_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_1
  simp only [← c119295, ← c119296, k2, ← c119297] at e_1955_1
  have hr := e_1955_1
  simp only [e_1955_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11780)) := by
  have c119298 := (publicBatchWrapper89_copies3728 a h).2.2.1
  have c119299 := (publicBatchWrapper89_copies3728 a h).2.2.2.1
  have c119300 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.1
  have c119301 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.1
  have c119302 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.1
  have c119303 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_2
  simp only [← c119298, ← c119299, ← c119300] at e_1955_2
  have e_1955_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_3
  simp only [← c119301, ← c119302, k2, ← c119303] at e_1955_3
  have hr := e_1955_3
  simp only [e_1955_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11781)) := by
  have c119304 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.1
  have c119305 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.1
  have c119306 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119307 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119308 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119309 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_4
  simp only [← c119304, ← c119305, ← c119306] at e_1955_4
  have e_1955_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_5
  simp only [← c119307, ← c119308, k2, ← c119309] at e_1955_5
  have hr := e_1955_5
  simp only [e_1955_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11782)) := by
  have c119310 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119311 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119312 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119313 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119314 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119315 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_6
  simp only [← c119310, ← c119311, ← c119312] at e_1955_6
  have e_1955_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_7
  simp only [← c119313, ← c119314, k2, ← c119315] at e_1955_7
  have hr := e_1955_7
  simp only [e_1955_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11783)) := by
  have c119316 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119317 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119318 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119319 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119320 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119321 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_8
  simp only [← c119316, ← c119317, ← c119318] at e_1955_8
  have e_1955_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_9
  simp only [← c119319, ← c119320, k2, ← c119321] at e_1955_9
  have hr := e_1955_9
  simp only [e_1955_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11784)) := by
  have c119322 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119323 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119324 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119325 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119326 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119327 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_10
  simp only [← c119322, ← c119323, ← c119324] at e_1955_10
  have e_1955_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_11
  simp only [← c119325, ← c119326, k2, ← c119327] at e_1955_11
  have hr := e_1955_11
  simp only [e_1955_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11785)) := by
  have c119328 := (publicBatchWrapper89_copies3729 a h).1
  have c119329 := (publicBatchWrapper89_copies3729 a h).2.1
  have c119330 := (publicBatchWrapper89_copies3729 a h).2.2.1
  have c119331 := (publicBatchWrapper89_copies3729 a h).2.2.2.1
  have c119332 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.1
  have c119333 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_12
  simp only [← c119328, ← c119329, ← c119330] at e_1955_12
  have e_1955_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_13
  simp only [← c119331, ← c119332, k2, ← c119333] at e_1955_13
  have hr := e_1955_13
  simp only [e_1955_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11786)) := by
  have c119334 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.1
  have c119335 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.1
  have c119336 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.1
  have c119337 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.1
  have c119338 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119339 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_14
  simp only [← c119334, ← c119335, ← c119336] at e_1955_14
  have e_1955_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_15
  simp only [← c119337, ← c119338, k2, ← c119339] at e_1955_15
  have hr := e_1955_15
  simp only [e_1955_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11787)) := by
  have c119340 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119341 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119342 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119343 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119344 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119345 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_16
  simp only [← c119340, ← c119341, ← c119342] at e_1955_16
  have e_1955_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_17
  simp only [← c119343, ← c119344, k2, ← c119345] at e_1955_17
  have hr := e_1955_17
  simp only [e_1955_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11788)) := by
  have c119346 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119347 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119348 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119349 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119350 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119351 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_18
  simp only [← c119346, ← c119347, ← c119348] at e_1955_18
  have e_1955_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_19
  simp only [← c119349, ← c119350, k2, ← c119351] at e_1955_19
  have hr := e_1955_19
  simp only [e_1955_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11789)) := by
  have c119352 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119353 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119354 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119355 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119356 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119357 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_0
  simp only [← c119352, ← c119353, ← c119354] at e_1956_0
  have e_1956_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_1
  simp only [← c119355, ← c119356, k2, ← c119357] at e_1956_1
  have hr := e_1956_1
  simp only [e_1956_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11790)) := by
  have c119358 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119359 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119360 := (publicBatchWrapper89_copies3730 a h).1
  have c119361 := (publicBatchWrapper89_copies3730 a h).2.1
  have c119362 := (publicBatchWrapper89_copies3730 a h).2.2.1
  have c119363 := (publicBatchWrapper89_copies3730 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_2
  simp only [← c119358, ← c119359, ← c119360] at e_1956_2
  have e_1956_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_3
  simp only [← c119361, ← c119362, k2, ← c119363] at e_1956_3
  have hr := e_1956_3
  simp only [e_1956_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11791)) := by
  have c119364 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.1
  have c119365 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.1
  have c119366 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.1
  have c119367 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.1
  have c119368 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.1
  have c119369 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_4
  simp only [← c119364, ← c119365, ← c119366] at e_1956_4
  have e_1956_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_5
  simp only [← c119367, ← c119368, k2, ← c119369] at e_1956_5
  have hr := e_1956_5
  simp only [e_1956_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11792)) := by
  have c119370 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119371 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119372 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119373 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119374 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119375 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_6
  simp only [← c119370, ← c119371, ← c119372] at e_1956_6
  have e_1956_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_7
  simp only [← c119373, ← c119374, k2, ← c119375] at e_1956_7
  have hr := e_1956_7
  simp only [e_1956_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11793)) := by
  have c119376 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119377 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119378 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119379 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119380 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119381 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_8
  simp only [← c119376, ← c119377, ← c119378] at e_1956_8
  have e_1956_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_9
  simp only [← c119379, ← c119380, k2, ← c119381] at e_1956_9
  have hr := e_1956_9
  simp only [e_1956_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11794)) := by
  have c119382 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119383 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119384 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119385 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119386 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119387 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_10
  simp only [← c119382, ← c119383, ← c119384] at e_1956_10
  have e_1956_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_11
  simp only [← c119385, ← c119386, k2, ← c119387] at e_1956_11
  have hr := e_1956_11
  simp only [e_1956_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11795)) := by
  have c119388 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119389 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119390 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119391 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119392 := (publicBatchWrapper89_copies3731 a h).1
  have c119393 := (publicBatchWrapper89_copies3731 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_12
  simp only [← c119388, ← c119389, ← c119390] at e_1956_12
  have e_1956_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_13
  simp only [← c119391, ← c119392, k2, ← c119393] at e_1956_13
  have hr := e_1956_13
  simp only [e_1956_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11796)) := by
  have c119394 := (publicBatchWrapper89_copies3731 a h).2.2.1
  have c119395 := (publicBatchWrapper89_copies3731 a h).2.2.2.1
  have c119396 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.1
  have c119397 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.1
  have c119398 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.1
  have c119399 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_14
  simp only [← c119394, ← c119395, ← c119396] at e_1956_14
  have e_1956_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_15
  simp only [← c119397, ← c119398, k2, ← c119399] at e_1956_15
  have hr := e_1956_15
  simp only [e_1956_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11797)) := by
  have c119400 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.1
  have c119401 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.1
  have c119402 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119403 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119404 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119405 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_16
  simp only [← c119400, ← c119401, ← c119402] at e_1956_16
  have e_1956_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_17
  simp only [← c119403, ← c119404, k2, ← c119405] at e_1956_17
  have hr := e_1956_17
  simp only [e_1956_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11798)) := by
  have c119406 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119407 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119408 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119409 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119410 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119411 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_18
  simp only [← c119406, ← c119407, ← c119408] at e_1956_18
  have e_1956_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_19
  simp only [← c119409, ← c119410, k2, ← c119411] at e_1956_19
  have hr := e_1956_19
  simp only [e_1956_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11799)) := by
  have c119412 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119413 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119414 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119415 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119416 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119417 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_0
  simp only [← c119412, ← c119413, ← c119414] at e_1957_0
  have e_1957_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_1
  simp only [← c119415, ← c119416, k2, ← c119417] at e_1957_1
  have hr := e_1957_1
  simp only [e_1957_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11800)) := by
  have c119418 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119419 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119420 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119421 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119422 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119423 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_2
  simp only [← c119418, ← c119419, ← c119420] at e_1957_2
  have e_1957_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_3
  simp only [← c119421, ← c119422, k2, ← c119423] at e_1957_3
  have hr := e_1957_3
  simp only [e_1957_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11801)) := by
  have c119424 := (publicBatchWrapper89_copies3732 a h).1
  have c119425 := (publicBatchWrapper89_copies3732 a h).2.1
  have c119426 := (publicBatchWrapper89_copies3732 a h).2.2.1
  have c119427 := (publicBatchWrapper89_copies3732 a h).2.2.2.1
  have c119428 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.1
  have c119429 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_4
  simp only [← c119424, ← c119425, ← c119426] at e_1957_4
  have e_1957_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_5
  simp only [← c119427, ← c119428, k2, ← c119429] at e_1957_5
  have hr := e_1957_5
  simp only [e_1957_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11970)) := by
  have c119430 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.1
  have c119431 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.1
  have c119432 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.1
  have c119433 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.1
  have c119434 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119435 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_6
  simp only [← c119430, ← c119431, ← c119432] at e_1957_6
  have e_1957_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_7
  simp only [← c119433, ← c119434, k2, ← c119435] at e_1957_7
  have hr := e_1957_7
  simp only [e_1957_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11971)) := by
  have c119436 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119437 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119438 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119439 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119440 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119441 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_8
  simp only [← c119436, ← c119437, ← c119438] at e_1957_8
  have e_1957_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_9
  simp only [← c119439, ← c119440, k2, ← c119441] at e_1957_9
  have hr := e_1957_9
  simp only [e_1957_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11972)) := by
  have c119442 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119443 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119444 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119445 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119446 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119447 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_10
  simp only [← c119442, ← c119443, ← c119444] at e_1957_10
  have e_1957_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_11
  simp only [← c119445, ← c119446, k2, ← c119447] at e_1957_11
  have hr := e_1957_11
  simp only [e_1957_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11973)) := by
  have c119448 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119449 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119450 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119451 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119452 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119453 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_12
  simp only [← c119448, ← c119449, ← c119450] at e_1957_12
  have e_1957_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_13
  simp only [← c119451, ← c119452, k2, ← c119453] at e_1957_13
  have hr := e_1957_13
  simp only [e_1957_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11974)) := by
  have c119454 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119455 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119456 := (publicBatchWrapper89_copies3733 a h).1
  have c119457 := (publicBatchWrapper89_copies3733 a h).2.1
  have c119458 := (publicBatchWrapper89_copies3733 a h).2.2.1
  have c119459 := (publicBatchWrapper89_copies3733 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_14
  simp only [← c119454, ← c119455, ← c119456] at e_1957_14
  have e_1957_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_15
  simp only [← c119457, ← c119458, k2, ← c119459] at e_1957_15
  have hr := e_1957_15
  simp only [e_1957_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11975)) := by
  have c119460 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.1
  have c119461 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.1
  have c119462 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.1
  have c119463 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.1
  have c119464 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.1
  have c119465 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_16
  simp only [← c119460, ← c119461, ← c119462] at e_1957_16
  have e_1957_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_17
  simp only [← c119463, ← c119464, k2, ← c119465] at e_1957_17
  have hr := e_1957_17
  simp only [e_1957_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11976)) := by
  have c119466 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119467 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119468 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119469 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119470 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119471 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_18
  simp only [← c119466, ← c119467, ← c119468] at e_1957_18
  have e_1957_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_19
  simp only [← c119469, ← c119470, k2, ← c119471] at e_1957_19
  have hr := e_1957_19
  simp only [e_1957_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11977)) := by
  have c119472 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119473 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119474 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119475 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119476 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119477 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_0
  simp only [← c119472, ← c119473, ← c119474] at e_1958_0
  have e_1958_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_1
  simp only [← c119475, ← c119476, k2, ← c119477] at e_1958_1
  have hr := e_1958_1
  simp only [e_1958_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11978)) := by
  have c119478 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119479 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119480 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119481 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119482 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119483 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_2
  simp only [← c119478, ← c119479, ← c119480] at e_1958_2
  have e_1958_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_3
  simp only [← c119481, ← c119482, k2, ← c119483] at e_1958_3
  have hr := e_1958_3
  simp only [e_1958_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11979)) := by
  have c119484 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119485 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119486 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119487 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119488 := (publicBatchWrapper89_copies3734 a h).1
  have c119489 := (publicBatchWrapper89_copies3734 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_4
  simp only [← c119484, ← c119485, ← c119486] at e_1958_4
  have e_1958_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_5
  simp only [← c119487, ← c119488, k2, ← c119489] at e_1958_5
  have hr := e_1958_5
  simp only [e_1958_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11980)) := by
  have c119490 := (publicBatchWrapper89_copies3734 a h).2.2.1
  have c119491 := (publicBatchWrapper89_copies3734 a h).2.2.2.1
  have c119492 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.1
  have c119493 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.1
  have c119494 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.1
  have c119495 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_6
  simp only [← c119490, ← c119491, ← c119492] at e_1958_6
  have e_1958_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_7
  simp only [← c119493, ← c119494, k2, ← c119495] at e_1958_7
  have hr := e_1958_7
  simp only [e_1958_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11981)) := by
  have c119496 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.1
  have c119497 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.1
  have c119498 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119499 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119500 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119501 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_8
  simp only [← c119496, ← c119497, ← c119498] at e_1958_8
  have e_1958_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_9
  simp only [← c119499, ← c119500, k2, ← c119501] at e_1958_9
  have hr := e_1958_9
  simp only [e_1958_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11982)) := by
  have c119502 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119503 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119504 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119505 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119506 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119507 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_10
  simp only [← c119502, ← c119503, ← c119504] at e_1958_10
  have e_1958_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_11
  simp only [← c119505, ← c119506, k2, ← c119507] at e_1958_11
  have hr := e_1958_11
  simp only [e_1958_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11983)) := by
  have c119508 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119509 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119510 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119511 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119512 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119513 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_12
  simp only [← c119508, ← c119509, ← c119510] at e_1958_12
  have e_1958_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_13
  simp only [← c119511, ← c119512, k2, ← c119513] at e_1958_13
  have hr := e_1958_13
  simp only [e_1958_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11984)) := by
  have c119514 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119515 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119516 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119517 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119518 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119519 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_14
  simp only [← c119514, ← c119515, ← c119516] at e_1958_14
  have e_1958_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_15
  simp only [← c119517, ← c119518, k2, ← c119519] at e_1958_15
  have hr := e_1958_15
  simp only [e_1958_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11985)) := by
  have c119520 := (publicBatchWrapper89_copies3735 a h).1
  have c119521 := (publicBatchWrapper89_copies3735 a h).2.1
  have c119522 := (publicBatchWrapper89_copies3735 a h).2.2.1
  have c119523 := (publicBatchWrapper89_copies3735 a h).2.2.2.1
  have c119524 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.1
  have c119525 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_16
  simp only [← c119520, ← c119521, ← c119522] at e_1958_16
  have e_1958_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_17
  simp only [← c119523, ← c119524, k2, ← c119525] at e_1958_17
  have hr := e_1958_17
  simp only [e_1958_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11986)) := by
  have c119526 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.1
  have c119527 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.1
  have c119528 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.1
  have c119529 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.1
  have c119530 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119531 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_18
  simp only [← c119526, ← c119527, ← c119528] at e_1958_18
  have e_1958_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_19
  simp only [← c119529, ← c119530, k2, ← c119531] at e_1958_19
  have hr := e_1958_19
  simp only [e_1958_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11987)) := by
  have c119532 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119533 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119534 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119535 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119536 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119537 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_0
  simp only [← c119532, ← c119533, ← c119534] at e_1959_0
  have e_1959_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_1
  simp only [← c119535, ← c119536, k2, ← c119537] at e_1959_1
  have hr := e_1959_1
  simp only [e_1959_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11988)) := by
  have c119538 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119539 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119540 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119541 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119542 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119543 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_2
  simp only [← c119538, ← c119539, ← c119540] at e_1959_2
  have e_1959_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_3
  simp only [← c119541, ← c119542, k2, ← c119543] at e_1959_3
  have hr := e_1959_3
  simp only [e_1959_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11989)) := by
  have c119544 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119545 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119546 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119547 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119548 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119549 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_4
  simp only [← c119544, ← c119545, ← c119546] at e_1959_4
  have e_1959_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_5
  simp only [← c119547, ← c119548, k2, ← c119549] at e_1959_5
  have hr := e_1959_5
  simp only [e_1959_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11990)) := by
  have c119550 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119551 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119552 := (publicBatchWrapper89_copies3736 a h).1
  have c119553 := (publicBatchWrapper89_copies3736 a h).2.1
  have c119554 := (publicBatchWrapper89_copies3736 a h).2.2.1
  have c119555 := (publicBatchWrapper89_copies3736 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_6
  simp only [← c119550, ← c119551, ← c119552] at e_1959_6
  have e_1959_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_7
  simp only [← c119553, ← c119554, k2, ← c119555] at e_1959_7
  have hr := e_1959_7
  simp only [e_1959_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11991)) := by
  have c119556 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.1
  have c119557 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.1
  have c119558 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.1
  have c119559 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.1
  have c119560 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.1
  have c119561 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_8
  simp only [← c119556, ← c119557, ← c119558] at e_1959_8
  have e_1959_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_9
  simp only [← c119559, ← c119560, k2, ← c119561] at e_1959_9
  have hr := e_1959_9
  simp only [e_1959_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11992)) := by
  have c119562 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119563 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119564 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119565 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119566 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119567 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_10
  simp only [← c119562, ← c119563, ← c119564] at e_1959_10
  have e_1959_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_11
  simp only [← c119565, ← c119566, k2, ← c119567] at e_1959_11
  have hr := e_1959_11
  simp only [e_1959_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11993)) := by
  have c119568 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119569 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119570 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119571 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119572 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119573 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_12
  simp only [← c119568, ← c119569, ← c119570] at e_1959_12
  have e_1959_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_13
  simp only [← c119571, ← c119572, k2, ← c119573] at e_1959_13
  have hr := e_1959_13
  simp only [e_1959_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11994)) := by
  have c119574 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119575 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119576 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119577 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119578 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119579 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_14
  simp only [← c119574, ← c119575, ← c119576] at e_1959_14
  have e_1959_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_15
  simp only [← c119577, ← c119578, k2, ← c119579] at e_1959_15
  have hr := e_1959_15
  simp only [e_1959_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11995)) := by
  have c119580 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119581 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119582 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119583 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119584 := (publicBatchWrapper89_copies3737 a h).1
  have c119585 := (publicBatchWrapper89_copies3737 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_16
  simp only [← c119580, ← c119581, ← c119582] at e_1959_16
  have e_1959_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_17
  simp only [← c119583, ← c119584, k2, ← c119585] at e_1959_17
  have hr := e_1959_17
  simp only [e_1959_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11996)) := by
  have c119586 := (publicBatchWrapper89_copies3737 a h).2.2.1
  have c119587 := (publicBatchWrapper89_copies3737 a h).2.2.2.1
  have c119588 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.1
  have c119589 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.1
  have c119590 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.1
  have c119591 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_18
  simp only [← c119586, ← c119587, ← c119588] at e_1959_18
  have e_1959_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_19
  simp only [← c119589, ← c119590, k2, ← c119591] at e_1959_19
  have hr := e_1959_19
  simp only [e_1959_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11997)) := by
  have c119592 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.1
  have c119593 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.1
  have c119594 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119595 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119596 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119597 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_0
  simp only [← c119592, ← c119593, ← c119594] at e_1960_0
  have e_1960_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_1
  simp only [← c119595, ← c119596, k2, ← c119597] at e_1960_1
  have hr := e_1960_1
  simp only [e_1960_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11998)) := by
  have c119598 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119599 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119600 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119601 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119602 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119603 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_2
  simp only [← c119598, ← c119599, ← c119600] at e_1960_2
  have e_1960_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_3
  simp only [← c119601, ← c119602, k2, ← c119603] at e_1960_3
  have hr := e_1960_3
  simp only [e_1960_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11999)) := by
  have c119604 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119605 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119606 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119607 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119608 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119609 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_4
  simp only [← c119604, ← c119605, ← c119606] at e_1960_4
  have e_1960_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_5
  simp only [← c119607, ← c119608, k2, ← c119609] at e_1960_5
  have hr := e_1960_5
  simp only [e_1960_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 12000)) := by
  have c119610 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119611 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119612 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119613 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119614 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119615 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_6
  simp only [← c119610, ← c119611, ← c119612] at e_1960_6
  have e_1960_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_7
  simp only [← c119613, ← c119614, k2, ← c119615] at e_1960_7
  have hr := e_1960_7
  simp only [e_1960_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 12001)) := by
  have c119616 := (publicBatchWrapper89_copies3738 a h).1
  have c119617 := (publicBatchWrapper89_copies3738 a h).2.1
  have c119618 := (publicBatchWrapper89_copies3738 a h).2.2.1
  have c119619 := (publicBatchWrapper89_copies3738 a h).2.2.2.1
  have c119620 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.1
  have c119621 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_8
  simp only [← c119616, ← c119617, ← c119618] at e_1960_8
  have e_1960_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_9
  simp only [← c119619, ← c119620, k2, ← c119621] at e_1960_9
  have hr := e_1960_9
  simp only [e_1960_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12170)) := by
  have c119622 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.1
  have c119623 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.1
  have c119624 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.1
  have c119625 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.1
  have c119626 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119627 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_10
  simp only [← c119622, ← c119623, ← c119624] at e_1960_10
  have e_1960_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_11
  simp only [← c119625, ← c119626, k2, ← c119627] at e_1960_11
  have hr := e_1960_11
  simp only [e_1960_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12171)) := by
  have c119628 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119629 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119630 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119631 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119632 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119633 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_12
  simp only [← c119628, ← c119629, ← c119630] at e_1960_12
  have e_1960_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_13
  simp only [← c119631, ← c119632, k2, ← c119633] at e_1960_13
  have hr := e_1960_13
  simp only [e_1960_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12172)) := by
  have c119634 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119635 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119636 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119637 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119638 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119639 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_14
  simp only [← c119634, ← c119635, ← c119636] at e_1960_14
  have e_1960_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_15
  simp only [← c119637, ← c119638, k2, ← c119639] at e_1960_15
  have hr := e_1960_15
  simp only [e_1960_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12173)) := by
  have c119640 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119641 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119642 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119643 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119644 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119645 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_16
  simp only [← c119640, ← c119641, ← c119642] at e_1960_16
  have e_1960_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_17
  simp only [← c119643, ← c119644, k2, ← c119645] at e_1960_17
  have hr := e_1960_17
  simp only [e_1960_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12174)) := by
  have c119646 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119647 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119648 := (publicBatchWrapper89_copies3739 a h).1
  have c119649 := (publicBatchWrapper89_copies3739 a h).2.1
  have c119650 := (publicBatchWrapper89_copies3739 a h).2.2.1
  have c119651 := (publicBatchWrapper89_copies3739 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_18
  simp only [← c119646, ← c119647, ← c119648] at e_1960_18
  have e_1960_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_19
  simp only [← c119649, ← c119650, k2, ← c119651] at e_1960_19
  have hr := e_1960_19
  simp only [e_1960_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12175)) := by
  have c119652 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.1
  have c119653 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.1
  have c119654 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.1
  have c119655 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.1
  have c119656 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.1
  have c119657 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_0
  simp only [← c119652, ← c119653, ← c119654] at e_1961_0
  have e_1961_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_1
  simp only [← c119655, ← c119656, k2, ← c119657] at e_1961_1
  have hr := e_1961_1
  simp only [e_1961_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12176)) := by
  have c119658 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119659 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119660 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119661 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119662 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119663 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_2
  simp only [← c119658, ← c119659, ← c119660] at e_1961_2
  have e_1961_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_3
  simp only [← c119661, ← c119662, k2, ← c119663] at e_1961_3
  have hr := e_1961_3
  simp only [e_1961_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12177)) := by
  have c119664 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119665 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119666 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119667 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119668 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119669 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_4
  simp only [← c119664, ← c119665, ← c119666] at e_1961_4
  have e_1961_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_5
  simp only [← c119667, ← c119668, k2, ← c119669] at e_1961_5
  have hr := e_1961_5
  simp only [e_1961_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12178)) := by
  have c119670 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119671 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119672 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119673 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119674 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119675 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_6
  simp only [← c119670, ← c119671, ← c119672] at e_1961_6
  have e_1961_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_7
  simp only [← c119673, ← c119674, k2, ← c119675] at e_1961_7
  have hr := e_1961_7
  simp only [e_1961_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12179)) := by
  have c119676 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119677 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119678 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119679 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119680 := (publicBatchWrapper89_copies3740 a h).1
  have c119681 := (publicBatchWrapper89_copies3740 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_8
  simp only [← c119676, ← c119677, ← c119678] at e_1961_8
  have e_1961_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_9
  simp only [← c119679, ← c119680, k2, ← c119681] at e_1961_9
  have hr := e_1961_9
  simp only [e_1961_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12180)) := by
  have c119682 := (publicBatchWrapper89_copies3740 a h).2.2.1
  have c119683 := (publicBatchWrapper89_copies3740 a h).2.2.2.1
  have c119684 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.1
  have c119685 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.1
  have c119686 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.1
  have c119687 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_10
  simp only [← c119682, ← c119683, ← c119684] at e_1961_10
  have e_1961_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_11
  simp only [← c119685, ← c119686, k2, ← c119687] at e_1961_11
  have hr := e_1961_11
  simp only [e_1961_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12181)) := by
  have c119688 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.1
  have c119689 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.1
  have c119690 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119691 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119692 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119693 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_12
  simp only [← c119688, ← c119689, ← c119690] at e_1961_12
  have e_1961_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_13
  simp only [← c119691, ← c119692, k2, ← c119693] at e_1961_13
  have hr := e_1961_13
  simp only [e_1961_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12182)) := by
  have c119694 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119695 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119696 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119697 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119698 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119699 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_14
  simp only [← c119694, ← c119695, ← c119696] at e_1961_14
  have e_1961_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_15
  simp only [← c119697, ← c119698, k2, ← c119699] at e_1961_15
  have hr := e_1961_15
  simp only [e_1961_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12183)) := by
  have c119700 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119701 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119702 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119703 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119704 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119705 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_16
  simp only [← c119700, ← c119701, ← c119702] at e_1961_16
  have e_1961_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_17
  simp only [← c119703, ← c119704, k2, ← c119705] at e_1961_17
  have hr := e_1961_17
  simp only [e_1961_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12184)) := by
  have c119706 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119707 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119708 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119709 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119710 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119711 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_18
  simp only [← c119706, ← c119707, ← c119708] at e_1961_18
  have e_1961_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_19
  simp only [← c119709, ← c119710, k2, ← c119711] at e_1961_19
  have hr := e_1961_19
  simp only [e_1961_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12185)) := by
  have c119712 := (publicBatchWrapper89_copies3741 a h).1
  have c119713 := (publicBatchWrapper89_copies3741 a h).2.1
  have c119714 := (publicBatchWrapper89_copies3741 a h).2.2.1
  have c119715 := (publicBatchWrapper89_copies3741 a h).2.2.2.1
  have c119716 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.1
  have c119717 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_0
  simp only [← c119712, ← c119713, ← c119714] at e_1962_0
  have e_1962_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_1
  simp only [← c119715, ← c119716, k2, ← c119717] at e_1962_1
  have hr := e_1962_1
  simp only [e_1962_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12186)) := by
  have c119718 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.1
  have c119719 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.1
  have c119720 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.1
  have c119721 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.1
  have c119722 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119723 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_2
  simp only [← c119718, ← c119719, ← c119720] at e_1962_2
  have e_1962_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_3
  simp only [← c119721, ← c119722, k2, ← c119723] at e_1962_3
  have hr := e_1962_3
  simp only [e_1962_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12187)) := by
  have c119724 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119725 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119726 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119727 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119728 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119729 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_4
  simp only [← c119724, ← c119725, ← c119726] at e_1962_4
  have e_1962_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_5
  simp only [← c119727, ← c119728, k2, ← c119729] at e_1962_5
  have hr := e_1962_5
  simp only [e_1962_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12188)) := by
  have c119730 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119731 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119732 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119733 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119734 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119735 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_6
  simp only [← c119730, ← c119731, ← c119732] at e_1962_6
  have e_1962_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_7
  simp only [← c119733, ← c119734, k2, ← c119735] at e_1962_7
  have hr := e_1962_7
  simp only [e_1962_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12189)) := by
  have c119736 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119737 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119738 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119739 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119740 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119741 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_8
  simp only [← c119736, ← c119737, ← c119738] at e_1962_8
  have e_1962_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_9
  simp only [← c119739, ← c119740, k2, ← c119741] at e_1962_9
  have hr := e_1962_9
  simp only [e_1962_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12190)) := by
  have c119742 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119743 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119744 := (publicBatchWrapper89_copies3742 a h).1
  have c119745 := (publicBatchWrapper89_copies3742 a h).2.1
  have c119746 := (publicBatchWrapper89_copies3742 a h).2.2.1
  have c119747 := (publicBatchWrapper89_copies3742 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_10
  simp only [← c119742, ← c119743, ← c119744] at e_1962_10
  have e_1962_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_11
  simp only [← c119745, ← c119746, k2, ← c119747] at e_1962_11
  have hr := e_1962_11
  simp only [e_1962_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12191)) := by
  have c119748 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.1
  have c119749 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.1
  have c119750 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.1
  have c119751 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.1
  have c119752 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.1
  have c119753 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_12
  simp only [← c119748, ← c119749, ← c119750] at e_1962_12
  have e_1962_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_13
  simp only [← c119751, ← c119752, k2, ← c119753] at e_1962_13
  have hr := e_1962_13
  simp only [e_1962_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12192)) := by
  have c119754 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119755 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119756 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119757 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119758 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119759 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_14
  simp only [← c119754, ← c119755, ← c119756] at e_1962_14
  have e_1962_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_15
  simp only [← c119757, ← c119758, k2, ← c119759] at e_1962_15
  have hr := e_1962_15
  simp only [e_1962_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12193)) := by
  have c119760 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119761 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119762 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119763 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119764 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119765 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_16
  simp only [← c119760, ← c119761, ← c119762] at e_1962_16
  have e_1962_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_17
  simp only [← c119763, ← c119764, k2, ← c119765] at e_1962_17
  have hr := e_1962_17
  simp only [e_1962_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12194)) := by
  have c119766 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119767 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119768 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119769 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119770 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119771 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_18
  simp only [← c119766, ← c119767, ← c119768] at e_1962_18
  have e_1962_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_19
  simp only [← c119769, ← c119770, k2, ← c119771] at e_1962_19
  have hr := e_1962_19
  simp only [e_1962_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12195)) := by
  have c119772 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119773 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119774 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119775 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119776 := (publicBatchWrapper89_copies3743 a h).1
  have c119777 := (publicBatchWrapper89_copies3743 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_0
  simp only [← c119772, ← c119773, ← c119774] at e_1963_0
  have e_1963_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_1
  simp only [← c119775, ← c119776, k2, ← c119777] at e_1963_1
  have hr := e_1963_1
  simp only [e_1963_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12196)) := by
  have c119778 := (publicBatchWrapper89_copies3743 a h).2.2.1
  have c119779 := (publicBatchWrapper89_copies3743 a h).2.2.2.1
  have c119780 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.1
  have c119781 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.1
  have c119782 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.1
  have c119783 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_2
  simp only [← c119778, ← c119779, ← c119780] at e_1963_2
  have e_1963_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_3
  simp only [← c119781, ← c119782, k2, ← c119783] at e_1963_3
  have hr := e_1963_3
  simp only [e_1963_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12197)) := by
  have c119784 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.1
  have c119785 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.1
  have c119786 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119787 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119788 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119789 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_4
  simp only [← c119784, ← c119785, ← c119786] at e_1963_4
  have e_1963_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_5
  simp only [← c119787, ← c119788, k2, ← c119789] at e_1963_5
  have hr := e_1963_5
  simp only [e_1963_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12198)) := by
  have c119790 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119791 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119792 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119793 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119794 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119795 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_6
  simp only [← c119790, ← c119791, ← c119792] at e_1963_6
  have e_1963_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_7
  simp only [← c119793, ← c119794, k2, ← c119795] at e_1963_7
  have hr := e_1963_7
  simp only [e_1963_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12199)) := by
  have c119796 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119797 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119798 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119799 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119800 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119801 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_8
  simp only [← c119796, ← c119797, ← c119798] at e_1963_8
  have e_1963_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_9
  simp only [← c119799, ← c119800, k2, ← c119801] at e_1963_9
  have hr := e_1963_9
  simp only [e_1963_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12200)) := by
  have c119802 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119803 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119804 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119805 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119806 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119807 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_10
  simp only [← c119802, ← c119803, ← c119804] at e_1963_10
  have e_1963_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_11
  simp only [← c119805, ← c119806, k2, ← c119807] at e_1963_11
  have hr := e_1963_11
  simp only [e_1963_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12201)) := by
  have c119808 := (publicBatchWrapper89_copies3744 a h).1
  have c119809 := (publicBatchWrapper89_copies3744 a h).2.1
  have c119810 := (publicBatchWrapper89_copies3744 a h).2.2.1
  have c119811 := (publicBatchWrapper89_copies3744 a h).2.2.2.1
  have c119812 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.1
  have c119813 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_12
  simp only [← c119808, ← c119809, ← c119810] at e_1963_12
  have e_1963_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_13
  simp only [← c119811, ← c119812, k2, ← c119813] at e_1963_13
  have hr := e_1963_13
  simp only [e_1963_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12370)) := by
  have c119814 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.1
  have c119815 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.1
  have c119816 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.1
  have c119817 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.1
  have c119818 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119819 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_14
  simp only [← c119814, ← c119815, ← c119816] at e_1963_14
  have e_1963_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_15
  simp only [← c119817, ← c119818, k2, ← c119819] at e_1963_15
  have hr := e_1963_15
  simp only [e_1963_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12371)) := by
  have c119820 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119821 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119822 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119823 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119824 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119825 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_16
  simp only [← c119820, ← c119821, ← c119822] at e_1963_16
  have e_1963_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_17
  simp only [← c119823, ← c119824, k2, ← c119825] at e_1963_17
  have hr := e_1963_17
  simp only [e_1963_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12372)) := by
  have c119826 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119827 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119828 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119829 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119830 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119831 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_18
  simp only [← c119826, ← c119827, ← c119828] at e_1963_18
  have e_1963_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_19
  simp only [← c119829, ← c119830, k2, ← c119831] at e_1963_19
  have hr := e_1963_19
  simp only [e_1963_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12373)) := by
  have c119832 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119833 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119834 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119835 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119836 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119837 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_0
  simp only [← c119832, ← c119833, ← c119834] at e_1964_0
  have e_1964_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_1
  simp only [← c119835, ← c119836, k2, ← c119837] at e_1964_1
  have hr := e_1964_1
  simp only [e_1964_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12374)) := by
  have c119838 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119839 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119840 := (publicBatchWrapper89_copies3745 a h).1
  have c119841 := (publicBatchWrapper89_copies3745 a h).2.1
  have c119842 := (publicBatchWrapper89_copies3745 a h).2.2.1
  have c119843 := (publicBatchWrapper89_copies3745 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_2
  simp only [← c119838, ← c119839, ← c119840] at e_1964_2
  have e_1964_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_3
  simp only [← c119841, ← c119842, k2, ← c119843] at e_1964_3
  have hr := e_1964_3
  simp only [e_1964_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12375)) := by
  have c119844 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.1
  have c119845 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.1
  have c119846 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.1
  have c119847 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.1
  have c119848 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.1
  have c119849 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_4
  simp only [← c119844, ← c119845, ← c119846] at e_1964_4
  have e_1964_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_5
  simp only [← c119847, ← c119848, k2, ← c119849] at e_1964_5
  have hr := e_1964_5
  simp only [e_1964_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12376)) := by
  have c119850 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119851 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119852 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119853 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119854 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119855 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_6
  simp only [← c119850, ← c119851, ← c119852] at e_1964_6
  have e_1964_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_7
  simp only [← c119853, ← c119854, k2, ← c119855] at e_1964_7
  have hr := e_1964_7
  simp only [e_1964_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12377)) := by
  have c119856 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119857 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119858 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119859 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119860 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119861 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_8
  simp only [← c119856, ← c119857, ← c119858] at e_1964_8
  have e_1964_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_9
  simp only [← c119859, ← c119860, k2, ← c119861] at e_1964_9
  have hr := e_1964_9
  simp only [e_1964_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12378)) := by
  have c119862 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119863 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119864 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119865 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119866 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119867 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_10
  simp only [← c119862, ← c119863, ← c119864] at e_1964_10
  have e_1964_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_11
  simp only [← c119865, ← c119866, k2, ← c119867] at e_1964_11
  have hr := e_1964_11
  simp only [e_1964_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12379)) := by
  have c119868 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119869 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119870 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119871 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119872 := (publicBatchWrapper89_copies3746 a h).1
  have c119873 := (publicBatchWrapper89_copies3746 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_12
  simp only [← c119868, ← c119869, ← c119870] at e_1964_12
  have e_1964_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_13
  simp only [← c119871, ← c119872, k2, ← c119873] at e_1964_13
  have hr := e_1964_13
  simp only [e_1964_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12380)) := by
  have c119874 := (publicBatchWrapper89_copies3746 a h).2.2.1
  have c119875 := (publicBatchWrapper89_copies3746 a h).2.2.2.1
  have c119876 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.1
  have c119877 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.1
  have c119878 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.1
  have c119879 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_14
  simp only [← c119874, ← c119875, ← c119876] at e_1964_14
  have e_1964_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_15
  simp only [← c119877, ← c119878, k2, ← c119879] at e_1964_15
  have hr := e_1964_15
  simp only [e_1964_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12381)) := by
  have c119880 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.1
  have c119881 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.1
  have c119882 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119883 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119884 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119885 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_16
  simp only [← c119880, ← c119881, ← c119882] at e_1964_16
  have e_1964_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_17
  simp only [← c119883, ← c119884, k2, ← c119885] at e_1964_17
  have hr := e_1964_17
  simp only [e_1964_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12382)) := by
  have c119886 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119887 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119888 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119889 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119890 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119891 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_18
  simp only [← c119886, ← c119887, ← c119888] at e_1964_18
  have e_1964_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_19
  simp only [← c119889, ← c119890, k2, ← c119891] at e_1964_19
  have hr := e_1964_19
  simp only [e_1964_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12383)) := by
  have c119892 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119893 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119894 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119895 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119896 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119897 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_0
  simp only [← c119892, ← c119893, ← c119894] at e_1965_0
  have e_1965_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_1
  simp only [← c119895, ← c119896, k2, ← c119897] at e_1965_1
  have hr := e_1965_1
  simp only [e_1965_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12384)) := by
  have c119898 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119899 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119900 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119901 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119902 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119903 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_2
  simp only [← c119898, ← c119899, ← c119900] at e_1965_2
  have e_1965_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_3
  simp only [← c119901, ← c119902, k2, ← c119903] at e_1965_3
  have hr := e_1965_3
  simp only [e_1965_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12385)) := by
  have c119904 := (publicBatchWrapper89_copies3747 a h).1
  have c119905 := (publicBatchWrapper89_copies3747 a h).2.1
  have c119906 := (publicBatchWrapper89_copies3747 a h).2.2.1
  have c119907 := (publicBatchWrapper89_copies3747 a h).2.2.2.1
  have c119908 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.1
  have c119909 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_4
  simp only [← c119904, ← c119905, ← c119906] at e_1965_4
  have e_1965_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_5
  simp only [← c119907, ← c119908, k2, ← c119909] at e_1965_5
  have hr := e_1965_5
  simp only [e_1965_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12386)) := by
  have c119910 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.1
  have c119911 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.1
  have c119912 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.1
  have c119913 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.1
  have c119914 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119915 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_6
  simp only [← c119910, ← c119911, ← c119912] at e_1965_6
  have e_1965_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_7
  simp only [← c119913, ← c119914, k2, ← c119915] at e_1965_7
  have hr := e_1965_7
  simp only [e_1965_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12387)) := by
  have c119916 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119917 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119918 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119919 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119920 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119921 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_8
  simp only [← c119916, ← c119917, ← c119918] at e_1965_8
  have e_1965_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_9
  simp only [← c119919, ← c119920, k2, ← c119921] at e_1965_9
  have hr := e_1965_9
  simp only [e_1965_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12388)) := by
  have c119922 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119923 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119924 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119925 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119926 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119927 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_10
  simp only [← c119922, ← c119923, ← c119924] at e_1965_10
  have e_1965_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_11
  simp only [← c119925, ← c119926, k2, ← c119927] at e_1965_11
  have hr := e_1965_11
  simp only [e_1965_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12389)) := by
  have c119928 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119929 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119930 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119931 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119932 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119933 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_12
  simp only [← c119928, ← c119929, ← c119930] at e_1965_12
  have e_1965_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_13
  simp only [← c119931, ← c119932, k2, ← c119933] at e_1965_13
  have hr := e_1965_13
  simp only [e_1965_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12390)) := by
  have c119934 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119935 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119936 := (publicBatchWrapper89_copies3748 a h).1
  have c119937 := (publicBatchWrapper89_copies3748 a h).2.1
  have c119938 := (publicBatchWrapper89_copies3748 a h).2.2.1
  have c119939 := (publicBatchWrapper89_copies3748 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_14
  simp only [← c119934, ← c119935, ← c119936] at e_1965_14
  have e_1965_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_15
  simp only [← c119937, ← c119938, k2, ← c119939] at e_1965_15
  have hr := e_1965_15
  simp only [e_1965_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12391)) := by
  have c119940 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.1
  have c119941 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.1
  have c119942 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.1
  have c119943 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.1
  have c119944 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.1
  have c119945 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_16
  simp only [← c119940, ← c119941, ← c119942] at e_1965_16
  have e_1965_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_17
  simp only [← c119943, ← c119944, k2, ← c119945] at e_1965_17
  have hr := e_1965_17
  simp only [e_1965_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12392)) := by
  have c119946 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119947 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119948 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119949 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119950 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119951 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_18
  simp only [← c119946, ← c119947, ← c119948] at e_1965_18
  have e_1965_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_19
  simp only [← c119949, ← c119950, k2, ← c119951] at e_1965_19
  have hr := e_1965_19
  simp only [e_1965_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12393)) := by
  have c119952 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119953 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119954 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119955 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119956 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119957 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_0
  simp only [← c119952, ← c119953, ← c119954] at e_1966_0
  have e_1966_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_1
  simp only [← c119955, ← c119956, k2, ← c119957] at e_1966_1
  have hr := e_1966_1
  simp only [e_1966_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12394)) := by
  have c119958 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119959 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119960 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119961 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119962 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119963 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_2
  simp only [← c119958, ← c119959, ← c119960] at e_1966_2
  have e_1966_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_3
  simp only [← c119961, ← c119962, k2, ← c119963] at e_1966_3
  have hr := e_1966_3
  simp only [e_1966_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12395)) := by
  have c119964 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119965 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119966 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119967 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119968 := (publicBatchWrapper89_copies3749 a h).1
  have c119969 := (publicBatchWrapper89_copies3749 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_4
  simp only [← c119964, ← c119965, ← c119966] at e_1966_4
  have e_1966_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_5
  simp only [← c119967, ← c119968, k2, ← c119969] at e_1966_5
  have hr := e_1966_5
  simp only [e_1966_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12396)) := by
  have c119970 := (publicBatchWrapper89_copies3749 a h).2.2.1
  have c119971 := (publicBatchWrapper89_copies3749 a h).2.2.2.1
  have c119972 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.1
  have c119973 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.1
  have c119974 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.1
  have c119975 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_6
  simp only [← c119970, ← c119971, ← c119972] at e_1966_6
  have e_1966_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_7
  simp only [← c119973, ← c119974, k2, ← c119975] at e_1966_7
  have hr := e_1966_7
  simp only [e_1966_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12397)) := by
  have c119976 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.1
  have c119977 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.1
  have c119978 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119979 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119980 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119981 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_8
  simp only [← c119976, ← c119977, ← c119978] at e_1966_8
  have e_1966_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_9
  simp only [← c119979, ← c119980, k2, ← c119981] at e_1966_9
  have hr := e_1966_9
  simp only [e_1966_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12398)) := by
  have c119982 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119983 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119984 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119985 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119986 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119987 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_10
  simp only [← c119982, ← c119983, ← c119984] at e_1966_10
  have e_1966_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_11
  simp only [← c119985, ← c119986, k2, ← c119987] at e_1966_11
  have hr := e_1966_11
  simp only [e_1966_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12399)) := by
  have c119988 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119989 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119990 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119991 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119992 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119993 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_12
  simp only [← c119988, ← c119989, ← c119990] at e_1966_12
  have e_1966_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_13
  simp only [← c119991, ← c119992, k2, ← c119993] at e_1966_13
  have hr := e_1966_13
  simp only [e_1966_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12400)) := by
  have c119994 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119995 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119996 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119997 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119998 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119999 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_14
  simp only [← c119994, ← c119995, ← c119996] at e_1966_14
  have e_1966_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_15
  simp only [← c119997, ← c119998, k2, ← c119999] at e_1966_15
  have hr := e_1966_15
  simp only [e_1966_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12401)) := by
  have c120000 := (publicBatchWrapper89_copies3750 a h).1
  have c120001 := (publicBatchWrapper89_copies3750 a h).2.1
  have c120002 := (publicBatchWrapper89_copies3750 a h).2.2.1
  have c120003 := (publicBatchWrapper89_copies3750 a h).2.2.2.1
  have c120004 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.1
  have c120005 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_16
  simp only [← c120000, ← c120001, ← c120002] at e_1966_16
  have e_1966_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_17
  simp only [← c120003, ← c120004, k2, ← c120005] at e_1966_17
  have hr := e_1966_17
  simp only [e_1966_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12570)) := by
  have c120006 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.1
  have c120007 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.1
  have c120008 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.1
  have c120009 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.1
  have c120010 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120011 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_18
  simp only [← c120006, ← c120007, ← c120008] at e_1966_18
  have e_1966_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_19
  simp only [← c120009, ← c120010, k2, ← c120011] at e_1966_19
  have hr := e_1966_19
  simp only [e_1966_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12571)) := by
  have c120012 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120013 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120014 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120015 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120016 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120017 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_0
  simp only [← c120012, ← c120013, ← c120014] at e_1967_0
  have e_1967_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_1
  simp only [← c120015, ← c120016, k2, ← c120017] at e_1967_1
  have hr := e_1967_1
  simp only [e_1967_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12572)) := by
  have c120018 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120019 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120020 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120021 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120022 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120023 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_2
  simp only [← c120018, ← c120019, ← c120020] at e_1967_2
  have e_1967_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_3
  simp only [← c120021, ← c120022, k2, ← c120023] at e_1967_3
  have hr := e_1967_3
  simp only [e_1967_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12573)) := by
  have c120024 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120025 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120026 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120027 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120028 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120029 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_4
  simp only [← c120024, ← c120025, ← c120026] at e_1967_4
  have e_1967_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_5
  simp only [← c120027, ← c120028, k2, ← c120029] at e_1967_5
  have hr := e_1967_5
  simp only [e_1967_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12574)) := by
  have c120030 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120031 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120032 := (publicBatchWrapper89_copies3751 a h).1
  have c120033 := (publicBatchWrapper89_copies3751 a h).2.1
  have c120034 := (publicBatchWrapper89_copies3751 a h).2.2.1
  have c120035 := (publicBatchWrapper89_copies3751 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_6
  simp only [← c120030, ← c120031, ← c120032] at e_1967_6
  have e_1967_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_7
  simp only [← c120033, ← c120034, k2, ← c120035] at e_1967_7
  have hr := e_1967_7
  simp only [e_1967_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12575)) := by
  have c120036 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.1
  have c120037 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.1
  have c120038 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.1
  have c120039 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.1
  have c120040 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.1
  have c120041 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_8
  simp only [← c120036, ← c120037, ← c120038] at e_1967_8
  have e_1967_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_9
  simp only [← c120039, ← c120040, k2, ← c120041] at e_1967_9
  have hr := e_1967_9
  simp only [e_1967_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12576)) := by
  have c120042 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120043 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120044 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120045 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120046 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120047 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_10
  simp only [← c120042, ← c120043, ← c120044] at e_1967_10
  have e_1967_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_11
  simp only [← c120045, ← c120046, k2, ← c120047] at e_1967_11
  have hr := e_1967_11
  simp only [e_1967_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12577)) := by
  have c120048 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120049 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120050 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120051 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120052 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120053 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_12
  simp only [← c120048, ← c120049, ← c120050] at e_1967_12
  have e_1967_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_13
  simp only [← c120051, ← c120052, k2, ← c120053] at e_1967_13
  have hr := e_1967_13
  simp only [e_1967_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12578)) := by
  have c120054 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120055 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120056 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120057 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120058 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120059 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_14
  simp only [← c120054, ← c120055, ← c120056] at e_1967_14
  have e_1967_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_15
  simp only [← c120057, ← c120058, k2, ← c120059] at e_1967_15
  have hr := e_1967_15
  simp only [e_1967_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12579)) := by
  have c120060 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120061 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120062 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120063 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120064 := (publicBatchWrapper89_copies3752 a h).1
  have c120065 := (publicBatchWrapper89_copies3752 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_16
  simp only [← c120060, ← c120061, ← c120062] at e_1967_16
  have e_1967_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_17
  simp only [← c120063, ← c120064, k2, ← c120065] at e_1967_17
  have hr := e_1967_17
  simp only [e_1967_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12580)) := by
  have c120066 := (publicBatchWrapper89_copies3752 a h).2.2.1
  have c120067 := (publicBatchWrapper89_copies3752 a h).2.2.2.1
  have c120068 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.1
  have c120069 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.1
  have c120070 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.1
  have c120071 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_18
  simp only [← c120066, ← c120067, ← c120068] at e_1967_18
  have e_1967_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_19
  simp only [← c120069, ← c120070, k2, ← c120071] at e_1967_19
  have hr := e_1967_19
  simp only [e_1967_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12581)) := by
  have c120072 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.1
  have c120073 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.1
  have c120074 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120075 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120076 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120077 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_0
  simp only [← c120072, ← c120073, ← c120074] at e_1968_0
  have e_1968_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_1
  simp only [← c120075, ← c120076, k2, ← c120077] at e_1968_1
  have hr := e_1968_1
  simp only [e_1968_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12582)) := by
  have c120078 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120079 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120080 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120081 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120082 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120083 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_2
  simp only [← c120078, ← c120079, ← c120080] at e_1968_2
  have e_1968_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_3
  simp only [← c120081, ← c120082, k2, ← c120083] at e_1968_3
  have hr := e_1968_3
  simp only [e_1968_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12583)) := by
  have c120084 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120085 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120086 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120087 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120088 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120089 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_4
  simp only [← c120084, ← c120085, ← c120086] at e_1968_4
  have e_1968_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_5
  simp only [← c120087, ← c120088, k2, ← c120089] at e_1968_5
  have hr := e_1968_5
  simp only [e_1968_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12584)) := by
  have c120090 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120091 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120092 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120093 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120094 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120095 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_6
  simp only [← c120090, ← c120091, ← c120092] at e_1968_6
  have e_1968_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_7
  simp only [← c120093, ← c120094, k2, ← c120095] at e_1968_7
  have hr := e_1968_7
  simp only [e_1968_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12585)) := by
  have c120096 := (publicBatchWrapper89_copies3753 a h).1
  have c120097 := (publicBatchWrapper89_copies3753 a h).2.1
  have c120098 := (publicBatchWrapper89_copies3753 a h).2.2.1
  have c120099 := (publicBatchWrapper89_copies3753 a h).2.2.2.1
  have c120100 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.1
  have c120101 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_8
  simp only [← c120096, ← c120097, ← c120098] at e_1968_8
  have e_1968_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_9
  simp only [← c120099, ← c120100, k2, ← c120101] at e_1968_9
  have hr := e_1968_9
  simp only [e_1968_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12586)) := by
  have c120102 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.1
  have c120103 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.1
  have c120104 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.1
  have c120105 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.1
  have c120106 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120107 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_10
  simp only [← c120102, ← c120103, ← c120104] at e_1968_10
  have e_1968_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_11
  simp only [← c120105, ← c120106, k2, ← c120107] at e_1968_11
  have hr := e_1968_11
  simp only [e_1968_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12587)) := by
  have c120108 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120109 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120110 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120111 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120112 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120113 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_12
  simp only [← c120108, ← c120109, ← c120110] at e_1968_12
  have e_1968_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_13
  simp only [← c120111, ← c120112, k2, ← c120113] at e_1968_13
  have hr := e_1968_13
  simp only [e_1968_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12588)) := by
  have c120114 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120115 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120116 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120117 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120118 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120119 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_14
  simp only [← c120114, ← c120115, ← c120116] at e_1968_14
  have e_1968_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_15
  simp only [← c120117, ← c120118, k2, ← c120119] at e_1968_15
  have hr := e_1968_15
  simp only [e_1968_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12589)) := by
  have c120120 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120121 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120122 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120123 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120124 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120125 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_16
  simp only [← c120120, ← c120121, ← c120122] at e_1968_16
  have e_1968_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_17
  simp only [← c120123, ← c120124, k2, ← c120125] at e_1968_17
  have hr := e_1968_17
  simp only [e_1968_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12590)) := by
  have c120126 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120127 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120128 := (publicBatchWrapper89_copies3754 a h).1
  have c120129 := (publicBatchWrapper89_copies3754 a h).2.1
  have c120130 := (publicBatchWrapper89_copies3754 a h).2.2.1
  have c120131 := (publicBatchWrapper89_copies3754 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_18
  simp only [← c120126, ← c120127, ← c120128] at e_1968_18
  have e_1968_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_19
  simp only [← c120129, ← c120130, k2, ← c120131] at e_1968_19
  have hr := e_1968_19
  simp only [e_1968_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12591)) := by
  have c120132 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.1
  have c120133 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.1
  have c120134 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.1
  have c120135 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.1
  have c120136 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.1
  have c120137 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_0
  simp only [← c120132, ← c120133, ← c120134] at e_1969_0
  have e_1969_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_1
  simp only [← c120135, ← c120136, k2, ← c120137] at e_1969_1
  have hr := e_1969_1
  simp only [e_1969_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12592)) := by
  have c120138 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120139 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120140 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120141 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120142 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120143 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_2
  simp only [← c120138, ← c120139, ← c120140] at e_1969_2
  have e_1969_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_3
  simp only [← c120141, ← c120142, k2, ← c120143] at e_1969_3
  have hr := e_1969_3
  simp only [e_1969_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12593)) := by
  have c120144 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120145 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120146 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120147 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120148 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120149 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_4
  simp only [← c120144, ← c120145, ← c120146] at e_1969_4
  have e_1969_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_5
  simp only [← c120147, ← c120148, k2, ← c120149] at e_1969_5
  have hr := e_1969_5
  simp only [e_1969_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12594)) := by
  have c120150 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120151 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120152 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120153 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120154 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120155 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_6
  simp only [← c120150, ← c120151, ← c120152] at e_1969_6
  have e_1969_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_7
  simp only [← c120153, ← c120154, k2, ← c120155] at e_1969_7
  have hr := e_1969_7
  simp only [e_1969_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12595)) := by
  have c120156 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120157 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120158 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120159 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120160 := (publicBatchWrapper89_copies3755 a h).1
  have c120161 := (publicBatchWrapper89_copies3755 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_8
  simp only [← c120156, ← c120157, ← c120158] at e_1969_8
  have e_1969_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_9
  simp only [← c120159, ← c120160, k2, ← c120161] at e_1969_9
  have hr := e_1969_9
  simp only [e_1969_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12596)) := by
  have c120162 := (publicBatchWrapper89_copies3755 a h).2.2.1
  have c120163 := (publicBatchWrapper89_copies3755 a h).2.2.2.1
  have c120164 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.1
  have c120165 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.1
  have c120166 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.1
  have c120167 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_10
  simp only [← c120162, ← c120163, ← c120164] at e_1969_10
  have e_1969_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_11
  simp only [← c120165, ← c120166, k2, ← c120167] at e_1969_11
  have hr := e_1969_11
  simp only [e_1969_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12597)) := by
  have c120168 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.1
  have c120169 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.1
  have c120170 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120171 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120172 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120173 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_12
  simp only [← c120168, ← c120169, ← c120170] at e_1969_12
  have e_1969_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_13
  simp only [← c120171, ← c120172, k2, ← c120173] at e_1969_13
  have hr := e_1969_13
  simp only [e_1969_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12598)) := by
  have c120174 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120175 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120176 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120177 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120178 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120179 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_14
  simp only [← c120174, ← c120175, ← c120176] at e_1969_14
  have e_1969_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_15
  simp only [← c120177, ← c120178, k2, ← c120179] at e_1969_15
  have hr := e_1969_15
  simp only [e_1969_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12599)) := by
  have c120180 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120181 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120182 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120183 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120184 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120185 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_16
  simp only [← c120180, ← c120181, ← c120182] at e_1969_16
  have e_1969_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_17
  simp only [← c120183, ← c120184, k2, ← c120185] at e_1969_17
  have hr := e_1969_17
  simp only [e_1969_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12600)) := by
  have c120186 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120187 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120188 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120189 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120190 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120191 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_18
  simp only [← c120186, ← c120187, ← c120188] at e_1969_18
  have e_1969_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_19
  simp only [← c120189, ← c120190, k2, ← c120191] at e_1969_19
  have hr := e_1969_19
  simp only [e_1969_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12601)) := by
  have c120192 := (publicBatchWrapper89_copies3756 a h).1
  have c120193 := (publicBatchWrapper89_copies3756 a h).2.1
  have c120194 := (publicBatchWrapper89_copies3756 a h).2.2.1
  have c120195 := (publicBatchWrapper89_copies3756 a h).2.2.2.1
  have c120196 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.1
  have c120197 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_0
  simp only [← c120192, ← c120193, ← c120194] at e_1970_0
  have e_1970_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_1
  simp only [← c120195, ← c120196, k2, ← c120197] at e_1970_1
  have hr := e_1970_1
  simp only [e_1970_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12770)) := by
  have c120198 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.1
  have c120199 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.1
  have c120200 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.1
  have c120201 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.1
  have c120202 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120203 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_2
  simp only [← c120198, ← c120199, ← c120200] at e_1970_2
  have e_1970_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_3
  simp only [← c120201, ← c120202, k2, ← c120203] at e_1970_3
  have hr := e_1970_3
  simp only [e_1970_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12771)) := by
  have c120204 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120205 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120206 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120207 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120208 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120209 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_4
  simp only [← c120204, ← c120205, ← c120206] at e_1970_4
  have e_1970_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_5
  simp only [← c120207, ← c120208, k2, ← c120209] at e_1970_5
  have hr := e_1970_5
  simp only [e_1970_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12772)) := by
  have c120210 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120211 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120212 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120213 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120214 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120215 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_6
  simp only [← c120210, ← c120211, ← c120212] at e_1970_6
  have e_1970_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_7
  simp only [← c120213, ← c120214, k2, ← c120215] at e_1970_7
  have hr := e_1970_7
  simp only [e_1970_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12773)) := by
  have c120216 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120217 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120218 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120219 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120220 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120221 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_8
  simp only [← c120216, ← c120217, ← c120218] at e_1970_8
  have e_1970_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_9
  simp only [← c120219, ← c120220, k2, ← c120221] at e_1970_9
  have hr := e_1970_9
  simp only [e_1970_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12774)) := by
  have c120222 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120223 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120224 := (publicBatchWrapper89_copies3757 a h).1
  have c120225 := (publicBatchWrapper89_copies3757 a h).2.1
  have c120226 := (publicBatchWrapper89_copies3757 a h).2.2.1
  have c120227 := (publicBatchWrapper89_copies3757 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_10
  simp only [← c120222, ← c120223, ← c120224] at e_1970_10
  have e_1970_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_11
  simp only [← c120225, ← c120226, k2, ← c120227] at e_1970_11
  have hr := e_1970_11
  simp only [e_1970_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12775)) := by
  have c120228 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.1
  have c120229 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.1
  have c120230 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.1
  have c120231 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.1
  have c120232 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.1
  have c120233 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_12
  simp only [← c120228, ← c120229, ← c120230] at e_1970_12
  have e_1970_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_13
  simp only [← c120231, ← c120232, k2, ← c120233] at e_1970_13
  have hr := e_1970_13
  simp only [e_1970_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12776)) := by
  have c120234 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120235 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120236 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120237 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120238 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120239 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_14
  simp only [← c120234, ← c120235, ← c120236] at e_1970_14
  have e_1970_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_15
  simp only [← c120237, ← c120238, k2, ← c120239] at e_1970_15
  have hr := e_1970_15
  simp only [e_1970_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12777)) := by
  have c120240 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120241 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120242 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120243 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120244 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120245 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_16
  simp only [← c120240, ← c120241, ← c120242] at e_1970_16
  have e_1970_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_17
  simp only [← c120243, ← c120244, k2, ← c120245] at e_1970_17
  have hr := e_1970_17
  simp only [e_1970_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12778)) := by
  have c120246 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120247 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120248 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120249 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120250 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120251 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_18
  simp only [← c120246, ← c120247, ← c120248] at e_1970_18
  have e_1970_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_19
  simp only [← c120249, ← c120250, k2, ← c120251] at e_1970_19
  have hr := e_1970_19
  simp only [e_1970_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12779)) := by
  have c120252 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120253 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120254 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120255 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120256 := (publicBatchWrapper89_copies3758 a h).1
  have c120257 := (publicBatchWrapper89_copies3758 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_0
  simp only [← c120252, ← c120253, ← c120254] at e_1971_0
  have e_1971_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_1
  simp only [← c120255, ← c120256, k2, ← c120257] at e_1971_1
  have hr := e_1971_1
  simp only [e_1971_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12780)) := by
  have c120258 := (publicBatchWrapper89_copies3758 a h).2.2.1
  have c120259 := (publicBatchWrapper89_copies3758 a h).2.2.2.1
  have c120260 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.1
  have c120261 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.1
  have c120262 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.1
  have c120263 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_2
  simp only [← c120258, ← c120259, ← c120260] at e_1971_2
  have e_1971_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_3
  simp only [← c120261, ← c120262, k2, ← c120263] at e_1971_3
  have hr := e_1971_3
  simp only [e_1971_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12781)) := by
  have c120264 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.1
  have c120265 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.1
  have c120266 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120267 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120268 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120269 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_4
  simp only [← c120264, ← c120265, ← c120266] at e_1971_4
  have e_1971_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_5
  simp only [← c120267, ← c120268, k2, ← c120269] at e_1971_5
  have hr := e_1971_5
  simp only [e_1971_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12782)) := by
  have c120270 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120271 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120272 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120273 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120274 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120275 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_6
  simp only [← c120270, ← c120271, ← c120272] at e_1971_6
  have e_1971_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_7
  simp only [← c120273, ← c120274, k2, ← c120275] at e_1971_7
  have hr := e_1971_7
  simp only [e_1971_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12783)) := by
  have c120276 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120277 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120278 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120279 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120280 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120281 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_8
  simp only [← c120276, ← c120277, ← c120278] at e_1971_8
  have e_1971_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_9
  simp only [← c120279, ← c120280, k2, ← c120281] at e_1971_9
  have hr := e_1971_9
  simp only [e_1971_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12784)) := by
  have c120282 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120283 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120284 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120285 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120286 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120287 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_10
  simp only [← c120282, ← c120283, ← c120284] at e_1971_10
  have e_1971_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_11
  simp only [← c120285, ← c120286, k2, ← c120287] at e_1971_11
  have hr := e_1971_11
  simp only [e_1971_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12785)) := by
  have c120288 := (publicBatchWrapper89_copies3759 a h).1
  have c120289 := (publicBatchWrapper89_copies3759 a h).2.1
  have c120290 := (publicBatchWrapper89_copies3759 a h).2.2.1
  have c120291 := (publicBatchWrapper89_copies3759 a h).2.2.2.1
  have c120292 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.1
  have c120293 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_12
  simp only [← c120288, ← c120289, ← c120290] at e_1971_12
  have e_1971_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_13
  simp only [← c120291, ← c120292, k2, ← c120293] at e_1971_13
  have hr := e_1971_13
  simp only [e_1971_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12786)) := by
  have c120294 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.1
  have c120295 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.1
  have c120296 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.1
  have c120297 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.1
  have c120298 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120299 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_14
  simp only [← c120294, ← c120295, ← c120296] at e_1971_14
  have e_1971_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_15
  simp only [← c120297, ← c120298, k2, ← c120299] at e_1971_15
  have hr := e_1971_15
  simp only [e_1971_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12787)) := by
  have c120300 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120301 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120302 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120303 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120304 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120305 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_16
  simp only [← c120300, ← c120301, ← c120302] at e_1971_16
  have e_1971_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_17
  simp only [← c120303, ← c120304, k2, ← c120305] at e_1971_17
  have hr := e_1971_17
  simp only [e_1971_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12788)) := by
  have c120306 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120307 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120308 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120309 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120310 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120311 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_18
  simp only [← c120306, ← c120307, ← c120308] at e_1971_18
  have e_1971_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_19
  simp only [← c120309, ← c120310, k2, ← c120311] at e_1971_19
  have hr := e_1971_19
  simp only [e_1971_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12789)) := by
  have c120312 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120313 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120314 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120315 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120316 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120317 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_0
  simp only [← c120312, ← c120313, ← c120314] at e_1972_0
  have e_1972_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_1
  simp only [← c120315, ← c120316, k2, ← c120317] at e_1972_1
  have hr := e_1972_1
  simp only [e_1972_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12790)) := by
  have c120318 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120319 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120320 := (publicBatchWrapper89_copies3760 a h).1
  have c120321 := (publicBatchWrapper89_copies3760 a h).2.1
  have c120322 := (publicBatchWrapper89_copies3760 a h).2.2.1
  have c120323 := (publicBatchWrapper89_copies3760 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_2
  simp only [← c120318, ← c120319, ← c120320] at e_1972_2
  have e_1972_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_3
  simp only [← c120321, ← c120322, k2, ← c120323] at e_1972_3
  have hr := e_1972_3
  simp only [e_1972_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12791)) := by
  have c120324 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.1
  have c120325 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.1
  have c120326 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.1
  have c120327 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.1
  have c120328 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.1
  have c120329 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_4
  simp only [← c120324, ← c120325, ← c120326] at e_1972_4
  have e_1972_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_5
  simp only [← c120327, ← c120328, k2, ← c120329] at e_1972_5
  have hr := e_1972_5
  simp only [e_1972_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12792)) := by
  have c120330 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120331 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120332 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120333 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120334 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120335 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_6
  simp only [← c120330, ← c120331, ← c120332] at e_1972_6
  have e_1972_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_7
  simp only [← c120333, ← c120334, k2, ← c120335] at e_1972_7
  have hr := e_1972_7
  simp only [e_1972_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12793)) := by
  have c120336 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120337 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120338 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120339 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120340 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120341 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_8
  simp only [← c120336, ← c120337, ← c120338] at e_1972_8
  have e_1972_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_9
  simp only [← c120339, ← c120340, k2, ← c120341] at e_1972_9
  have hr := e_1972_9
  simp only [e_1972_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12794)) := by
  have c120342 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120343 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120344 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120345 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120346 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120347 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_10
  simp only [← c120342, ← c120343, ← c120344] at e_1972_10
  have e_1972_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_11
  simp only [← c120345, ← c120346, k2, ← c120347] at e_1972_11
  have hr := e_1972_11
  simp only [e_1972_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12795)) := by
  have c120348 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120349 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120350 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120351 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120352 := (publicBatchWrapper89_copies3761 a h).1
  have c120353 := (publicBatchWrapper89_copies3761 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_12
  simp only [← c120348, ← c120349, ← c120350] at e_1972_12
  have e_1972_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_13
  simp only [← c120351, ← c120352, k2, ← c120353] at e_1972_13
  have hr := e_1972_13
  simp only [e_1972_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12796)) := by
  have c120354 := (publicBatchWrapper89_copies3761 a h).2.2.1
  have c120355 := (publicBatchWrapper89_copies3761 a h).2.2.2.1
  have c120356 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.1
  have c120357 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.1
  have c120358 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.1
  have c120359 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_14
  simp only [← c120354, ← c120355, ← c120356] at e_1972_14
  have e_1972_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_15
  simp only [← c120357, ← c120358, k2, ← c120359] at e_1972_15
  have hr := e_1972_15
  simp only [e_1972_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12797)) := by
  have c120360 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.1
  have c120361 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.1
  have c120362 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120363 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120364 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120365 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_16
  simp only [← c120360, ← c120361, ← c120362] at e_1972_16
  have e_1972_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_17
  simp only [← c120363, ← c120364, k2, ← c120365] at e_1972_17
  have hr := e_1972_17
  simp only [e_1972_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12798)) := by
  have c120366 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120367 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120368 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120369 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120370 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120371 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_18
  simp only [← c120366, ← c120367, ← c120368] at e_1972_18
  have e_1972_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_19
  simp only [← c120369, ← c120370, k2, ← c120371] at e_1972_19
  have hr := e_1972_19
  simp only [e_1972_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12799)) := by
  have c120372 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120373 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120374 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120375 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120376 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120377 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_0
  simp only [← c120372, ← c120373, ← c120374] at e_1973_0
  have e_1973_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_1
  simp only [← c120375, ← c120376, k2, ← c120377] at e_1973_1
  have hr := e_1973_1
  simp only [e_1973_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12800)) := by
  have c120378 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120379 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120380 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120381 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120382 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120383 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_2
  simp only [← c120378, ← c120379, ← c120380] at e_1973_2
  have e_1973_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_3
  simp only [← c120381, ← c120382, k2, ← c120383] at e_1973_3
  have hr := e_1973_3
  simp only [e_1973_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12801)) := by
  have c120384 := (publicBatchWrapper89_copies3762 a h).1
  have c120385 := (publicBatchWrapper89_copies3762 a h).2.1
  have c120386 := (publicBatchWrapper89_copies3762 a h).2.2.1
  have c120387 := (publicBatchWrapper89_copies3762 a h).2.2.2.1
  have c120388 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.1
  have c120389 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_4
  simp only [← c120384, ← c120385, ← c120386] at e_1973_4
  have e_1973_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_5
  simp only [← c120387, ← c120388, k2, ← c120389] at e_1973_5
  have hr := e_1973_5
  simp only [e_1973_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12970)) := by
  have c120390 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.1
  have c120391 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.1
  have c120392 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.1
  have c120393 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.1
  have c120394 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120395 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_6
  simp only [← c120390, ← c120391, ← c120392] at e_1973_6
  have e_1973_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_7
  simp only [← c120393, ← c120394, k2, ← c120395] at e_1973_7
  have hr := e_1973_7
  simp only [e_1973_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12971)) := by
  have c120396 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120397 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120398 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120399 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120400 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120401 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_8
  simp only [← c120396, ← c120397, ← c120398] at e_1973_8
  have e_1973_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_9
  simp only [← c120399, ← c120400, k2, ← c120401] at e_1973_9
  have hr := e_1973_9
  simp only [e_1973_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12972)) := by
  have c120402 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120403 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120404 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120405 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120406 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120407 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_10
  simp only [← c120402, ← c120403, ← c120404] at e_1973_10
  have e_1973_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_11
  simp only [← c120405, ← c120406, k2, ← c120407] at e_1973_11
  have hr := e_1973_11
  simp only [e_1973_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12973)) := by
  have c120408 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120409 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120410 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120411 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120412 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120413 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_12
  simp only [← c120408, ← c120409, ← c120410] at e_1973_12
  have e_1973_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_13
  simp only [← c120411, ← c120412, k2, ← c120413] at e_1973_13
  have hr := e_1973_13
  simp only [e_1973_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12974)) := by
  have c120414 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120415 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120416 := (publicBatchWrapper89_copies3763 a h).1
  have c120417 := (publicBatchWrapper89_copies3763 a h).2.1
  have c120418 := (publicBatchWrapper89_copies3763 a h).2.2.1
  have c120419 := (publicBatchWrapper89_copies3763 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_14
  simp only [← c120414, ← c120415, ← c120416] at e_1973_14
  have e_1973_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_15
  simp only [← c120417, ← c120418, k2, ← c120419] at e_1973_15
  have hr := e_1973_15
  simp only [e_1973_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12975)) := by
  have c120420 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.1
  have c120421 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.1
  have c120422 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.1
  have c120423 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.1
  have c120424 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.1
  have c120425 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_16
  simp only [← c120420, ← c120421, ← c120422] at e_1973_16
  have e_1973_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_17
  simp only [← c120423, ← c120424, k2, ← c120425] at e_1973_17
  have hr := e_1973_17
  simp only [e_1973_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12976)) := by
  have c120426 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120427 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120428 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120429 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120430 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120431 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_18
  simp only [← c120426, ← c120427, ← c120428] at e_1973_18
  have e_1973_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_19
  simp only [← c120429, ← c120430, k2, ← c120431] at e_1973_19
  have hr := e_1973_19
  simp only [e_1973_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12977)) := by
  have c120432 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120433 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120434 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120435 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120436 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120437 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_0
  simp only [← c120432, ← c120433, ← c120434] at e_1974_0
  have e_1974_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_1
  simp only [← c120435, ← c120436, k2, ← c120437] at e_1974_1
  have hr := e_1974_1
  simp only [e_1974_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12978)) := by
  have c120438 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120439 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120440 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120441 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120442 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120443 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_2
  simp only [← c120438, ← c120439, ← c120440] at e_1974_2
  have e_1974_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_3
  simp only [← c120441, ← c120442, k2, ← c120443] at e_1974_3
  have hr := e_1974_3
  simp only [e_1974_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12979)) := by
  have c120444 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120445 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120446 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120447 := (publicBatchWrapper89_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120448 := (publicBatchWrapper89_copies3764 a h).1
  have c120449 := (publicBatchWrapper89_copies3764 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_4
  simp only [← c120444, ← c120445, ← c120446] at e_1974_4
  have e_1974_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_5
  simp only [← c120447, ← c120448, k2, ← c120449] at e_1974_5
  have hr := e_1974_5
  simp only [e_1974_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12980)) := by
  have c120450 := (publicBatchWrapper89_copies3764 a h).2.2.1
  have c120451 := (publicBatchWrapper89_copies3764 a h).2.2.2.1
  have c120452 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.1
  have c120453 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.1
  have c120454 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.1
  have c120455 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_6
  simp only [← c120450, ← c120451, ← c120452] at e_1974_6
  have e_1974_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_7
  simp only [← c120453, ← c120454, k2, ← c120455] at e_1974_7
  have hr := e_1974_7
  simp only [e_1974_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12981)) := by
  have c120456 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.1
  have c120457 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.1
  have c120458 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120459 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120460 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120461 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_8
  simp only [← c120456, ← c120457, ← c120458] at e_1974_8
  have e_1974_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_9
  simp only [← c120459, ← c120460, k2, ← c120461] at e_1974_9
  have hr := e_1974_9
  simp only [e_1974_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12982)) := by
  have c120462 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120463 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120464 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120465 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120466 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120467 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_10
  simp only [← c120462, ← c120463, ← c120464] at e_1974_10
  have e_1974_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_11
  simp only [← c120465, ← c120466, k2, ← c120467] at e_1974_11
  have hr := e_1974_11
  simp only [e_1974_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12983)) := by
  have c120468 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120469 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120470 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120471 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120472 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120473 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_12
  simp only [← c120468, ← c120469, ← c120470] at e_1974_12
  have e_1974_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_13
  simp only [← c120471, ← c120472, k2, ← c120473] at e_1974_13
  have hr := e_1974_13
  simp only [e_1974_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12984)) := by
  have c120474 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120475 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120476 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120477 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120478 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120479 := (publicBatchWrapper89_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_14
  simp only [← c120474, ← c120475, ← c120476] at e_1974_14
  have e_1974_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_15
  simp only [← c120477, ← c120478, k2, ← c120479] at e_1974_15
  have hr := e_1974_15
  simp only [e_1974_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12985)) := by
  have c120480 := (publicBatchWrapper89_copies3765 a h).1
  have c120481 := (publicBatchWrapper89_copies3765 a h).2.1
  have c120482 := (publicBatchWrapper89_copies3765 a h).2.2.1
  have c120483 := (publicBatchWrapper89_copies3765 a h).2.2.2.1
  have c120484 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.1
  have c120485 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_16
  simp only [← c120480, ← c120481, ← c120482] at e_1974_16
  have e_1974_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_17
  simp only [← c120483, ← c120484, k2, ← c120485] at e_1974_17
  have hr := e_1974_17
  simp only [e_1974_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1974 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12986)) := by
  have c120486 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.1
  have c120487 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.1
  have c120488 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.1
  have c120489 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.1
  have c120490 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120491 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1974_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_18
  simp only [← c120486, ← c120487, ← c120488] at e_1974_18
  have e_1974_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_19
  simp only [← c120489, ← c120490, k2, ← c120491] at e_1974_19
  have hr := e_1974_19
  simp only [e_1974_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12987)) := by
  have c120492 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120493 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120494 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120495 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120496 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120497 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_0
  simp only [← c120492, ← c120493, ← c120494] at e_1975_0
  have e_1975_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_1
  simp only [← c120495, ← c120496, k2, ← c120497] at e_1975_1
  have hr := e_1975_1
  simp only [e_1975_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12988)) := by
  have c120498 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120499 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120500 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120501 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120502 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120503 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_2
  simp only [← c120498, ← c120499, ← c120500] at e_1975_2
  have e_1975_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_3
  simp only [← c120501, ← c120502, k2, ← c120503] at e_1975_3
  have hr := e_1975_3
  simp only [e_1975_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12989)) := by
  have c120504 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120505 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120506 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120507 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120508 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120509 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_4
  simp only [← c120504, ← c120505, ← c120506] at e_1975_4
  have e_1975_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_5
  simp only [← c120507, ← c120508, k2, ← c120509] at e_1975_5
  have hr := e_1975_5
  simp only [e_1975_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12990)) := by
  have c120510 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120511 := (publicBatchWrapper89_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120512 := (publicBatchWrapper89_copies3766 a h).1
  have c120513 := (publicBatchWrapper89_copies3766 a h).2.1
  have c120514 := (publicBatchWrapper89_copies3766 a h).2.2.1
  have c120515 := (publicBatchWrapper89_copies3766 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_6
  simp only [← c120510, ← c120511, ← c120512] at e_1975_6
  have e_1975_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_7
  simp only [← c120513, ← c120514, k2, ← c120515] at e_1975_7
  have hr := e_1975_7
  simp only [e_1975_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12991)) := by
  have c120516 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.1
  have c120517 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.1
  have c120518 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.1
  have c120519 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.1
  have c120520 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.1
  have c120521 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_8
  simp only [← c120516, ← c120517, ← c120518] at e_1975_8
  have e_1975_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_9
  simp only [← c120519, ← c120520, k2, ← c120521] at e_1975_9
  have hr := e_1975_9
  simp only [e_1975_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12992)) := by
  have c120522 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120523 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120524 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120525 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120526 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120527 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_10
  simp only [← c120522, ← c120523, ← c120524] at e_1975_10
  have e_1975_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_11
  simp only [← c120525, ← c120526, k2, ← c120527] at e_1975_11
  have hr := e_1975_11
  simp only [e_1975_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12993)) := by
  have c120528 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120529 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120530 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120531 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120532 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120533 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_12
  simp only [← c120528, ← c120529, ← c120530] at e_1975_12
  have e_1975_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_13
  simp only [← c120531, ← c120532, k2, ← c120533] at e_1975_13
  have hr := e_1975_13
  simp only [e_1975_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 63) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12994)) := by
  have c120534 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120535 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120536 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120537 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120538 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120539 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_14
  simp only [← c120534, ← c120535, ← c120536] at e_1975_14
  have e_1975_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_15
  simp only [← c120537, ← c120538, k2, ← c120539] at e_1975_15
  have hr := e_1975_15
  simp only [e_1975_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 71) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12995)) := by
  have c120540 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120541 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120542 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120543 := (publicBatchWrapper89_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120544 := (publicBatchWrapper89_copies3767 a h).1
  have c120545 := (publicBatchWrapper89_copies3767 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_16
  simp only [← c120540, ← c120541, ← c120542] at e_1975_16
  have e_1975_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_17
  simp only [← c120543, ← c120544, k2, ← c120545] at e_1975_17
  have hr := e_1975_17
  simp only [e_1975_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1975 79) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12996)) := by
  have c120546 := (publicBatchWrapper89_copies3767 a h).2.2.1
  have c120547 := (publicBatchWrapper89_copies3767 a h).2.2.2.1
  have c120548 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.1
  have c120549 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.1
  have c120550 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.1
  have c120551 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1975_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_18
  simp only [← c120546, ← c120547, ← c120548] at e_1975_18
  have e_1975_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_19
  simp only [← c120549, ← c120550, k2, ← c120551] at e_1975_19
  have hr := e_1975_19
  simp only [e_1975_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 7) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12997)) := by
  have c120552 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.1
  have c120553 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.1
  have c120554 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120555 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120556 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120557 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_0
  simp only [← c120552, ← c120553, ← c120554] at e_1976_0
  have e_1976_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_1
  simp only [← c120555, ← c120556, k2, ← c120557] at e_1976_1
  have hr := e_1976_1
  simp only [e_1976_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 15) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12998)) := by
  have c120558 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120559 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120560 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120561 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120562 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120563 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_2
  simp only [← c120558, ← c120559, ← c120560] at e_1976_2
  have e_1976_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_3
  simp only [← c120561, ← c120562, k2, ← c120563] at e_1976_3
  have hr := e_1976_3
  simp only [e_1976_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 23) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12999)) := by
  have c120564 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120565 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120566 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120567 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120568 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120569 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_4
  simp only [← c120564, ← c120565, ← c120566] at e_1976_4
  have e_1976_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_5
  simp only [← c120567, ← c120568, k2, ← c120569] at e_1976_5
  have hr := e_1976_5
  simp only [e_1976_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 13000)) := by
  have c120570 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120571 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120572 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120573 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120574 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120575 := (publicBatchWrapper89_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_6
  simp only [← c120570, ← c120571, ← c120572] at e_1976_6
  have e_1976_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_7
  simp only [← c120573, ← c120574, k2, ← c120575] at e_1976_7
  have hr := e_1976_7
  simp only [e_1976_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 13001)) := by
  have c120576 := (publicBatchWrapper89_copies3768 a h).1
  have c120577 := (publicBatchWrapper89_copies3768 a h).2.1
  have c120578 := (publicBatchWrapper89_copies3768 a h).2.2.1
  have c120579 := (publicBatchWrapper89_copies3768 a h).2.2.2.1
  have c120580 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.1
  have c120581 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_8
  simp only [← c120576, ← c120577, ← c120578] at e_1976_8
  have e_1976_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_9
  simp only [← c120579, ← c120580, k2, ← c120581] at e_1976_9
  have hr := e_1976_9
  simp only [e_1976_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13170)) := by
  have c120582 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.1
  have c120583 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.1
  have c120584 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.1
  have c120585 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.1
  have c120586 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120587 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_10
  simp only [← c120582, ← c120583, ← c120584] at e_1976_10
  have e_1976_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_11
  simp only [← c120585, ← c120586, k2, ← c120587] at e_1976_11
  have hr := e_1976_11
  simp only [e_1976_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13171)) := by
  have c120588 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120589 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120590 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120591 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120592 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120593 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_12
  simp only [← c120588, ← c120589, ← c120590] at e_1976_12
  have e_1976_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_13
  simp only [← c120591, ← c120592, k2, ← c120593] at e_1976_13
  have hr := e_1976_13
  simp only [e_1976_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13172)) := by
  have c120594 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120595 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120596 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120597 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120598 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120599 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_14
  simp only [← c120594, ← c120595, ← c120596] at e_1976_14
  have e_1976_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_15
  simp only [← c120597, ← c120598, k2, ← c120599] at e_1976_15
  have hr := e_1976_15
  simp only [e_1976_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13173)) := by
  have c120600 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120601 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120602 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120603 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120604 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120605 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_16
  simp only [← c120600, ← c120601, ← c120602] at e_1976_16
  have e_1976_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_17
  simp only [← c120603, ← c120604, k2, ← c120605] at e_1976_17
  have hr := e_1976_17
  simp only [e_1976_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1976 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13174)) := by
  have c120606 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120607 := (publicBatchWrapper89_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120608 := (publicBatchWrapper89_copies3769 a h).1
  have c120609 := (publicBatchWrapper89_copies3769 a h).2.1
  have c120610 := (publicBatchWrapper89_copies3769 a h).2.2.1
  have c120611 := (publicBatchWrapper89_copies3769 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1976_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_18
  simp only [← c120606, ← c120607, ← c120608] at e_1976_18
  have e_1976_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_19
  simp only [← c120609, ← c120610, k2, ← c120611] at e_1976_19
  have hr := e_1976_19
  simp only [e_1976_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13175)) := by
  have c120612 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.1
  have c120613 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.1
  have c120614 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.1
  have c120615 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.1
  have c120616 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.1
  have c120617 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_0
  simp only [← c120612, ← c120613, ← c120614] at e_1977_0
  have e_1977_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_1
  simp only [← c120615, ← c120616, k2, ← c120617] at e_1977_1
  have hr := e_1977_1
  simp only [e_1977_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13176)) := by
  have c120618 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120619 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120620 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120621 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120622 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120623 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_2
  simp only [← c120618, ← c120619, ← c120620] at e_1977_2
  have e_1977_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_3
  simp only [← c120621, ← c120622, k2, ← c120623] at e_1977_3
  have hr := e_1977_3
  simp only [e_1977_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13177)) := by
  have c120624 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120625 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120626 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120627 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120628 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120629 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_4
  simp only [← c120624, ← c120625, ← c120626] at e_1977_4
  have e_1977_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_5
  simp only [← c120627, ← c120628, k2, ← c120629] at e_1977_5
  have hr := e_1977_5
  simp only [e_1977_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13178)) := by
  have c120630 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120631 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120632 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120633 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120634 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120635 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_6
  simp only [← c120630, ← c120631, ← c120632] at e_1977_6
  have e_1977_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_7
  simp only [← c120633, ← c120634, k2, ← c120635] at e_1977_7
  have hr := e_1977_7
  simp only [e_1977_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13179)) := by
  have c120636 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120637 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120638 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120639 := (publicBatchWrapper89_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120640 := (publicBatchWrapper89_copies3770 a h).1
  have c120641 := (publicBatchWrapper89_copies3770 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_8
  simp only [← c120636, ← c120637, ← c120638] at e_1977_8
  have e_1977_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_9
  simp only [← c120639, ← c120640, k2, ← c120641] at e_1977_9
  have hr := e_1977_9
  simp only [e_1977_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13180)) := by
  have c120642 := (publicBatchWrapper89_copies3770 a h).2.2.1
  have c120643 := (publicBatchWrapper89_copies3770 a h).2.2.2.1
  have c120644 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.1
  have c120645 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.1
  have c120646 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.1
  have c120647 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_10
  simp only [← c120642, ← c120643, ← c120644] at e_1977_10
  have e_1977_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_11
  simp only [← c120645, ← c120646, k2, ← c120647] at e_1977_11
  have hr := e_1977_11
  simp only [e_1977_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13181)) := by
  have c120648 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.1
  have c120649 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.1
  have c120650 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120651 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120652 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120653 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_12
  simp only [← c120648, ← c120649, ← c120650] at e_1977_12
  have e_1977_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_13
  simp only [← c120651, ← c120652, k2, ← c120653] at e_1977_13
  have hr := e_1977_13
  simp only [e_1977_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13182)) := by
  have c120654 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120655 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120656 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120657 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120658 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120659 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_14
  simp only [← c120654, ← c120655, ← c120656] at e_1977_14
  have e_1977_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_15
  simp only [← c120657, ← c120658, k2, ← c120659] at e_1977_15
  have hr := e_1977_15
  simp only [e_1977_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13183)) := by
  have c120660 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120661 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120662 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120663 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120664 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120665 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_16
  simp only [← c120660, ← c120661, ← c120662] at e_1977_16
  have e_1977_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_17
  simp only [← c120663, ← c120664, k2, ← c120665] at e_1977_17
  have hr := e_1977_17
  simp only [e_1977_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1977 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13184)) := by
  have c120666 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120667 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120668 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120669 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120670 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120671 := (publicBatchWrapper89_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1977_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_18
  simp only [← c120666, ← c120667, ← c120668] at e_1977_18
  have e_1977_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_19
  simp only [← c120669, ← c120670, k2, ← c120671] at e_1977_19
  have hr := e_1977_19
  simp only [e_1977_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13185)) := by
  have c120672 := (publicBatchWrapper89_copies3771 a h).1
  have c120673 := (publicBatchWrapper89_copies3771 a h).2.1
  have c120674 := (publicBatchWrapper89_copies3771 a h).2.2.1
  have c120675 := (publicBatchWrapper89_copies3771 a h).2.2.2.1
  have c120676 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.1
  have c120677 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_0
  simp only [← c120672, ← c120673, ← c120674] at e_1978_0
  have e_1978_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_1
  simp only [← c120675, ← c120676, k2, ← c120677] at e_1978_1
  have hr := e_1978_1
  simp only [e_1978_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13186)) := by
  have c120678 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.1
  have c120679 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.1
  have c120680 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.1
  have c120681 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.1
  have c120682 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120683 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_2
  simp only [← c120678, ← c120679, ← c120680] at e_1978_2
  have e_1978_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_3
  simp only [← c120681, ← c120682, k2, ← c120683] at e_1978_3
  have hr := e_1978_3
  simp only [e_1978_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13187)) := by
  have c120684 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120685 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120686 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120687 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120688 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120689 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_4
  simp only [← c120684, ← c120685, ← c120686] at e_1978_4
  have e_1978_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_5
  simp only [← c120687, ← c120688, k2, ← c120689] at e_1978_5
  have hr := e_1978_5
  simp only [e_1978_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13188)) := by
  have c120690 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120691 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120692 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120693 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120694 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120695 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_6
  simp only [← c120690, ← c120691, ← c120692] at e_1978_6
  have e_1978_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_7
  simp only [← c120693, ← c120694, k2, ← c120695] at e_1978_7
  have hr := e_1978_7
  simp only [e_1978_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13189)) := by
  have c120696 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120697 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120698 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120699 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120700 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120701 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_8
  simp only [← c120696, ← c120697, ← c120698] at e_1978_8
  have e_1978_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_9
  simp only [← c120699, ← c120700, k2, ← c120701] at e_1978_9
  have hr := e_1978_9
  simp only [e_1978_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13190)) := by
  have c120702 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120703 := (publicBatchWrapper89_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120704 := (publicBatchWrapper89_copies3772 a h).1
  have c120705 := (publicBatchWrapper89_copies3772 a h).2.1
  have c120706 := (publicBatchWrapper89_copies3772 a h).2.2.1
  have c120707 := (publicBatchWrapper89_copies3772 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_10
  simp only [← c120702, ← c120703, ← c120704] at e_1978_10
  have e_1978_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_11
  simp only [← c120705, ← c120706, k2, ← c120707] at e_1978_11
  have hr := e_1978_11
  simp only [e_1978_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13191)) := by
  have c120708 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.1
  have c120709 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.1
  have c120710 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.1
  have c120711 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.1
  have c120712 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.1
  have c120713 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_12
  simp only [← c120708, ← c120709, ← c120710] at e_1978_12
  have e_1978_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_13
  simp only [← c120711, ← c120712, k2, ← c120713] at e_1978_13
  have hr := e_1978_13
  simp only [e_1978_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 63) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13192)) := by
  have c120714 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120715 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120716 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120717 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120718 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120719 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_14
  simp only [← c120714, ← c120715, ← c120716] at e_1978_14
  have e_1978_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_15
  simp only [← c120717, ← c120718, k2, ← c120719] at e_1978_15
  have hr := e_1978_15
  simp only [e_1978_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 71) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13193)) := by
  have c120720 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120721 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120722 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120723 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120724 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120725 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_16
  simp only [← c120720, ← c120721, ← c120722] at e_1978_16
  have e_1978_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_17
  simp only [← c120723, ← c120724, k2, ← c120725] at e_1978_17
  have hr := e_1978_17
  simp only [e_1978_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1978 79) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13194)) := by
  have c120726 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120727 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120728 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120729 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120730 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120731 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1978_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_18
  simp only [← c120726, ← c120727, ← c120728] at e_1978_18
  have e_1978_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_19
  simp only [← c120729, ← c120730, k2, ← c120731] at e_1978_19
  have hr := e_1978_19
  simp only [e_1978_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 7) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13195)) := by
  have c120732 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120733 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120734 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120735 := (publicBatchWrapper89_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120736 := (publicBatchWrapper89_copies3773 a h).1
  have c120737 := (publicBatchWrapper89_copies3773 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_0
  simp only [← c120732, ← c120733, ← c120734] at e_1979_0
  have e_1979_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_1
  simp only [← c120735, ← c120736, k2, ← c120737] at e_1979_1
  have hr := e_1979_1
  simp only [e_1979_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 15) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13196)) := by
  have c120738 := (publicBatchWrapper89_copies3773 a h).2.2.1
  have c120739 := (publicBatchWrapper89_copies3773 a h).2.2.2.1
  have c120740 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.1
  have c120741 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.1
  have c120742 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.1
  have c120743 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_2
  simp only [← c120738, ← c120739, ← c120740] at e_1979_2
  have e_1979_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_3
  simp only [← c120741, ← c120742, k2, ← c120743] at e_1979_3
  have hr := e_1979_3
  simp only [e_1979_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 23) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13197)) := by
  have c120744 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.1
  have c120745 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.1
  have c120746 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120747 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120748 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120749 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_4
  simp only [← c120744, ← c120745, ← c120746] at e_1979_4
  have e_1979_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_5
  simp only [← c120747, ← c120748, k2, ← c120749] at e_1979_5
  have hr := e_1979_5
  simp only [e_1979_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 31) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13198)) := by
  have c120750 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120751 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120752 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120753 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120754 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120755 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_6
  simp only [← c120750, ← c120751, ← c120752] at e_1979_6
  have e_1979_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_7
  simp only [← c120753, ← c120754, k2, ← c120755] at e_1979_7
  have hr := e_1979_7
  simp only [e_1979_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 39) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13199)) := by
  have c120756 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120757 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120758 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120759 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120760 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120761 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_8
  simp only [← c120756, ← c120757, ← c120758] at e_1979_8
  have e_1979_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_9
  simp only [← c120759, ← c120760, k2, ← c120761] at e_1979_9
  have hr := e_1979_9
  simp only [e_1979_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 47) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13200)) := by
  have c120762 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120763 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120764 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120765 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120766 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120767 := (publicBatchWrapper89_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_10
  simp only [← c120762, ← c120763, ← c120764] at e_1979_10
  have e_1979_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_11
  simp only [← c120765, ← c120766, k2, ← c120767] at e_1979_11
  have hr := e_1979_11
  simp only [e_1979_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 55) = bselect (a (.wire 64 39)) (a (.virt 200)) (a (.virt 13201)) := by
  have c120768 := (publicBatchWrapper89_copies3774 a h).1
  have c120769 := (publicBatchWrapper89_copies3774 a h).2.1
  have c120770 := (publicBatchWrapper89_copies3774 a h).2.2.1
  have c120771 := (publicBatchWrapper89_copies3774 a h).2.2.2.1
  have c120772 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.1
  have c120773 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_12
  simp only [← c120768, ← c120769, ← c120770] at e_1979_12
  have e_1979_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_13
  simp only [← c120771, ← c120772, k2, ← c120773] at e_1979_13
  have hr := e_1979_13
  simp only [e_1979_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13370)) := by
  have c120774 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.1
  have c120775 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.1
  have c120776 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.1
  have c120777 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.1
  have c120778 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120779 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_14
  simp only [← c120774, ← c120775, ← c120776] at e_1979_14
  have e_1979_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_15
  simp only [← c120777, ← c120778, k2, ← c120779] at e_1979_15
  have hr := e_1979_15
  simp only [e_1979_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13371)) := by
  have c120780 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120781 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120782 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120783 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120784 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120785 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_16
  simp only [← c120780, ← c120781, ← c120782] at e_1979_16
  have e_1979_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_17
  simp only [← c120783, ← c120784, k2, ← c120785] at e_1979_17
  have hr := e_1979_17
  simp only [e_1979_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1979 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13372)) := by
  have c120786 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120787 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120788 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120789 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120790 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120791 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1979_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_18
  simp only [← c120786, ← c120787, ← c120788] at e_1979_18
  have e_1979_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_19
  simp only [← c120789, ← c120790, k2, ← c120791] at e_1979_19
  have hr := e_1979_19
  simp only [e_1979_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13373)) := by
  have c120792 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120793 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120794 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120795 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120796 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120797 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_0
  simp only [← c120792, ← c120793, ← c120794] at e_1980_0
  have e_1980_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_1
  simp only [← c120795, ← c120796, k2, ← c120797] at e_1980_1
  have hr := e_1980_1
  simp only [e_1980_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13374)) := by
  have c120798 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120799 := (publicBatchWrapper89_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120800 := (publicBatchWrapper89_copies3775 a h).1
  have c120801 := (publicBatchWrapper89_copies3775 a h).2.1
  have c120802 := (publicBatchWrapper89_copies3775 a h).2.2.1
  have c120803 := (publicBatchWrapper89_copies3775 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_2
  simp only [← c120798, ← c120799, ← c120800] at e_1980_2
  have e_1980_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_3
  simp only [← c120801, ← c120802, k2, ← c120803] at e_1980_3
  have hr := e_1980_3
  simp only [e_1980_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13375)) := by
  have c120804 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.1
  have c120805 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.1
  have c120806 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.1
  have c120807 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.1
  have c120808 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.1
  have c120809 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_4
  simp only [← c120804, ← c120805, ← c120806] at e_1980_4
  have e_1980_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_5
  simp only [← c120807, ← c120808, k2, ← c120809] at e_1980_5
  have hr := e_1980_5
  simp only [e_1980_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13376)) := by
  have c120810 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120811 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120812 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120813 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120814 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120815 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_6
  simp only [← c120810, ← c120811, ← c120812] at e_1980_6
  have e_1980_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_7
  simp only [← c120813, ← c120814, k2, ← c120815] at e_1980_7
  have hr := e_1980_7
  simp only [e_1980_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13377)) := by
  have c120816 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120817 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120818 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120819 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120820 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120821 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_8
  simp only [← c120816, ← c120817, ← c120818] at e_1980_8
  have e_1980_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_9
  simp only [← c120819, ← c120820, k2, ← c120821] at e_1980_9
  have hr := e_1980_9
  simp only [e_1980_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13378)) := by
  have c120822 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120823 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120824 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120825 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120826 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120827 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_10
  simp only [← c120822, ← c120823, ← c120824] at e_1980_10
  have e_1980_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_11
  simp only [← c120825, ← c120826, k2, ← c120827] at e_1980_11
  have hr := e_1980_11
  simp only [e_1980_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13379)) := by
  have c120828 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120829 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120830 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120831 := (publicBatchWrapper89_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120832 := (publicBatchWrapper89_copies3776 a h).1
  have c120833 := (publicBatchWrapper89_copies3776 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_12
  simp only [← c120828, ← c120829, ← c120830] at e_1980_12
  have e_1980_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_13
  simp only [← c120831, ← c120832, k2, ← c120833] at e_1980_13
  have hr := e_1980_13
  simp only [e_1980_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13380)) := by
  have c120834 := (publicBatchWrapper89_copies3776 a h).2.2.1
  have c120835 := (publicBatchWrapper89_copies3776 a h).2.2.2.1
  have c120836 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.1
  have c120837 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.1
  have c120838 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.1
  have c120839 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_14
  simp only [← c120834, ← c120835, ← c120836] at e_1980_14
  have e_1980_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_15
  simp only [← c120837, ← c120838, k2, ← c120839] at e_1980_15
  have hr := e_1980_15
  simp only [e_1980_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13381)) := by
  have c120840 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.1
  have c120841 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.1
  have c120842 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120843 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120844 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120845 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_16
  simp only [← c120840, ← c120841, ← c120842] at e_1980_16
  have e_1980_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_17
  simp only [← c120843, ← c120844, k2, ← c120845] at e_1980_17
  have hr := e_1980_17
  simp only [e_1980_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1980 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13382)) := by
  have c120846 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120847 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120848 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120849 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120850 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120851 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1980_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_18
  simp only [← c120846, ← c120847, ← c120848] at e_1980_18
  have e_1980_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_19
  simp only [← c120849, ← c120850, k2, ← c120851] at e_1980_19
  have hr := e_1980_19
  simp only [e_1980_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13383)) := by
  have c120852 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120853 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120854 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120855 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120856 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120857 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_0
  simp only [← c120852, ← c120853, ← c120854] at e_1981_0
  have e_1981_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_1
  simp only [← c120855, ← c120856, k2, ← c120857] at e_1981_1
  have hr := e_1981_1
  simp only [e_1981_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13384)) := by
  have c120858 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120859 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120860 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120861 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120862 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120863 := (publicBatchWrapper89_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_2
  simp only [← c120858, ← c120859, ← c120860] at e_1981_2
  have e_1981_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_3
  simp only [← c120861, ← c120862, k2, ← c120863] at e_1981_3
  have hr := e_1981_3
  simp only [e_1981_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13385)) := by
  have c120864 := (publicBatchWrapper89_copies3777 a h).1
  have c120865 := (publicBatchWrapper89_copies3777 a h).2.1
  have c120866 := (publicBatchWrapper89_copies3777 a h).2.2.1
  have c120867 := (publicBatchWrapper89_copies3777 a h).2.2.2.1
  have c120868 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.1
  have c120869 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_4
  simp only [← c120864, ← c120865, ← c120866] at e_1981_4
  have e_1981_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_5
  simp only [← c120867, ← c120868, k2, ← c120869] at e_1981_5
  have hr := e_1981_5
  simp only [e_1981_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13386)) := by
  have c120870 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.1
  have c120871 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.1
  have c120872 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.1
  have c120873 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.1
  have c120874 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120875 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_6
  simp only [← c120870, ← c120871, ← c120872] at e_1981_6
  have e_1981_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_7
  simp only [← c120873, ← c120874, k2, ← c120875] at e_1981_7
  have hr := e_1981_7
  simp only [e_1981_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13387)) := by
  have c120876 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120877 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120878 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120879 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120880 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120881 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_8
  simp only [← c120876, ← c120877, ← c120878] at e_1981_8
  have e_1981_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_9
  simp only [← c120879, ← c120880, k2, ← c120881] at e_1981_9
  have hr := e_1981_9
  simp only [e_1981_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13388)) := by
  have c120882 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120883 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120884 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120885 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120886 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120887 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_10
  simp only [← c120882, ← c120883, ← c120884] at e_1981_10
  have e_1981_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_11
  simp only [← c120885, ← c120886, k2, ← c120887] at e_1981_11
  have hr := e_1981_11
  simp only [e_1981_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13389)) := by
  have c120888 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120889 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120890 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120891 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120892 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120893 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_12
  simp only [← c120888, ← c120889, ← c120890] at e_1981_12
  have e_1981_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_13
  simp only [← c120891, ← c120892, k2, ← c120893] at e_1981_13
  have hr := e_1981_13
  simp only [e_1981_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13390)) := by
  have c120894 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120895 := (publicBatchWrapper89_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120896 := (publicBatchWrapper89_copies3778 a h).1
  have c120897 := (publicBatchWrapper89_copies3778 a h).2.1
  have c120898 := (publicBatchWrapper89_copies3778 a h).2.2.1
  have c120899 := (publicBatchWrapper89_copies3778 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_14
  simp only [← c120894, ← c120895, ← c120896] at e_1981_14
  have e_1981_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_15
  simp only [← c120897, ← c120898, k2, ← c120899] at e_1981_15
  have hr := e_1981_15
  simp only [e_1981_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13391)) := by
  have c120900 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.1
  have c120901 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.1
  have c120902 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.1
  have c120903 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.1
  have c120904 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.1
  have c120905 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_16
  simp only [← c120900, ← c120901, ← c120902] at e_1981_16
  have e_1981_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_17
  simp only [← c120903, ← c120904, k2, ← c120905] at e_1981_17
  have hr := e_1981_17
  simp only [e_1981_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1981 79) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13392)) := by
  have c120906 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120907 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120908 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120909 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120910 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120911 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1981_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_18
  simp only [← c120906, ← c120907, ← c120908] at e_1981_18
  have e_1981_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_19
  simp only [← c120909, ← c120910, k2, ← c120911] at e_1981_19
  have hr := e_1981_19
  simp only [e_1981_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 7) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13393)) := by
  have c120912 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120913 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120914 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120915 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120916 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120917 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_0
  simp only [← c120912, ← c120913, ← c120914] at e_1982_0
  have e_1982_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_1
  simp only [← c120915, ← c120916, k2, ← c120917] at e_1982_1
  have hr := e_1982_1
  simp only [e_1982_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 15) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13394)) := by
  have c120918 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120919 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120920 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120921 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120922 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120923 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_2
  simp only [← c120918, ← c120919, ← c120920] at e_1982_2
  have e_1982_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_3
  simp only [← c120921, ← c120922, k2, ← c120923] at e_1982_3
  have hr := e_1982_3
  simp only [e_1982_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 23) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13395)) := by
  have c120924 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120925 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120926 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120927 := (publicBatchWrapper89_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120928 := (publicBatchWrapper89_copies3779 a h).1
  have c120929 := (publicBatchWrapper89_copies3779 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_4
  simp only [← c120924, ← c120925, ← c120926] at e_1982_4
  have e_1982_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_5
  simp only [← c120927, ← c120928, k2, ← c120929] at e_1982_5
  have hr := e_1982_5
  simp only [e_1982_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 31) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13396)) := by
  have c120930 := (publicBatchWrapper89_copies3779 a h).2.2.1
  have c120931 := (publicBatchWrapper89_copies3779 a h).2.2.2.1
  have c120932 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.1
  have c120933 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.1
  have c120934 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.1
  have c120935 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_6
  simp only [← c120930, ← c120931, ← c120932] at e_1982_6
  have e_1982_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_7
  simp only [← c120933, ← c120934, k2, ← c120935] at e_1982_7
  have hr := e_1982_7
  simp only [e_1982_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 39) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13397)) := by
  have c120936 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.1
  have c120937 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.1
  have c120938 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120939 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120940 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120941 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_8
  simp only [← c120936, ← c120937, ← c120938] at e_1982_8
  have e_1982_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_9
  simp only [← c120939, ← c120940, k2, ← c120941] at e_1982_9
  have hr := e_1982_9
  simp only [e_1982_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 47) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13398)) := by
  have c120942 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120943 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120944 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120945 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120946 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120947 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_10
  simp only [← c120942, ← c120943, ← c120944] at e_1982_10
  have e_1982_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_11
  simp only [← c120945, ← c120946, k2, ← c120947] at e_1982_11
  have hr := e_1982_11
  simp only [e_1982_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 55) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13399)) := by
  have c120948 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120949 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120950 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120951 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120952 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120953 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_12
  simp only [← c120948, ← c120949, ← c120950] at e_1982_12
  have e_1982_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_13
  simp only [← c120951, ← c120952, k2, ← c120953] at e_1982_13
  have hr := e_1982_13
  simp only [e_1982_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 63) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13400)) := by
  have c120954 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120955 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120956 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120957 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120958 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120959 := (publicBatchWrapper89_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_14
  simp only [← c120954, ← c120955, ← c120956] at e_1982_14
  have e_1982_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_15
  simp only [← c120957, ← c120958, k2, ← c120959] at e_1982_15
  have hr := e_1982_15
  simp only [e_1982_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 71) = bselect (a (.wire 65 3)) (a (.virt 200)) (a (.virt 13401)) := by
  have c120960 := (publicBatchWrapper89_copies3780 a h).1
  have c120961 := (publicBatchWrapper89_copies3780 a h).2.1
  have c120962 := (publicBatchWrapper89_copies3780 a h).2.2.1
  have c120963 := (publicBatchWrapper89_copies3780 a h).2.2.2.1
  have c120964 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.1
  have c120965 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_16
  simp only [← c120960, ← c120961, ← c120962] at e_1982_16
  have e_1982_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_17
  simp only [← c120963, ← c120964, k2, ← c120965] at e_1982_17
  have hr := e_1982_17
  simp only [e_1982_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1982 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13570)) := by
  have c120966 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.1
  have c120967 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.1
  have c120968 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.1
  have c120969 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.1
  have c120970 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120971 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1982_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_18
  simp only [← c120966, ← c120967, ← c120968] at e_1982_18
  have e_1982_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_19
  simp only [← c120969, ← c120970, k2, ← c120971] at e_1982_19
  have hr := e_1982_19
  simp only [e_1982_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13571)) := by
  have c120972 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120973 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120974 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120975 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120976 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120977 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_0
  simp only [← c120972, ← c120973, ← c120974] at e_1983_0
  have e_1983_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_1
  simp only [← c120975, ← c120976, k2, ← c120977] at e_1983_1
  have hr := e_1983_1
  simp only [e_1983_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13572)) := by
  have c120978 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120979 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120980 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120981 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120982 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120983 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_2
  simp only [← c120978, ← c120979, ← c120980] at e_1983_2
  have e_1983_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_3
  simp only [← c120981, ← c120982, k2, ← c120983] at e_1983_3
  have hr := e_1983_3
  simp only [e_1983_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13573)) := by
  have c120984 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120985 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120986 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120987 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120988 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120989 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_4
  simp only [← c120984, ← c120985, ← c120986] at e_1983_4
  have e_1983_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_5
  simp only [← c120987, ← c120988, k2, ← c120989] at e_1983_5
  have hr := e_1983_5
  simp only [e_1983_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13574)) := by
  have c120990 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120991 := (publicBatchWrapper89_copies3780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120992 := (publicBatchWrapper89_copies3781 a h).1
  have c120993 := (publicBatchWrapper89_copies3781 a h).2.1
  have c120994 := (publicBatchWrapper89_copies3781 a h).2.2.1
  have c120995 := (publicBatchWrapper89_copies3781 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_6
  simp only [← c120990, ← c120991, ← c120992] at e_1983_6
  have e_1983_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_7
  simp only [← c120993, ← c120994, k2, ← c120995] at e_1983_7
  have hr := e_1983_7
  simp only [e_1983_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13575)) := by
  have c120996 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.1
  have c120997 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.1
  have c120998 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.1
  have c120999 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.1
  have c121000 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.1
  have c121001 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_8
  simp only [← c120996, ← c120997, ← c120998] at e_1983_8
  have e_1983_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_9
  simp only [← c120999, ← c121000, k2, ← c121001] at e_1983_9
  have hr := e_1983_9
  simp only [e_1983_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13576)) := by
  have c121002 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121003 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121004 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121005 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121006 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121007 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_10
  simp only [← c121002, ← c121003, ← c121004] at e_1983_10
  have e_1983_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_11
  simp only [← c121005, ← c121006, k2, ← c121007] at e_1983_11
  have hr := e_1983_11
  simp only [e_1983_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13577)) := by
  have c121008 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121009 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121010 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121011 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121012 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121013 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_12
  simp only [← c121008, ← c121009, ← c121010] at e_1983_12
  have e_1983_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_13
  simp only [← c121011, ← c121012, k2, ← c121013] at e_1983_13
  have hr := e_1983_13
  simp only [e_1983_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13578)) := by
  have c121014 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121015 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121016 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121017 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121018 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121019 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_14
  simp only [← c121014, ← c121015, ← c121016] at e_1983_14
  have e_1983_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_15
  simp only [← c121017, ← c121018, k2, ← c121019] at e_1983_15
  have hr := e_1983_15
  simp only [e_1983_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13579)) := by
  have c121020 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121021 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121022 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121023 := (publicBatchWrapper89_copies3781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121024 := (publicBatchWrapper89_copies3782 a h).1
  have c121025 := (publicBatchWrapper89_copies3782 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_16
  simp only [← c121020, ← c121021, ← c121022] at e_1983_16
  have e_1983_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_17
  simp only [← c121023, ← c121024, k2, ← c121025] at e_1983_17
  have hr := e_1983_17
  simp only [e_1983_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1983 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13580)) := by
  have c121026 := (publicBatchWrapper89_copies3782 a h).2.2.1
  have c121027 := (publicBatchWrapper89_copies3782 a h).2.2.2.1
  have c121028 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.1
  have c121029 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.1
  have c121030 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.1
  have c121031 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1983_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_18
  simp only [← c121026, ← c121027, ← c121028] at e_1983_18
  have e_1983_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_19
  simp only [← c121029, ← c121030, k2, ← c121031] at e_1983_19
  have hr := e_1983_19
  simp only [e_1983_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13581)) := by
  have c121032 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.1
  have c121033 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.1
  have c121034 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121035 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121036 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121037 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_0
  simp only [← c121032, ← c121033, ← c121034] at e_1984_0
  have e_1984_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_1
  simp only [← c121035, ← c121036, k2, ← c121037] at e_1984_1
  have hr := e_1984_1
  simp only [e_1984_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13582)) := by
  have c121038 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121039 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121040 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121041 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121042 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121043 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_2
  simp only [← c121038, ← c121039, ← c121040] at e_1984_2
  have e_1984_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_3
  simp only [← c121041, ← c121042, k2, ← c121043] at e_1984_3
  have hr := e_1984_3
  simp only [e_1984_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13583)) := by
  have c121044 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121045 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121046 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121047 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121048 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121049 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_4
  simp only [← c121044, ← c121045, ← c121046] at e_1984_4
  have e_1984_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_5
  simp only [← c121047, ← c121048, k2, ← c121049] at e_1984_5
  have hr := e_1984_5
  simp only [e_1984_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13584)) := by
  have c121050 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121051 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121052 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121053 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121054 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121055 := (publicBatchWrapper89_copies3782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_6
  simp only [← c121050, ← c121051, ← c121052] at e_1984_6
  have e_1984_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_7
  simp only [← c121053, ← c121054, k2, ← c121055] at e_1984_7
  have hr := e_1984_7
  simp only [e_1984_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13585)) := by
  have c121056 := (publicBatchWrapper89_copies3783 a h).1
  have c121057 := (publicBatchWrapper89_copies3783 a h).2.1
  have c121058 := (publicBatchWrapper89_copies3783 a h).2.2.1
  have c121059 := (publicBatchWrapper89_copies3783 a h).2.2.2.1
  have c121060 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.1
  have c121061 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_8
  simp only [← c121056, ← c121057, ← c121058] at e_1984_8
  have e_1984_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_9
  simp only [← c121059, ← c121060, k2, ← c121061] at e_1984_9
  have hr := e_1984_9
  simp only [e_1984_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13586)) := by
  have c121062 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.1
  have c121063 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.1
  have c121064 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.1
  have c121065 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.1
  have c121066 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121067 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_10
  simp only [← c121062, ← c121063, ← c121064] at e_1984_10
  have e_1984_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_11
  simp only [← c121065, ← c121066, k2, ← c121067] at e_1984_11
  have hr := e_1984_11
  simp only [e_1984_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13587)) := by
  have c121068 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121069 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121070 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121071 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121072 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121073 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_12
  simp only [← c121068, ← c121069, ← c121070] at e_1984_12
  have e_1984_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_13
  simp only [← c121071, ← c121072, k2, ← c121073] at e_1984_13
  have hr := e_1984_13
  simp only [e_1984_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13588)) := by
  have c121074 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121075 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121076 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121077 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121078 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121079 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_14
  simp only [← c121074, ← c121075, ← c121076] at e_1984_14
  have e_1984_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_15
  simp only [← c121077, ← c121078, k2, ← c121079] at e_1984_15
  have hr := e_1984_15
  simp only [e_1984_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13589)) := by
  have c121080 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121081 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121082 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121083 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121084 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121085 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_16
  simp only [← c121080, ← c121081, ← c121082] at e_1984_16
  have e_1984_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_17
  simp only [← c121083, ← c121084, k2, ← c121085] at e_1984_17
  have hr := e_1984_17
  simp only [e_1984_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1984 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13590)) := by
  have c121086 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121087 := (publicBatchWrapper89_copies3783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121088 := (publicBatchWrapper89_copies3784 a h).1
  have c121089 := (publicBatchWrapper89_copies3784 a h).2.1
  have c121090 := (publicBatchWrapper89_copies3784 a h).2.2.1
  have c121091 := (publicBatchWrapper89_copies3784 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1984_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_18
  simp only [← c121086, ← c121087, ← c121088] at e_1984_18
  have e_1984_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_19
  simp only [← c121089, ← c121090, k2, ← c121091] at e_1984_19
  have hr := e_1984_19
  simp only [e_1984_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13591)) := by
  have c121092 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.1
  have c121093 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.1
  have c121094 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.1
  have c121095 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.1
  have c121096 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.1
  have c121097 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_0
  simp only [← c121092, ← c121093, ← c121094] at e_1985_0
  have e_1985_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_1
  simp only [← c121095, ← c121096, k2, ← c121097] at e_1985_1
  have hr := e_1985_1
  simp only [e_1985_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 15) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13592)) := by
  have c121098 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121099 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121100 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121101 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121102 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121103 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_2
  simp only [← c121098, ← c121099, ← c121100] at e_1985_2
  have e_1985_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_3
  simp only [← c121101, ← c121102, k2, ← c121103] at e_1985_3
  have hr := e_1985_3
  simp only [e_1985_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 23) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13593)) := by
  have c121104 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121105 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121106 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121107 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121108 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121109 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_4
  simp only [← c121104, ← c121105, ← c121106] at e_1985_4
  have e_1985_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_5
  simp only [← c121107, ← c121108, k2, ← c121109] at e_1985_5
  have hr := e_1985_5
  simp only [e_1985_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 31) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13594)) := by
  have c121110 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121111 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121112 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121113 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121114 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121115 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_6
  simp only [← c121110, ← c121111, ← c121112] at e_1985_6
  have e_1985_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_7
  simp only [← c121113, ← c121114, k2, ← c121115] at e_1985_7
  have hr := e_1985_7
  simp only [e_1985_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 39) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13595)) := by
  have c121116 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121117 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121118 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121119 := (publicBatchWrapper89_copies3784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121120 := (publicBatchWrapper89_copies3785 a h).1
  have c121121 := (publicBatchWrapper89_copies3785 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_8
  simp only [← c121116, ← c121117, ← c121118] at e_1985_8
  have e_1985_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_9
  simp only [← c121119, ← c121120, k2, ← c121121] at e_1985_9
  have hr := e_1985_9
  simp only [e_1985_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 47) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13596)) := by
  have c121122 := (publicBatchWrapper89_copies3785 a h).2.2.1
  have c121123 := (publicBatchWrapper89_copies3785 a h).2.2.2.1
  have c121124 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.1
  have c121125 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.1
  have c121126 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.1
  have c121127 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_10
  simp only [← c121122, ← c121123, ← c121124] at e_1985_10
  have e_1985_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_11
  simp only [← c121125, ← c121126, k2, ← c121127] at e_1985_11
  have hr := e_1985_11
  simp only [e_1985_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 55) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13597)) := by
  have c121128 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.1
  have c121129 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.1
  have c121130 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121131 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121132 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121133 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_12
  simp only [← c121128, ← c121129, ← c121130] at e_1985_12
  have e_1985_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_13
  simp only [← c121131, ← c121132, k2, ← c121133] at e_1985_13
  have hr := e_1985_13
  simp only [e_1985_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 63) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13598)) := by
  have c121134 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121135 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121136 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121137 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121138 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121139 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_14
  simp only [← c121134, ← c121135, ← c121136] at e_1985_14
  have e_1985_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_15
  simp only [← c121137, ← c121138, k2, ← c121139] at e_1985_15
  have hr := e_1985_15
  simp only [e_1985_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 71) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13599)) := by
  have c121140 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121141 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121142 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121143 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121144 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121145 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_16
  simp only [← c121140, ← c121141, ← c121142] at e_1985_16
  have e_1985_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_17
  simp only [← c121143, ← c121144, k2, ← c121145] at e_1985_17
  have hr := e_1985_17
  simp only [e_1985_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1985 79) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13600)) := by
  have c121146 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121147 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121148 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121149 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121150 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121151 := (publicBatchWrapper89_copies3785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1985_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_18
  simp only [← c121146, ← c121147, ← c121148] at e_1985_18
  have e_1985_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1985 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1985_19
  simp only [← c121149, ← c121150, k2, ← c121151] at e_1985_19
  have hr := e_1985_19
  simp only [e_1985_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 7) = bselect (a (.wire 65 47)) (a (.virt 200)) (a (.virt 13601)) := by
  have c121152 := (publicBatchWrapper89_copies3786 a h).1
  have c121153 := (publicBatchWrapper89_copies3786 a h).2.1
  have c121154 := (publicBatchWrapper89_copies3786 a h).2.2.1
  have c121155 := (publicBatchWrapper89_copies3786 a h).2.2.2.1
  have c121156 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.1
  have c121157 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_0
  simp only [← c121152, ← c121153, ← c121154] at e_1986_0
  have e_1986_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_1
  simp only [← c121155, ← c121156, k2, ← c121157] at e_1986_1
  have hr := e_1986_1
  simp only [e_1986_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13770)) := by
  have c121158 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.1
  have c121159 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.1
  have c121160 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.1
  have c121161 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.1
  have c121162 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121163 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_2
  simp only [← c121158, ← c121159, ← c121160] at e_1986_2
  have e_1986_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_3
  simp only [← c121161, ← c121162, k2, ← c121163] at e_1986_3
  have hr := e_1986_3
  simp only [e_1986_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13771)) := by
  have c121164 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121165 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121166 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121167 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121168 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121169 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_4
  simp only [← c121164, ← c121165, ← c121166] at e_1986_4
  have e_1986_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_5
  simp only [← c121167, ← c121168, k2, ← c121169] at e_1986_5
  have hr := e_1986_5
  simp only [e_1986_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13772)) := by
  have c121170 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121171 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121172 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121173 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121174 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121175 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_6
  simp only [← c121170, ← c121171, ← c121172] at e_1986_6
  have e_1986_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_7
  simp only [← c121173, ← c121174, k2, ← c121175] at e_1986_7
  have hr := e_1986_7
  simp only [e_1986_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13773)) := by
  have c121176 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121177 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121178 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121179 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121180 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121181 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_8
  simp only [← c121176, ← c121177, ← c121178] at e_1986_8
  have e_1986_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_9
  simp only [← c121179, ← c121180, k2, ← c121181] at e_1986_9
  have hr := e_1986_9
  simp only [e_1986_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13774)) := by
  have c121182 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121183 := (publicBatchWrapper89_copies3786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121184 := (publicBatchWrapper89_copies3787 a h).1
  have c121185 := (publicBatchWrapper89_copies3787 a h).2.1
  have c121186 := (publicBatchWrapper89_copies3787 a h).2.2.1
  have c121187 := (publicBatchWrapper89_copies3787 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_10
  simp only [← c121182, ← c121183, ← c121184] at e_1986_10
  have e_1986_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_11
  simp only [← c121185, ← c121186, k2, ← c121187] at e_1986_11
  have hr := e_1986_11
  simp only [e_1986_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13775)) := by
  have c121188 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.1
  have c121189 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.1
  have c121190 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.1
  have c121191 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.1
  have c121192 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.1
  have c121193 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_12
  simp only [← c121188, ← c121189, ← c121190] at e_1986_12
  have e_1986_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_13
  simp only [← c121191, ← c121192, k2, ← c121193] at e_1986_13
  have hr := e_1986_13
  simp only [e_1986_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13776)) := by
  have c121194 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121195 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121196 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121197 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121198 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121199 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_14
  simp only [← c121194, ← c121195, ← c121196] at e_1986_14
  have e_1986_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_15
  simp only [← c121197, ← c121198, k2, ← c121199] at e_1986_15
  have hr := e_1986_15
  simp only [e_1986_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13777)) := by
  have c121200 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121201 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121202 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121203 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121204 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121205 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_16
  simp only [← c121200, ← c121201, ← c121202] at e_1986_16
  have e_1986_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_17
  simp only [← c121203, ← c121204, k2, ← c121205] at e_1986_17
  have hr := e_1986_17
  simp only [e_1986_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1986 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13778)) := by
  have c121206 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121207 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121208 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121209 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121210 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121211 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1986_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_18
  simp only [← c121206, ← c121207, ← c121208] at e_1986_18
  have e_1986_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1986 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1986_19
  simp only [← c121209, ← c121210, k2, ← c121211] at e_1986_19
  have hr := e_1986_19
  simp only [e_1986_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13779)) := by
  have c121212 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121213 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121214 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121215 := (publicBatchWrapper89_copies3787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121216 := (publicBatchWrapper89_copies3788 a h).1
  have c121217 := (publicBatchWrapper89_copies3788 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_0
  simp only [← c121212, ← c121213, ← c121214] at e_1987_0
  have e_1987_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_1
  simp only [← c121215, ← c121216, k2, ← c121217] at e_1987_1
  have hr := e_1987_1
  simp only [e_1987_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13780)) := by
  have c121218 := (publicBatchWrapper89_copies3788 a h).2.2.1
  have c121219 := (publicBatchWrapper89_copies3788 a h).2.2.2.1
  have c121220 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.1
  have c121221 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.1
  have c121222 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.1
  have c121223 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_2
  simp only [← c121218, ← c121219, ← c121220] at e_1987_2
  have e_1987_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_3
  simp only [← c121221, ← c121222, k2, ← c121223] at e_1987_3
  have hr := e_1987_3
  simp only [e_1987_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13781)) := by
  have c121224 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.1
  have c121225 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.1
  have c121226 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121227 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121228 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121229 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_4
  simp only [← c121224, ← c121225, ← c121226] at e_1987_4
  have e_1987_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_5
  simp only [← c121227, ← c121228, k2, ← c121229] at e_1987_5
  have hr := e_1987_5
  simp only [e_1987_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13782)) := by
  have c121230 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121231 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121232 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121233 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121234 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121235 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_6
  simp only [← c121230, ← c121231, ← c121232] at e_1987_6
  have e_1987_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_7
  simp only [← c121233, ← c121234, k2, ← c121235] at e_1987_7
  have hr := e_1987_7
  simp only [e_1987_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13783)) := by
  have c121236 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121237 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121238 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121239 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121240 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121241 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_8
  simp only [← c121236, ← c121237, ← c121238] at e_1987_8
  have e_1987_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_9
  simp only [← c121239, ← c121240, k2, ← c121241] at e_1987_9
  have hr := e_1987_9
  simp only [e_1987_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13784)) := by
  have c121242 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121243 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121244 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121245 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121246 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121247 := (publicBatchWrapper89_copies3788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_10
  simp only [← c121242, ← c121243, ← c121244] at e_1987_10
  have e_1987_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_11
  simp only [← c121245, ← c121246, k2, ← c121247] at e_1987_11
  have hr := e_1987_11
  simp only [e_1987_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13785)) := by
  have c121248 := (publicBatchWrapper89_copies3789 a h).1
  have c121249 := (publicBatchWrapper89_copies3789 a h).2.1
  have c121250 := (publicBatchWrapper89_copies3789 a h).2.2.1
  have c121251 := (publicBatchWrapper89_copies3789 a h).2.2.2.1
  have c121252 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.1
  have c121253 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_12
  simp only [← c121248, ← c121249, ← c121250] at e_1987_12
  have e_1987_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_13
  simp only [← c121251, ← c121252, k2, ← c121253] at e_1987_13
  have hr := e_1987_13
  simp only [e_1987_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13786)) := by
  have c121254 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.1
  have c121255 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.1
  have c121256 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.1
  have c121257 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.1
  have c121258 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121259 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_14
  simp only [← c121254, ← c121255, ← c121256] at e_1987_14
  have e_1987_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_15
  simp only [← c121257, ← c121258, k2, ← c121259] at e_1987_15
  have hr := e_1987_15
  simp only [e_1987_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13787)) := by
  have c121260 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121261 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121262 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121263 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121264 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121265 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_16
  simp only [← c121260, ← c121261, ← c121262] at e_1987_16
  have e_1987_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_17
  simp only [← c121263, ← c121264, k2, ← c121265] at e_1987_17
  have hr := e_1987_17
  simp only [e_1987_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1987 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13788)) := by
  have c121266 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121267 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121268 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121269 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121270 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121271 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1987_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_18
  simp only [← c121266, ← c121267, ← c121268] at e_1987_18
  have e_1987_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1987 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1987_19
  simp only [← c121269, ← c121270, k2, ← c121271] at e_1987_19
  have hr := e_1987_19
  simp only [e_1987_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13789)) := by
  have c121272 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121273 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121274 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121275 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121276 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121277 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_0
  simp only [← c121272, ← c121273, ← c121274] at e_1988_0
  have e_1988_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_1
  simp only [← c121275, ← c121276, k2, ← c121277] at e_1988_1
  have hr := e_1988_1
  simp only [e_1988_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13790)) := by
  have c121278 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121279 := (publicBatchWrapper89_copies3789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121280 := (publicBatchWrapper89_copies3790 a h).1
  have c121281 := (publicBatchWrapper89_copies3790 a h).2.1
  have c121282 := (publicBatchWrapper89_copies3790 a h).2.2.1
  have c121283 := (publicBatchWrapper89_copies3790 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_2
  simp only [← c121278, ← c121279, ← c121280] at e_1988_2
  have e_1988_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_3
  simp only [← c121281, ← c121282, k2, ← c121283] at e_1988_3
  have hr := e_1988_3
  simp only [e_1988_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13791)) := by
  have c121284 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.1
  have c121285 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.1
  have c121286 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.1
  have c121287 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.1
  have c121288 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.1
  have c121289 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_4
  simp only [← c121284, ← c121285, ← c121286] at e_1988_4
  have e_1988_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_5
  simp only [← c121287, ← c121288, k2, ← c121289] at e_1988_5
  have hr := e_1988_5
  simp only [e_1988_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 31) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13792)) := by
  have c121290 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121291 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121292 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121293 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121294 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121295 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_6
  simp only [← c121290, ← c121291, ← c121292] at e_1988_6
  have e_1988_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_7
  simp only [← c121293, ← c121294, k2, ← c121295] at e_1988_7
  have hr := e_1988_7
  simp only [e_1988_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 39) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13793)) := by
  have c121296 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121297 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121298 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121299 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121300 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121301 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_8
  simp only [← c121296, ← c121297, ← c121298] at e_1988_8
  have e_1988_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_9
  simp only [← c121299, ← c121300, k2, ← c121301] at e_1988_9
  have hr := e_1988_9
  simp only [e_1988_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 47) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13794)) := by
  have c121302 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121303 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121304 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121305 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121306 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121307 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_10
  simp only [← c121302, ← c121303, ← c121304] at e_1988_10
  have e_1988_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_11
  simp only [← c121305, ← c121306, k2, ← c121307] at e_1988_11
  have hr := e_1988_11
  simp only [e_1988_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 55) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13795)) := by
  have c121308 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121309 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121310 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121311 := (publicBatchWrapper89_copies3790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121312 := (publicBatchWrapper89_copies3791 a h).1
  have c121313 := (publicBatchWrapper89_copies3791 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_12
  simp only [← c121308, ← c121309, ← c121310] at e_1988_12
  have e_1988_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_13
  simp only [← c121311, ← c121312, k2, ← c121313] at e_1988_13
  have hr := e_1988_13
  simp only [e_1988_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 63) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13796)) := by
  have c121314 := (publicBatchWrapper89_copies3791 a h).2.2.1
  have c121315 := (publicBatchWrapper89_copies3791 a h).2.2.2.1
  have c121316 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.1
  have c121317 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.1
  have c121318 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.1
  have c121319 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_14
  simp only [← c121314, ← c121315, ← c121316] at e_1988_14
  have e_1988_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_15
  simp only [← c121317, ← c121318, k2, ← c121319] at e_1988_15
  have hr := e_1988_15
  simp only [e_1988_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 71) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13797)) := by
  have c121320 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.1
  have c121321 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.1
  have c121322 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121323 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121324 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121325 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_16
  simp only [← c121320, ← c121321, ← c121322] at e_1988_16
  have e_1988_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_17
  simp only [← c121323, ← c121324, k2, ← c121325] at e_1988_17
  have hr := e_1988_17
  simp only [e_1988_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1988 79) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13798)) := by
  have c121326 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121327 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121328 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121329 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121330 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121331 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1988_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_18
  simp only [← c121326, ← c121327, ← c121328] at e_1988_18
  have e_1988_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1988 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1988_19
  simp only [← c121329, ← c121330, k2, ← c121331] at e_1988_19
  have hr := e_1988_19
  simp only [e_1988_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 7) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13799)) := by
  have c121332 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121333 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121334 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121335 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121336 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121337 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_0
  simp only [← c121332, ← c121333, ← c121334] at e_1989_0
  have e_1989_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_1
  simp only [← c121335, ← c121336, k2, ← c121337] at e_1989_1
  have hr := e_1989_1
  simp only [e_1989_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 15) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13800)) := by
  have c121338 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121339 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121340 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121341 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121342 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121343 := (publicBatchWrapper89_copies3791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_2
  simp only [← c121338, ← c121339, ← c121340] at e_1989_2
  have e_1989_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_3
  simp only [← c121341, ← c121342, k2, ← c121343] at e_1989_3
  have hr := e_1989_3
  simp only [e_1989_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 23) = bselect (a (.wire 67 11)) (a (.virt 200)) (a (.virt 13801)) := by
  have c121344 := (publicBatchWrapper89_copies3792 a h).1
  have c121345 := (publicBatchWrapper89_copies3792 a h).2.1
  have c121346 := (publicBatchWrapper89_copies3792 a h).2.2.1
  have c121347 := (publicBatchWrapper89_copies3792 a h).2.2.2.1
  have c121348 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.1
  have c121349 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_4
  simp only [← c121344, ← c121345, ← c121346] at e_1989_4
  have e_1989_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_5
  simp only [← c121347, ← c121348, k2, ← c121349] at e_1989_5
  have hr := e_1989_5
  simp only [e_1989_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13970)) := by
  have c121350 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.1
  have c121351 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.1
  have c121352 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.1
  have c121353 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.1
  have c121354 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121355 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_6
  simp only [← c121350, ← c121351, ← c121352] at e_1989_6
  have e_1989_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_7
  simp only [← c121353, ← c121354, k2, ← c121355] at e_1989_7
  have hr := e_1989_7
  simp only [e_1989_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13971)) := by
  have c121356 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121357 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121358 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121359 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121360 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121361 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_8
  simp only [← c121356, ← c121357, ← c121358] at e_1989_8
  have e_1989_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_9
  simp only [← c121359, ← c121360, k2, ← c121361] at e_1989_9
  have hr := e_1989_9
  simp only [e_1989_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13972)) := by
  have c121362 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121363 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121364 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121365 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121366 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121367 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_10
  simp only [← c121362, ← c121363, ← c121364] at e_1989_10
  have e_1989_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_11
  simp only [← c121365, ← c121366, k2, ← c121367] at e_1989_11
  have hr := e_1989_11
  simp only [e_1989_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13973)) := by
  have c121368 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121369 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121370 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121371 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121372 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121373 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_12
  simp only [← c121368, ← c121369, ← c121370] at e_1989_12
  have e_1989_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_13
  simp only [← c121371, ← c121372, k2, ← c121373] at e_1989_13
  have hr := e_1989_13
  simp only [e_1989_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13974)) := by
  have c121374 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121375 := (publicBatchWrapper89_copies3792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121376 := (publicBatchWrapper89_copies3793 a h).1
  have c121377 := (publicBatchWrapper89_copies3793 a h).2.1
  have c121378 := (publicBatchWrapper89_copies3793 a h).2.2.1
  have c121379 := (publicBatchWrapper89_copies3793 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_14
  simp only [← c121374, ← c121375, ← c121376] at e_1989_14
  have e_1989_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_15
  simp only [← c121377, ← c121378, k2, ← c121379] at e_1989_15
  have hr := e_1989_15
  simp only [e_1989_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13975)) := by
  have c121380 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.1
  have c121381 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.1
  have c121382 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.1
  have c121383 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.1
  have c121384 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.1
  have c121385 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_16
  simp only [← c121380, ← c121381, ← c121382] at e_1989_16
  have e_1989_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_17
  simp only [← c121383, ← c121384, k2, ← c121385] at e_1989_17
  have hr := e_1989_17
  simp only [e_1989_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1989 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13976)) := by
  have c121386 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121387 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121388 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121389 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121390 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121391 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1989_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_18
  simp only [← c121386, ← c121387, ← c121388] at e_1989_18
  have e_1989_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1989 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1989_19
  simp only [← c121389, ← c121390, k2, ← c121391] at e_1989_19
  have hr := e_1989_19
  simp only [e_1989_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13977)) := by
  have c121392 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121393 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121394 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121395 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121396 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121397 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_0
  simp only [← c121392, ← c121393, ← c121394] at e_1990_0
  have e_1990_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_1
  simp only [← c121395, ← c121396, k2, ← c121397] at e_1990_1
  have hr := e_1990_1
  simp only [e_1990_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13978)) := by
  have c121398 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121399 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121400 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121401 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121402 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121403 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_2
  simp only [← c121398, ← c121399, ← c121400] at e_1990_2
  have e_1990_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_3
  simp only [← c121401, ← c121402, k2, ← c121403] at e_1990_3
  have hr := e_1990_3
  simp only [e_1990_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13979)) := by
  have c121404 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121405 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121406 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121407 := (publicBatchWrapper89_copies3793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121408 := (publicBatchWrapper89_copies3794 a h).1
  have c121409 := (publicBatchWrapper89_copies3794 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_4
  simp only [← c121404, ← c121405, ← c121406] at e_1990_4
  have e_1990_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_5
  simp only [← c121407, ← c121408, k2, ← c121409] at e_1990_5
  have hr := e_1990_5
  simp only [e_1990_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13980)) := by
  have c121410 := (publicBatchWrapper89_copies3794 a h).2.2.1
  have c121411 := (publicBatchWrapper89_copies3794 a h).2.2.2.1
  have c121412 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.1
  have c121413 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.1
  have c121414 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.1
  have c121415 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_6
  simp only [← c121410, ← c121411, ← c121412] at e_1990_6
  have e_1990_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_7
  simp only [← c121413, ← c121414, k2, ← c121415] at e_1990_7
  have hr := e_1990_7
  simp only [e_1990_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13981)) := by
  have c121416 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.1
  have c121417 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.1
  have c121418 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121419 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121420 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121421 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_8
  simp only [← c121416, ← c121417, ← c121418] at e_1990_8
  have e_1990_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_9
  simp only [← c121419, ← c121420, k2, ← c121421] at e_1990_9
  have hr := e_1990_9
  simp only [e_1990_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13982)) := by
  have c121422 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121423 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121424 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121425 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121426 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121427 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_10
  simp only [← c121422, ← c121423, ← c121424] at e_1990_10
  have e_1990_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_11
  simp only [← c121425, ← c121426, k2, ← c121427] at e_1990_11
  have hr := e_1990_11
  simp only [e_1990_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13983)) := by
  have c121428 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121429 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121430 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121431 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121432 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121433 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_12
  simp only [← c121428, ← c121429, ← c121430] at e_1990_12
  have e_1990_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_13
  simp only [← c121431, ← c121432, k2, ← c121433] at e_1990_13
  have hr := e_1990_13
  simp only [e_1990_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13984)) := by
  have c121434 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121435 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121436 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121437 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121438 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121439 := (publicBatchWrapper89_copies3794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_14
  simp only [← c121434, ← c121435, ← c121436] at e_1990_14
  have e_1990_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_15
  simp only [← c121437, ← c121438, k2, ← c121439] at e_1990_15
  have hr := e_1990_15
  simp only [e_1990_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13985)) := by
  have c121440 := (publicBatchWrapper89_copies3795 a h).1
  have c121441 := (publicBatchWrapper89_copies3795 a h).2.1
  have c121442 := (publicBatchWrapper89_copies3795 a h).2.2.1
  have c121443 := (publicBatchWrapper89_copies3795 a h).2.2.2.1
  have c121444 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.1
  have c121445 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_16
  simp only [← c121440, ← c121441, ← c121442] at e_1990_16
  have e_1990_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_17
  simp only [← c121443, ← c121444, k2, ← c121445] at e_1990_17
  have hr := e_1990_17
  simp only [e_1990_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1990 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13986)) := by
  have c121446 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.1
  have c121447 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.1
  have c121448 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.1
  have c121449 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.1
  have c121450 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121451 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1990_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_18
  simp only [← c121446, ← c121447, ← c121448] at e_1990_18
  have e_1990_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1990 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1990_19
  simp only [← c121449, ← c121450, k2, ← c121451] at e_1990_19
  have hr := e_1990_19
  simp only [e_1990_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13987)) := by
  have c121452 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121453 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121454 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121455 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121456 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121457 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_0
  simp only [← c121452, ← c121453, ← c121454] at e_1991_0
  have e_1991_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_1
  simp only [← c121455, ← c121456, k2, ← c121457] at e_1991_1
  have hr := e_1991_1
  simp only [e_1991_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13988)) := by
  have c121458 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121459 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121460 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121461 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121462 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121463 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_2
  simp only [← c121458, ← c121459, ← c121460] at e_1991_2
  have e_1991_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_3
  simp only [← c121461, ← c121462, k2, ← c121463] at e_1991_3
  have hr := e_1991_3
  simp only [e_1991_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13989)) := by
  have c121464 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121465 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121466 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121467 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121468 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121469 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_4
  simp only [← c121464, ← c121465, ← c121466] at e_1991_4
  have e_1991_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_5
  simp only [← c121467, ← c121468, k2, ← c121469] at e_1991_5
  have hr := e_1991_5
  simp only [e_1991_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13990)) := by
  have c121470 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121471 := (publicBatchWrapper89_copies3795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121472 := (publicBatchWrapper89_copies3796 a h).1
  have c121473 := (publicBatchWrapper89_copies3796 a h).2.1
  have c121474 := (publicBatchWrapper89_copies3796 a h).2.2.1
  have c121475 := (publicBatchWrapper89_copies3796 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_6
  simp only [← c121470, ← c121471, ← c121472] at e_1991_6
  have e_1991_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_7
  simp only [← c121473, ← c121474, k2, ← c121475] at e_1991_7
  have hr := e_1991_7
  simp only [e_1991_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13991)) := by
  have c121476 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.1
  have c121477 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.1
  have c121478 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.1
  have c121479 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.1
  have c121480 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.1
  have c121481 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_8
  simp only [← c121476, ← c121477, ← c121478] at e_1991_8
  have e_1991_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_9
  simp only [← c121479, ← c121480, k2, ← c121481] at e_1991_9
  have hr := e_1991_9
  simp only [e_1991_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 47) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13992)) := by
  have c121482 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121483 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121484 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121485 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121486 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121487 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_10
  simp only [← c121482, ← c121483, ← c121484] at e_1991_10
  have e_1991_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_11
  simp only [← c121485, ← c121486, k2, ← c121487] at e_1991_11
  have hr := e_1991_11
  simp only [e_1991_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 55) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13993)) := by
  have c121488 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121489 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121490 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121491 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121492 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121493 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_12
  simp only [← c121488, ← c121489, ← c121490] at e_1991_12
  have e_1991_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_13
  simp only [← c121491, ← c121492, k2, ← c121493] at e_1991_13
  have hr := e_1991_13
  simp only [e_1991_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 63) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13994)) := by
  have c121494 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121495 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121496 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121497 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121498 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121499 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_14
  simp only [← c121494, ← c121495, ← c121496] at e_1991_14
  have e_1991_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_15
  simp only [← c121497, ← c121498, k2, ← c121499] at e_1991_15
  have hr := e_1991_15
  simp only [e_1991_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 71) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13995)) := by
  have c121500 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121501 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121502 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121503 := (publicBatchWrapper89_copies3796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121504 := (publicBatchWrapper89_copies3797 a h).1
  have c121505 := (publicBatchWrapper89_copies3797 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_16
  simp only [← c121500, ← c121501, ← c121502] at e_1991_16
  have e_1991_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_17
  simp only [← c121503, ← c121504, k2, ← c121505] at e_1991_17
  have hr := e_1991_17
  simp only [e_1991_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1991 79) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13996)) := by
  have c121506 := (publicBatchWrapper89_copies3797 a h).2.2.1
  have c121507 := (publicBatchWrapper89_copies3797 a h).2.2.2.1
  have c121508 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.1
  have c121509 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.1
  have c121510 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.1
  have c121511 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1991_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_18
  simp only [← c121506, ← c121507, ← c121508] at e_1991_18
  have e_1991_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1991 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1991_19
  simp only [← c121509, ← c121510, k2, ← c121511] at e_1991_19
  have hr := e_1991_19
  simp only [e_1991_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 7) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13997)) := by
  have c121512 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.1
  have c121513 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.1
  have c121514 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121515 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121516 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121517 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_0
  simp only [← c121512, ← c121513, ← c121514] at e_1992_0
  have e_1992_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_1
  simp only [← c121515, ← c121516, k2, ← c121517] at e_1992_1
  have hr := e_1992_1
  simp only [e_1992_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 15) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13998)) := by
  have c121518 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121519 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121520 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121521 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121522 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121523 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_2
  simp only [← c121518, ← c121519, ← c121520] at e_1992_2
  have e_1992_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_3
  simp only [← c121521, ← c121522, k2, ← c121523] at e_1992_3
  have hr := e_1992_3
  simp only [e_1992_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 23) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 13999)) := by
  have c121524 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121525 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121526 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121527 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121528 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121529 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_4
  simp only [← c121524, ← c121525, ← c121526] at e_1992_4
  have e_1992_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_5
  simp only [← c121527, ← c121528, k2, ← c121529] at e_1992_5
  have hr := e_1992_5
  simp only [e_1992_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 31) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 14000)) := by
  have c121530 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121531 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121532 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121533 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121534 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121535 := (publicBatchWrapper89_copies3797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_6
  simp only [← c121530, ← c121531, ← c121532] at e_1992_6
  have e_1992_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_7
  simp only [← c121533, ← c121534, k2, ← c121535] at e_1992_7
  have hr := e_1992_7
  simp only [e_1992_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 39) = bselect (a (.wire 67 55)) (a (.virt 200)) (a (.virt 14001)) := by
  have c121536 := (publicBatchWrapper89_copies3798 a h).1
  have c121537 := (publicBatchWrapper89_copies3798 a h).2.1
  have c121538 := (publicBatchWrapper89_copies3798 a h).2.2.1
  have c121539 := (publicBatchWrapper89_copies3798 a h).2.2.2.1
  have c121540 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.1
  have c121541 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_8
  simp only [← c121536, ← c121537, ← c121538] at e_1992_8
  have e_1992_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_9
  simp only [← c121539, ← c121540, k2, ← c121541] at e_1992_9
  have hr := e_1992_9
  simp only [e_1992_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14170)) := by
  have c121542 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.1
  have c121543 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.1
  have c121544 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.1
  have c121545 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.1
  have c121546 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121547 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_10
  simp only [← c121542, ← c121543, ← c121544] at e_1992_10
  have e_1992_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_11
  simp only [← c121545, ← c121546, k2, ← c121547] at e_1992_11
  have hr := e_1992_11
  simp only [e_1992_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14171)) := by
  have c121548 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121549 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121550 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121551 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121552 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121553 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_12
  simp only [← c121548, ← c121549, ← c121550] at e_1992_12
  have e_1992_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_13
  simp only [← c121551, ← c121552, k2, ← c121553] at e_1992_13
  have hr := e_1992_13
  simp only [e_1992_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14172)) := by
  have c121554 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121555 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121556 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121557 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121558 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121559 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_14
  simp only [← c121554, ← c121555, ← c121556] at e_1992_14
  have e_1992_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_15
  simp only [← c121557, ← c121558, k2, ← c121559] at e_1992_15
  have hr := e_1992_15
  simp only [e_1992_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14173)) := by
  have c121560 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121561 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121562 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121563 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121564 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121565 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_16
  simp only [← c121560, ← c121561, ← c121562] at e_1992_16
  have e_1992_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_17
  simp only [← c121563, ← c121564, k2, ← c121565] at e_1992_17
  have hr := e_1992_17
  simp only [e_1992_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1992 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14174)) := by
  have c121566 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121567 := (publicBatchWrapper89_copies3798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121568 := (publicBatchWrapper89_copies3799 a h).1
  have c121569 := (publicBatchWrapper89_copies3799 a h).2.1
  have c121570 := (publicBatchWrapper89_copies3799 a h).2.2.1
  have c121571 := (publicBatchWrapper89_copies3799 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1992_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_18
  simp only [← c121566, ← c121567, ← c121568] at e_1992_18
  have e_1992_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1992 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1992_19
  simp only [← c121569, ← c121570, k2, ← c121571] at e_1992_19
  have hr := e_1992_19
  simp only [e_1992_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14175)) := by
  have c121572 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.1
  have c121573 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.1
  have c121574 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.1
  have c121575 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.1
  have c121576 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.1
  have c121577 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_0
  simp only [← c121572, ← c121573, ← c121574] at e_1993_0
  have e_1993_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_1
  simp only [← c121575, ← c121576, k2, ← c121577] at e_1993_1
  have hr := e_1993_1
  simp only [e_1993_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14176)) := by
  have c121578 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121579 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121580 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121581 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121582 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121583 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_2
  simp only [← c121578, ← c121579, ← c121580] at e_1993_2
  have e_1993_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_3
  simp only [← c121581, ← c121582, k2, ← c121583] at e_1993_3
  have hr := e_1993_3
  simp only [e_1993_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14177)) := by
  have c121584 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121585 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121586 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121587 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121588 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121589 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_4
  simp only [← c121584, ← c121585, ← c121586] at e_1993_4
  have e_1993_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_5
  simp only [← c121587, ← c121588, k2, ← c121589] at e_1993_5
  have hr := e_1993_5
  simp only [e_1993_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14178)) := by
  have c121590 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121591 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121592 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121593 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121594 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121595 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_6
  simp only [← c121590, ← c121591, ← c121592] at e_1993_6
  have e_1993_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_7
  simp only [← c121593, ← c121594, k2, ← c121595] at e_1993_7
  have hr := e_1993_7
  simp only [e_1993_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14179)) := by
  have c121596 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121597 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121598 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121599 := (publicBatchWrapper89_copies3799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121600 := (publicBatchWrapper89_copies3800 a h).1
  have c121601 := (publicBatchWrapper89_copies3800 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_8
  simp only [← c121596, ← c121597, ← c121598] at e_1993_8
  have e_1993_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_9
  simp only [← c121599, ← c121600, k2, ← c121601] at e_1993_9
  have hr := e_1993_9
  simp only [e_1993_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14180)) := by
  have c121602 := (publicBatchWrapper89_copies3800 a h).2.2.1
  have c121603 := (publicBatchWrapper89_copies3800 a h).2.2.2.1
  have c121604 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.1
  have c121605 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.1
  have c121606 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.1
  have c121607 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_10
  simp only [← c121602, ← c121603, ← c121604] at e_1993_10
  have e_1993_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_11
  simp only [← c121605, ← c121606, k2, ← c121607] at e_1993_11
  have hr := e_1993_11
  simp only [e_1993_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14181)) := by
  have c121608 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.1
  have c121609 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.1
  have c121610 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121611 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121612 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121613 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_12
  simp only [← c121608, ← c121609, ← c121610] at e_1993_12
  have e_1993_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_13
  simp only [← c121611, ← c121612, k2, ← c121613] at e_1993_13
  have hr := e_1993_13
  simp only [e_1993_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14182)) := by
  have c121614 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121615 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121616 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121617 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121618 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121619 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_14
  simp only [← c121614, ← c121615, ← c121616] at e_1993_14
  have e_1993_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_15
  simp only [← c121617, ← c121618, k2, ← c121619] at e_1993_15
  have hr := e_1993_15
  simp only [e_1993_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14183)) := by
  have c121620 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121621 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121622 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121623 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121624 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121625 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_16
  simp only [← c121620, ← c121621, ← c121622] at e_1993_16
  have e_1993_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_17
  simp only [← c121623, ← c121624, k2, ← c121625] at e_1993_17
  have hr := e_1993_17
  simp only [e_1993_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1993 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14184)) := by
  have c121626 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121627 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121628 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121629 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121630 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121631 := (publicBatchWrapper89_copies3800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1993_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_18
  simp only [← c121626, ← c121627, ← c121628] at e_1993_18
  have e_1993_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1993 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1993_19
  simp only [← c121629, ← c121630, k2, ← c121631] at e_1993_19
  have hr := e_1993_19
  simp only [e_1993_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14185)) := by
  have c121632 := (publicBatchWrapper89_copies3801 a h).1
  have c121633 := (publicBatchWrapper89_copies3801 a h).2.1
  have c121634 := (publicBatchWrapper89_copies3801 a h).2.2.1
  have c121635 := (publicBatchWrapper89_copies3801 a h).2.2.2.1
  have c121636 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.1
  have c121637 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_0
  simp only [← c121632, ← c121633, ← c121634] at e_1994_0
  have e_1994_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_1
  simp only [← c121635, ← c121636, k2, ← c121637] at e_1994_1
  have hr := e_1994_1
  simp only [e_1994_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14186)) := by
  have c121638 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.1
  have c121639 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.1
  have c121640 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.1
  have c121641 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.1
  have c121642 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121643 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_2
  simp only [← c121638, ← c121639, ← c121640] at e_1994_2
  have e_1994_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_3
  simp only [← c121641, ← c121642, k2, ← c121643] at e_1994_3
  have hr := e_1994_3
  simp only [e_1994_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14187)) := by
  have c121644 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121645 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121646 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121647 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121648 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121649 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_4
  simp only [← c121644, ← c121645, ← c121646] at e_1994_4
  have e_1994_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_5
  simp only [← c121647, ← c121648, k2, ← c121649] at e_1994_5
  have hr := e_1994_5
  simp only [e_1994_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14188)) := by
  have c121650 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121651 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121652 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121653 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121654 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121655 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_6
  simp only [← c121650, ← c121651, ← c121652] at e_1994_6
  have e_1994_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_7
  simp only [← c121653, ← c121654, k2, ← c121655] at e_1994_7
  have hr := e_1994_7
  simp only [e_1994_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14189)) := by
  have c121656 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121657 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121658 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121659 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121660 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121661 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_8
  simp only [← c121656, ← c121657, ← c121658] at e_1994_8
  have e_1994_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_9
  simp only [← c121659, ← c121660, k2, ← c121661] at e_1994_9
  have hr := e_1994_9
  simp only [e_1994_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14190)) := by
  have c121662 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121663 := (publicBatchWrapper89_copies3801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121664 := (publicBatchWrapper89_copies3802 a h).1
  have c121665 := (publicBatchWrapper89_copies3802 a h).2.1
  have c121666 := (publicBatchWrapper89_copies3802 a h).2.2.1
  have c121667 := (publicBatchWrapper89_copies3802 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_10
  simp only [← c121662, ← c121663, ← c121664] at e_1994_10
  have e_1994_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_11
  simp only [← c121665, ← c121666, k2, ← c121667] at e_1994_11
  have hr := e_1994_11
  simp only [e_1994_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14191)) := by
  have c121668 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.1
  have c121669 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.1
  have c121670 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.1
  have c121671 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.1
  have c121672 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.1
  have c121673 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_12
  simp only [← c121668, ← c121669, ← c121670] at e_1994_12
  have e_1994_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_13
  simp only [← c121671, ← c121672, k2, ← c121673] at e_1994_13
  have hr := e_1994_13
  simp only [e_1994_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 63) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14192)) := by
  have c121674 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121675 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121676 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121677 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121678 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121679 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_14
  simp only [← c121674, ← c121675, ← c121676] at e_1994_14
  have e_1994_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_15
  simp only [← c121677, ← c121678, k2, ← c121679] at e_1994_15
  have hr := e_1994_15
  simp only [e_1994_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 71) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14193)) := by
  have c121680 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121681 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121682 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121683 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121684 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121685 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_16
  simp only [← c121680, ← c121681, ← c121682] at e_1994_16
  have e_1994_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_17
  simp only [← c121683, ← c121684, k2, ← c121685] at e_1994_17
  have hr := e_1994_17
  simp only [e_1994_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1994 79) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14194)) := by
  have c121686 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121687 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121688 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121689 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121690 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121691 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1994_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_18
  simp only [← c121686, ← c121687, ← c121688] at e_1994_18
  have e_1994_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1994 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1994_19
  simp only [← c121689, ← c121690, k2, ← c121691] at e_1994_19
  have hr := e_1994_19
  simp only [e_1994_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 7) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14195)) := by
  have c121692 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121693 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121694 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121695 := (publicBatchWrapper89_copies3802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121696 := (publicBatchWrapper89_copies3803 a h).1
  have c121697 := (publicBatchWrapper89_copies3803 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_0
  simp only [← c121692, ← c121693, ← c121694] at e_1995_0
  have e_1995_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_1
  simp only [← c121695, ← c121696, k2, ← c121697] at e_1995_1
  have hr := e_1995_1
  simp only [e_1995_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 15) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14196)) := by
  have c121698 := (publicBatchWrapper89_copies3803 a h).2.2.1
  have c121699 := (publicBatchWrapper89_copies3803 a h).2.2.2.1
  have c121700 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.1
  have c121701 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.1
  have c121702 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.1
  have c121703 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_2
  simp only [← c121698, ← c121699, ← c121700] at e_1995_2
  have e_1995_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_3
  simp only [← c121701, ← c121702, k2, ← c121703] at e_1995_3
  have hr := e_1995_3
  simp only [e_1995_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 23) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14197)) := by
  have c121704 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.1
  have c121705 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.1
  have c121706 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121707 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121708 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121709 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_4
  simp only [← c121704, ← c121705, ← c121706] at e_1995_4
  have e_1995_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_5
  simp only [← c121707, ← c121708, k2, ← c121709] at e_1995_5
  have hr := e_1995_5
  simp only [e_1995_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 31) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14198)) := by
  have c121710 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121711 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121712 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121713 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121714 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121715 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_6
  simp only [← c121710, ← c121711, ← c121712] at e_1995_6
  have e_1995_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_7
  simp only [← c121713, ← c121714, k2, ← c121715] at e_1995_7
  have hr := e_1995_7
  simp only [e_1995_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 39) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14199)) := by
  have c121716 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121717 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121718 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121719 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121720 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121721 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_8
  simp only [← c121716, ← c121717, ← c121718] at e_1995_8
  have e_1995_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_9
  simp only [← c121719, ← c121720, k2, ← c121721] at e_1995_9
  have hr := e_1995_9
  simp only [e_1995_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 47) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14200)) := by
  have c121722 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121723 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121724 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121725 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121726 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121727 := (publicBatchWrapper89_copies3803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_10
  simp only [← c121722, ← c121723, ← c121724] at e_1995_10
  have e_1995_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_11
  simp only [← c121725, ← c121726, k2, ← c121727] at e_1995_11
  have hr := e_1995_11
  simp only [e_1995_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 55) = bselect (a (.wire 69 19)) (a (.virt 200)) (a (.virt 14201)) := by
  have c121728 := (publicBatchWrapper89_copies3804 a h).1
  have c121729 := (publicBatchWrapper89_copies3804 a h).2.1
  have c121730 := (publicBatchWrapper89_copies3804 a h).2.2.1
  have c121731 := (publicBatchWrapper89_copies3804 a h).2.2.2.1
  have c121732 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.1
  have c121733 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_12
  simp only [← c121728, ← c121729, ← c121730] at e_1995_12
  have e_1995_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_13
  simp only [← c121731, ← c121732, k2, ← c121733] at e_1995_13
  have hr := e_1995_13
  simp only [e_1995_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14370)) := by
  have c121734 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.1
  have c121735 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.1
  have c121736 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.1
  have c121737 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.1
  have c121738 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121739 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_14
  simp only [← c121734, ← c121735, ← c121736] at e_1995_14
  have e_1995_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_15
  simp only [← c121737, ← c121738, k2, ← c121739] at e_1995_15
  have hr := e_1995_15
  simp only [e_1995_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14371)) := by
  have c121740 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121741 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121742 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121743 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121744 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121745 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_16
  simp only [← c121740, ← c121741, ← c121742] at e_1995_16
  have e_1995_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_17
  simp only [← c121743, ← c121744, k2, ← c121745] at e_1995_17
  have hr := e_1995_17
  simp only [e_1995_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1995 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14372)) := by
  have c121746 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121747 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121748 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121749 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121750 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121751 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1995_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_18
  simp only [← c121746, ← c121747, ← c121748] at e_1995_18
  have e_1995_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1995 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1995_19
  simp only [← c121749, ← c121750, k2, ← c121751] at e_1995_19
  have hr := e_1995_19
  simp only [e_1995_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14373)) := by
  have c121752 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121753 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121754 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121755 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121756 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121757 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_0
  simp only [← c121752, ← c121753, ← c121754] at e_1996_0
  have e_1996_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_1
  simp only [← c121755, ← c121756, k2, ← c121757] at e_1996_1
  have hr := e_1996_1
  simp only [e_1996_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14374)) := by
  have c121758 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121759 := (publicBatchWrapper89_copies3804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121760 := (publicBatchWrapper89_copies3805 a h).1
  have c121761 := (publicBatchWrapper89_copies3805 a h).2.1
  have c121762 := (publicBatchWrapper89_copies3805 a h).2.2.1
  have c121763 := (publicBatchWrapper89_copies3805 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_2
  simp only [← c121758, ← c121759, ← c121760] at e_1996_2
  have e_1996_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_3
  simp only [← c121761, ← c121762, k2, ← c121763] at e_1996_3
  have hr := e_1996_3
  simp only [e_1996_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14375)) := by
  have c121764 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.1
  have c121765 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.1
  have c121766 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.1
  have c121767 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.1
  have c121768 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.1
  have c121769 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_4
  simp only [← c121764, ← c121765, ← c121766] at e_1996_4
  have e_1996_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_5
  simp only [← c121767, ← c121768, k2, ← c121769] at e_1996_5
  have hr := e_1996_5
  simp only [e_1996_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14376)) := by
  have c121770 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121771 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121772 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121773 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121774 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121775 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_6
  simp only [← c121770, ← c121771, ← c121772] at e_1996_6
  have e_1996_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_7
  simp only [← c121773, ← c121774, k2, ← c121775] at e_1996_7
  have hr := e_1996_7
  simp only [e_1996_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14377)) := by
  have c121776 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121777 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121778 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121779 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121780 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121781 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_8
  simp only [← c121776, ← c121777, ← c121778] at e_1996_8
  have e_1996_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_9
  simp only [← c121779, ← c121780, k2, ← c121781] at e_1996_9
  have hr := e_1996_9
  simp only [e_1996_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14378)) := by
  have c121782 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121783 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121784 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121785 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121786 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121787 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_10
  simp only [← c121782, ← c121783, ← c121784] at e_1996_10
  have e_1996_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_11
  simp only [← c121785, ← c121786, k2, ← c121787] at e_1996_11
  have hr := e_1996_11
  simp only [e_1996_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14379)) := by
  have c121788 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121789 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121790 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121791 := (publicBatchWrapper89_copies3805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121792 := (publicBatchWrapper89_copies3806 a h).1
  have c121793 := (publicBatchWrapper89_copies3806 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_12
  simp only [← c121788, ← c121789, ← c121790] at e_1996_12
  have e_1996_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_13
  simp only [← c121791, ← c121792, k2, ← c121793] at e_1996_13
  have hr := e_1996_13
  simp only [e_1996_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14380)) := by
  have c121794 := (publicBatchWrapper89_copies3806 a h).2.2.1
  have c121795 := (publicBatchWrapper89_copies3806 a h).2.2.2.1
  have c121796 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.1
  have c121797 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.1
  have c121798 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.1
  have c121799 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_14
  simp only [← c121794, ← c121795, ← c121796] at e_1996_14
  have e_1996_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_15
  simp only [← c121797, ← c121798, k2, ← c121799] at e_1996_15
  have hr := e_1996_15
  simp only [e_1996_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14381)) := by
  have c121800 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.1
  have c121801 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.1
  have c121802 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121803 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121804 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121805 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_16
  simp only [← c121800, ← c121801, ← c121802] at e_1996_16
  have e_1996_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_17
  simp only [← c121803, ← c121804, k2, ← c121805] at e_1996_17
  have hr := e_1996_17
  simp only [e_1996_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1996 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14382)) := by
  have c121806 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121807 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121808 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121809 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121810 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121811 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1996_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_18
  simp only [← c121806, ← c121807, ← c121808] at e_1996_18
  have e_1996_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1996 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1996_19
  simp only [← c121809, ← c121810, k2, ← c121811] at e_1996_19
  have hr := e_1996_19
  simp only [e_1996_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14383)) := by
  have c121812 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121813 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121814 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121815 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121816 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121817 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_0
  simp only [← c121812, ← c121813, ← c121814] at e_1997_0
  have e_1997_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_1
  simp only [← c121815, ← c121816, k2, ← c121817] at e_1997_1
  have hr := e_1997_1
  simp only [e_1997_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14384)) := by
  have c121818 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121819 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121820 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121821 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121822 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121823 := (publicBatchWrapper89_copies3806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_2
  simp only [← c121818, ← c121819, ← c121820] at e_1997_2
  have e_1997_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_3
  simp only [← c121821, ← c121822, k2, ← c121823] at e_1997_3
  have hr := e_1997_3
  simp only [e_1997_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14385)) := by
  have c121824 := (publicBatchWrapper89_copies3807 a h).1
  have c121825 := (publicBatchWrapper89_copies3807 a h).2.1
  have c121826 := (publicBatchWrapper89_copies3807 a h).2.2.1
  have c121827 := (publicBatchWrapper89_copies3807 a h).2.2.2.1
  have c121828 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.1
  have c121829 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_4
  simp only [← c121824, ← c121825, ← c121826] at e_1997_4
  have e_1997_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_5
  simp only [← c121827, ← c121828, k2, ← c121829] at e_1997_5
  have hr := e_1997_5
  simp only [e_1997_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14386)) := by
  have c121830 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.1
  have c121831 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.1
  have c121832 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.1
  have c121833 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.1
  have c121834 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121835 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_6
  simp only [← c121830, ← c121831, ← c121832] at e_1997_6
  have e_1997_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_7
  simp only [← c121833, ← c121834, k2, ← c121835] at e_1997_7
  have hr := e_1997_7
  simp only [e_1997_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14387)) := by
  have c121836 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121837 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121838 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121839 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121840 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121841 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_8
  simp only [← c121836, ← c121837, ← c121838] at e_1997_8
  have e_1997_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_9
  simp only [← c121839, ← c121840, k2, ← c121841] at e_1997_9
  have hr := e_1997_9
  simp only [e_1997_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14388)) := by
  have c121842 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121843 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121844 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121845 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121846 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121847 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_10
  simp only [← c121842, ← c121843, ← c121844] at e_1997_10
  have e_1997_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_11
  simp only [← c121845, ← c121846, k2, ← c121847] at e_1997_11
  have hr := e_1997_11
  simp only [e_1997_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14389)) := by
  have c121848 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121849 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121850 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121851 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121852 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121853 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_12
  simp only [← c121848, ← c121849, ← c121850] at e_1997_12
  have e_1997_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_13
  simp only [← c121851, ← c121852, k2, ← c121853] at e_1997_13
  have hr := e_1997_13
  simp only [e_1997_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14390)) := by
  have c121854 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121855 := (publicBatchWrapper89_copies3807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121856 := (publicBatchWrapper89_copies3808 a h).1
  have c121857 := (publicBatchWrapper89_copies3808 a h).2.1
  have c121858 := (publicBatchWrapper89_copies3808 a h).2.2.1
  have c121859 := (publicBatchWrapper89_copies3808 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_14
  simp only [← c121854, ← c121855, ← c121856] at e_1997_14
  have e_1997_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_15
  simp only [← c121857, ← c121858, k2, ← c121859] at e_1997_15
  have hr := e_1997_15
  simp only [e_1997_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14391)) := by
  have c121860 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.1
  have c121861 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.1
  have c121862 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.1
  have c121863 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.1
  have c121864 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.1
  have c121865 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_16
  simp only [← c121860, ← c121861, ← c121862] at e_1997_16
  have e_1997_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_17
  simp only [← c121863, ← c121864, k2, ← c121865] at e_1997_17
  have hr := e_1997_17
  simp only [e_1997_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1997 79) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14392)) := by
  have c121866 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121867 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121868 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121869 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121870 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121871 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1997_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_18
  simp only [← c121866, ← c121867, ← c121868] at e_1997_18
  have e_1997_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1997 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1997_19
  simp only [← c121869, ← c121870, k2, ← c121871] at e_1997_19
  have hr := e_1997_19
  simp only [e_1997_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 7) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14393)) := by
  have c121872 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121873 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121874 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121875 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121876 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121877 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_0
  simp only [← c121872, ← c121873, ← c121874] at e_1998_0
  have e_1998_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_1
  simp only [← c121875, ← c121876, k2, ← c121877] at e_1998_1
  have hr := e_1998_1
  simp only [e_1998_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 15) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14394)) := by
  have c121878 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121879 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121880 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121881 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121882 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121883 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_2
  simp only [← c121878, ← c121879, ← c121880] at e_1998_2
  have e_1998_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_3
  simp only [← c121881, ← c121882, k2, ← c121883] at e_1998_3
  have hr := e_1998_3
  simp only [e_1998_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 23) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14395)) := by
  have c121884 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121885 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121886 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121887 := (publicBatchWrapper89_copies3808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121888 := (publicBatchWrapper89_copies3809 a h).1
  have c121889 := (publicBatchWrapper89_copies3809 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_4
  simp only [← c121884, ← c121885, ← c121886] at e_1998_4
  have e_1998_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_5
  simp only [← c121887, ← c121888, k2, ← c121889] at e_1998_5
  have hr := e_1998_5
  simp only [e_1998_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 31) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14396)) := by
  have c121890 := (publicBatchWrapper89_copies3809 a h).2.2.1
  have c121891 := (publicBatchWrapper89_copies3809 a h).2.2.2.1
  have c121892 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.1
  have c121893 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.1
  have c121894 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.1
  have c121895 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_6
  simp only [← c121890, ← c121891, ← c121892] at e_1998_6
  have e_1998_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_7
  simp only [← c121893, ← c121894, k2, ← c121895] at e_1998_7
  have hr := e_1998_7
  simp only [e_1998_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 39) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14397)) := by
  have c121896 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.1
  have c121897 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.1
  have c121898 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121899 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121900 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121901 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_8
  simp only [← c121896, ← c121897, ← c121898] at e_1998_8
  have e_1998_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_9
  simp only [← c121899, ← c121900, k2, ← c121901] at e_1998_9
  have hr := e_1998_9
  simp only [e_1998_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 47) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14398)) := by
  have c121902 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121903 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121904 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121905 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121906 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121907 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_10
  simp only [← c121902, ← c121903, ← c121904] at e_1998_10
  have e_1998_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_11
  simp only [← c121905, ← c121906, k2, ← c121907] at e_1998_11
  have hr := e_1998_11
  simp only [e_1998_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 55) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14399)) := by
  have c121908 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121909 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121910 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121911 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121912 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121913 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_12
  simp only [← c121908, ← c121909, ← c121910] at e_1998_12
  have e_1998_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_13
  simp only [← c121911, ← c121912, k2, ← c121913] at e_1998_13
  have hr := e_1998_13
  simp only [e_1998_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 63) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14400)) := by
  have c121914 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121915 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121916 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121917 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121918 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121919 := (publicBatchWrapper89_copies3809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_14
  simp only [← c121914, ← c121915, ← c121916] at e_1998_14
  have e_1998_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_15
  simp only [← c121917, ← c121918, k2, ← c121919] at e_1998_15
  have hr := e_1998_15
  simp only [e_1998_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 71) = bselect (a (.wire 69 63)) (a (.virt 200)) (a (.virt 14401)) := by
  have c121920 := (publicBatchWrapper89_copies3810 a h).1
  have c121921 := (publicBatchWrapper89_copies3810 a h).2.1
  have c121922 := (publicBatchWrapper89_copies3810 a h).2.2.1
  have c121923 := (publicBatchWrapper89_copies3810 a h).2.2.2.1
  have c121924 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.1
  have c121925 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_16
  simp only [← c121920, ← c121921, ← c121922] at e_1998_16
  have e_1998_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_17
  simp only [← c121923, ← c121924, k2, ← c121925] at e_1998_17
  have hr := e_1998_17
  simp only [e_1998_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1998 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14570)) := by
  have c121926 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.1
  have c121927 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.1
  have c121928 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.1
  have c121929 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.1
  have c121930 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121931 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1998_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_18
  simp only [← c121926, ← c121927, ← c121928] at e_1998_18
  have e_1998_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1998 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1998_19
  simp only [← c121929, ← c121930, k2, ← c121931] at e_1998_19
  have hr := e_1998_19
  simp only [e_1998_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14571)) := by
  have c121932 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121933 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121934 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121935 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121936 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121937 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_0
  simp only [← c121932, ← c121933, ← c121934] at e_1999_0
  have e_1999_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_1
  simp only [← c121935, ← c121936, k2, ← c121937] at e_1999_1
  have hr := e_1999_1
  simp only [e_1999_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14572)) := by
  have c121938 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121939 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121940 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121941 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121942 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121943 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_2
  simp only [← c121938, ← c121939, ← c121940] at e_1999_2
  have e_1999_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_3
  simp only [← c121941, ← c121942, k2, ← c121943] at e_1999_3
  have hr := e_1999_3
  simp only [e_1999_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14573)) := by
  have c121944 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121945 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121946 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121947 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121948 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121949 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_4
  simp only [← c121944, ← c121945, ← c121946] at e_1999_4
  have e_1999_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_5
  simp only [← c121947, ← c121948, k2, ← c121949] at e_1999_5
  have hr := e_1999_5
  simp only [e_1999_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14574)) := by
  have c121950 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121951 := (publicBatchWrapper89_copies3810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121952 := (publicBatchWrapper89_copies3811 a h).1
  have c121953 := (publicBatchWrapper89_copies3811 a h).2.1
  have c121954 := (publicBatchWrapper89_copies3811 a h).2.2.1
  have c121955 := (publicBatchWrapper89_copies3811 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_6
  simp only [← c121950, ← c121951, ← c121952] at e_1999_6
  have e_1999_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_7
  simp only [← c121953, ← c121954, k2, ← c121955] at e_1999_7
  have hr := e_1999_7
  simp only [e_1999_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14575)) := by
  have c121956 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.1
  have c121957 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.1
  have c121958 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.1
  have c121959 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.1
  have c121960 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.1
  have c121961 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_8
  simp only [← c121956, ← c121957, ← c121958] at e_1999_8
  have e_1999_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_9
  simp only [← c121959, ← c121960, k2, ← c121961] at e_1999_9
  have hr := e_1999_9
  simp only [e_1999_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14576)) := by
  have c121962 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121963 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121964 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121965 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121966 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121967 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_10
  simp only [← c121962, ← c121963, ← c121964] at e_1999_10
  have e_1999_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_11
  simp only [← c121965, ← c121966, k2, ← c121967] at e_1999_11
  have hr := e_1999_11
  simp only [e_1999_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14577)) := by
  have c121968 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121969 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121970 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121971 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121972 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121973 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_12
  simp only [← c121968, ← c121969, ← c121970] at e_1999_12
  have e_1999_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_13
  simp only [← c121971, ← c121972, k2, ← c121973] at e_1999_13
  have hr := e_1999_13
  simp only [e_1999_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14578)) := by
  have c121974 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121975 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121976 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121977 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121978 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121979 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_14
  simp only [← c121974, ← c121975, ← c121976] at e_1999_14
  have e_1999_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_15
  simp only [← c121977, ← c121978, k2, ← c121979] at e_1999_15
  have hr := e_1999_15
  simp only [e_1999_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14579)) := by
  have c121980 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121981 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121982 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121983 := (publicBatchWrapper89_copies3811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c121984 := (publicBatchWrapper89_copies3812 a h).1
  have c121985 := (publicBatchWrapper89_copies3812 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_16
  simp only [← c121980, ← c121981, ← c121982] at e_1999_16
  have e_1999_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_17
  simp only [← c121983, ← c121984, k2, ← c121985] at e_1999_17
  have hr := e_1999_17
  simp only [e_1999_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1999 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14580)) := by
  have c121986 := (publicBatchWrapper89_copies3812 a h).2.2.1
  have c121987 := (publicBatchWrapper89_copies3812 a h).2.2.2.1
  have c121988 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.1
  have c121989 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.1
  have c121990 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.1
  have c121991 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1999_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_18
  simp only [← c121986, ← c121987, ← c121988] at e_1999_18
  have e_1999_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1999 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1999_19
  simp only [← c121989, ← c121990, k2, ← c121991] at e_1999_19
  have hr := e_1999_19
  simp only [e_1999_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14581)) := by
  have c121992 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.1
  have c121993 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.1
  have c121994 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.1
  have c121995 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c121996 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121997 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_0
  simp only [← c121992, ← c121993, ← c121994] at e_2000_0
  have e_2000_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_1
  simp only [← c121995, ← c121996, k2, ← c121997] at e_2000_1
  have hr := e_2000_1
  simp only [e_2000_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14582)) := by
  have c121998 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c121999 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122000 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122001 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122002 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122003 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_2
  simp only [← c121998, ← c121999, ← c122000] at e_2000_2
  have e_2000_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_3
  simp only [← c122001, ← c122002, k2, ← c122003] at e_2000_3
  have hr := e_2000_3
  simp only [e_2000_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14583)) := by
  have c122004 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122005 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122006 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122007 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122008 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122009 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_4
  simp only [← c122004, ← c122005, ← c122006] at e_2000_4
  have e_2000_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_5
  simp only [← c122007, ← c122008, k2, ← c122009] at e_2000_5
  have hr := e_2000_5
  simp only [e_2000_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14584)) := by
  have c122010 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122011 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122012 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122013 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122014 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122015 := (publicBatchWrapper89_copies3812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_6
  simp only [← c122010, ← c122011, ← c122012] at e_2000_6
  have e_2000_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_7
  simp only [← c122013, ← c122014, k2, ← c122015] at e_2000_7
  have hr := e_2000_7
  simp only [e_2000_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14585)) := by
  have c122016 := (publicBatchWrapper89_copies3813 a h).1
  have c122017 := (publicBatchWrapper89_copies3813 a h).2.1
  have c122018 := (publicBatchWrapper89_copies3813 a h).2.2.1
  have c122019 := (publicBatchWrapper89_copies3813 a h).2.2.2.1
  have c122020 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.1
  have c122021 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_8
  simp only [← c122016, ← c122017, ← c122018] at e_2000_8
  have e_2000_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_9
  simp only [← c122019, ← c122020, k2, ← c122021] at e_2000_9
  have hr := e_2000_9
  simp only [e_2000_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14586)) := by
  have c122022 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.1
  have c122023 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.1
  have c122024 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.1
  have c122025 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.1
  have c122026 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122027 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_10
  simp only [← c122022, ← c122023, ← c122024] at e_2000_10
  have e_2000_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_11
  simp only [← c122025, ← c122026, k2, ← c122027] at e_2000_11
  have hr := e_2000_11
  simp only [e_2000_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14587)) := by
  have c122028 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122029 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122030 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122031 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122032 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122033 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_12
  simp only [← c122028, ← c122029, ← c122030] at e_2000_12
  have e_2000_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_13
  simp only [← c122031, ← c122032, k2, ← c122033] at e_2000_13
  have hr := e_2000_13
  simp only [e_2000_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14588)) := by
  have c122034 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122035 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122036 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122037 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122038 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122039 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_14
  simp only [← c122034, ← c122035, ← c122036] at e_2000_14
  have e_2000_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_15
  simp only [← c122037, ← c122038, k2, ← c122039] at e_2000_15
  have hr := e_2000_15
  simp only [e_2000_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14589)) := by
  have c122040 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122041 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122042 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122043 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122044 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122045 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_16
  simp only [← c122040, ← c122041, ← c122042] at e_2000_16
  have e_2000_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_17
  simp only [← c122043, ← c122044, k2, ← c122045] at e_2000_17
  have hr := e_2000_17
  simp only [e_2000_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2000 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14590)) := by
  have c122046 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122047 := (publicBatchWrapper89_copies3813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122048 := (publicBatchWrapper89_copies3814 a h).1
  have c122049 := (publicBatchWrapper89_copies3814 a h).2.1
  have c122050 := (publicBatchWrapper89_copies3814 a h).2.2.1
  have c122051 := (publicBatchWrapper89_copies3814 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2000_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_18
  simp only [← c122046, ← c122047, ← c122048] at e_2000_18
  have e_2000_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2000 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2000_19
  simp only [← c122049, ← c122050, k2, ← c122051] at e_2000_19
  have hr := e_2000_19
  simp only [e_2000_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14591)) := by
  have c122052 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.1
  have c122053 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.1
  have c122054 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.1
  have c122055 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.1
  have c122056 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.1
  have c122057 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_0
  simp only [← c122052, ← c122053, ← c122054] at e_2001_0
  have e_2001_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_1
  simp only [← c122055, ← c122056, k2, ← c122057] at e_2001_1
  have hr := e_2001_1
  simp only [e_2001_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 15) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14592)) := by
  have c122058 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122059 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122060 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122061 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122062 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122063 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_2
  simp only [← c122058, ← c122059, ← c122060] at e_2001_2
  have e_2001_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_3
  simp only [← c122061, ← c122062, k2, ← c122063] at e_2001_3
  have hr := e_2001_3
  simp only [e_2001_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 23) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14593)) := by
  have c122064 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122065 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122066 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122067 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122068 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122069 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_4
  simp only [← c122064, ← c122065, ← c122066] at e_2001_4
  have e_2001_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_5
  simp only [← c122067, ← c122068, k2, ← c122069] at e_2001_5
  have hr := e_2001_5
  simp only [e_2001_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 31) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14594)) := by
  have c122070 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122071 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122072 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122073 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122074 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122075 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_6
  simp only [← c122070, ← c122071, ← c122072] at e_2001_6
  have e_2001_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_7
  simp only [← c122073, ← c122074, k2, ← c122075] at e_2001_7
  have hr := e_2001_7
  simp only [e_2001_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 39) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14595)) := by
  have c122076 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122077 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122078 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122079 := (publicBatchWrapper89_copies3814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122080 := (publicBatchWrapper89_copies3815 a h).1
  have c122081 := (publicBatchWrapper89_copies3815 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_8
  simp only [← c122076, ← c122077, ← c122078] at e_2001_8
  have e_2001_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_9
  simp only [← c122079, ← c122080, k2, ← c122081] at e_2001_9
  have hr := e_2001_9
  simp only [e_2001_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 47) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14596)) := by
  have c122082 := (publicBatchWrapper89_copies3815 a h).2.2.1
  have c122083 := (publicBatchWrapper89_copies3815 a h).2.2.2.1
  have c122084 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.1
  have c122085 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.1
  have c122086 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.1
  have c122087 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_10
  simp only [← c122082, ← c122083, ← c122084] at e_2001_10
  have e_2001_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_11
  simp only [← c122085, ← c122086, k2, ← c122087] at e_2001_11
  have hr := e_2001_11
  simp only [e_2001_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 55) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14597)) := by
  have c122088 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.1
  have c122089 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.1
  have c122090 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122091 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122092 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122093 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_12
  simp only [← c122088, ← c122089, ← c122090] at e_2001_12
  have e_2001_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_13
  simp only [← c122091, ← c122092, k2, ← c122093] at e_2001_13
  have hr := e_2001_13
  simp only [e_2001_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 63) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14598)) := by
  have c122094 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122095 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122096 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122097 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122098 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122099 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_14
  simp only [← c122094, ← c122095, ← c122096] at e_2001_14
  have e_2001_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_15
  simp only [← c122097, ← c122098, k2, ← c122099] at e_2001_15
  have hr := e_2001_15
  simp only [e_2001_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 71) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14599)) := by
  have c122100 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122101 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122102 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122103 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122104 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122105 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_16
  simp only [← c122100, ← c122101, ← c122102] at e_2001_16
  have e_2001_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_17
  simp only [← c122103, ← c122104, k2, ← c122105] at e_2001_17
  have hr := e_2001_17
  simp only [e_2001_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2001 79) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14600)) := by
  have c122106 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122107 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122108 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122109 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122110 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122111 := (publicBatchWrapper89_copies3815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2001_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_18
  simp only [← c122106, ← c122107, ← c122108] at e_2001_18
  have e_2001_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2001 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2001_19
  simp only [← c122109, ← c122110, k2, ← c122111] at e_2001_19
  have hr := e_2001_19
  simp only [e_2001_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 7) = bselect (a (.wire 71 27)) (a (.virt 200)) (a (.virt 14601)) := by
  have c122112 := (publicBatchWrapper89_copies3816 a h).1
  have c122113 := (publicBatchWrapper89_copies3816 a h).2.1
  have c122114 := (publicBatchWrapper89_copies3816 a h).2.2.1
  have c122115 := (publicBatchWrapper89_copies3816 a h).2.2.2.1
  have c122116 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.1
  have c122117 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_0
  simp only [← c122112, ← c122113, ← c122114] at e_2002_0
  have e_2002_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_1
  simp only [← c122115, ← c122116, k2, ← c122117] at e_2002_1
  have hr := e_2002_1
  simp only [e_2002_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14770)) := by
  have c122118 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.1
  have c122119 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.1
  have c122120 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.1
  have c122121 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.1
  have c122122 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122123 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_2
  simp only [← c122118, ← c122119, ← c122120] at e_2002_2
  have e_2002_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_3
  simp only [← c122121, ← c122122, k2, ← c122123] at e_2002_3
  have hr := e_2002_3
  simp only [e_2002_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14771)) := by
  have c122124 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122125 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122126 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122127 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122128 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122129 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_4
  simp only [← c122124, ← c122125, ← c122126] at e_2002_4
  have e_2002_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_5
  simp only [← c122127, ← c122128, k2, ← c122129] at e_2002_5
  have hr := e_2002_5
  simp only [e_2002_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14772)) := by
  have c122130 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122131 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122132 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122133 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122134 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122135 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_6
  simp only [← c122130, ← c122131, ← c122132] at e_2002_6
  have e_2002_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_7
  simp only [← c122133, ← c122134, k2, ← c122135] at e_2002_7
  have hr := e_2002_7
  simp only [e_2002_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14773)) := by
  have c122136 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122137 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122138 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122139 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122140 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122141 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_8
  simp only [← c122136, ← c122137, ← c122138] at e_2002_8
  have e_2002_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_9
  simp only [← c122139, ← c122140, k2, ← c122141] at e_2002_9
  have hr := e_2002_9
  simp only [e_2002_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14774)) := by
  have c122142 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122143 := (publicBatchWrapper89_copies3816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122144 := (publicBatchWrapper89_copies3817 a h).1
  have c122145 := (publicBatchWrapper89_copies3817 a h).2.1
  have c122146 := (publicBatchWrapper89_copies3817 a h).2.2.1
  have c122147 := (publicBatchWrapper89_copies3817 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_10
  simp only [← c122142, ← c122143, ← c122144] at e_2002_10
  have e_2002_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_11
  simp only [← c122145, ← c122146, k2, ← c122147] at e_2002_11
  have hr := e_2002_11
  simp only [e_2002_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14775)) := by
  have c122148 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.1
  have c122149 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.1
  have c122150 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.1
  have c122151 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.1
  have c122152 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.1
  have c122153 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_12
  simp only [← c122148, ← c122149, ← c122150] at e_2002_12
  have e_2002_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_13
  simp only [← c122151, ← c122152, k2, ← c122153] at e_2002_13
  have hr := e_2002_13
  simp only [e_2002_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14776)) := by
  have c122154 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122155 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122156 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122157 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122158 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122159 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_14
  simp only [← c122154, ← c122155, ← c122156] at e_2002_14
  have e_2002_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_15
  simp only [← c122157, ← c122158, k2, ← c122159] at e_2002_15
  have hr := e_2002_15
  simp only [e_2002_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14777)) := by
  have c122160 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122161 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122162 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122163 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122164 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122165 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_16
  simp only [← c122160, ← c122161, ← c122162] at e_2002_16
  have e_2002_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_17
  simp only [← c122163, ← c122164, k2, ← c122165] at e_2002_17
  have hr := e_2002_17
  simp only [e_2002_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2002 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14778)) := by
  have c122166 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122167 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122168 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122169 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122170 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122171 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2002_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_18
  simp only [← c122166, ← c122167, ← c122168] at e_2002_18
  have e_2002_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2002 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2002_19
  simp only [← c122169, ← c122170, k2, ← c122171] at e_2002_19
  have hr := e_2002_19
  simp only [e_2002_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14779)) := by
  have c122172 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122173 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122174 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122175 := (publicBatchWrapper89_copies3817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122176 := (publicBatchWrapper89_copies3818 a h).1
  have c122177 := (publicBatchWrapper89_copies3818 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_0
  simp only [← c122172, ← c122173, ← c122174] at e_2003_0
  have e_2003_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_1
  simp only [← c122175, ← c122176, k2, ← c122177] at e_2003_1
  have hr := e_2003_1
  simp only [e_2003_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14780)) := by
  have c122178 := (publicBatchWrapper89_copies3818 a h).2.2.1
  have c122179 := (publicBatchWrapper89_copies3818 a h).2.2.2.1
  have c122180 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.1
  have c122181 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.1
  have c122182 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.1
  have c122183 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_2
  simp only [← c122178, ← c122179, ← c122180] at e_2003_2
  have e_2003_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_3
  simp only [← c122181, ← c122182, k2, ← c122183] at e_2003_3
  have hr := e_2003_3
  simp only [e_2003_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14781)) := by
  have c122184 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.1
  have c122185 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.1
  have c122186 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122187 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122188 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122189 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_4
  simp only [← c122184, ← c122185, ← c122186] at e_2003_4
  have e_2003_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_5
  simp only [← c122187, ← c122188, k2, ← c122189] at e_2003_5
  have hr := e_2003_5
  simp only [e_2003_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14782)) := by
  have c122190 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122191 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122192 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122193 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122194 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122195 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_6
  simp only [← c122190, ← c122191, ← c122192] at e_2003_6
  have e_2003_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_7
  simp only [← c122193, ← c122194, k2, ← c122195] at e_2003_7
  have hr := e_2003_7
  simp only [e_2003_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14783)) := by
  have c122196 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122197 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122198 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122199 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122200 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122201 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_8
  simp only [← c122196, ← c122197, ← c122198] at e_2003_8
  have e_2003_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_9
  simp only [← c122199, ← c122200, k2, ← c122201] at e_2003_9
  have hr := e_2003_9
  simp only [e_2003_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14784)) := by
  have c122202 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122203 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122204 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122205 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122206 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122207 := (publicBatchWrapper89_copies3818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_10
  simp only [← c122202, ← c122203, ← c122204] at e_2003_10
  have e_2003_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_11
  simp only [← c122205, ← c122206, k2, ← c122207] at e_2003_11
  have hr := e_2003_11
  simp only [e_2003_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14785)) := by
  have c122208 := (publicBatchWrapper89_copies3819 a h).1
  have c122209 := (publicBatchWrapper89_copies3819 a h).2.1
  have c122210 := (publicBatchWrapper89_copies3819 a h).2.2.1
  have c122211 := (publicBatchWrapper89_copies3819 a h).2.2.2.1
  have c122212 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.1
  have c122213 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_12
  simp only [← c122208, ← c122209, ← c122210] at e_2003_12
  have e_2003_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_13
  simp only [← c122211, ← c122212, k2, ← c122213] at e_2003_13
  have hr := e_2003_13
  simp only [e_2003_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14786)) := by
  have c122214 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.1
  have c122215 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.1
  have c122216 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.1
  have c122217 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.1
  have c122218 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122219 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_14
  simp only [← c122214, ← c122215, ← c122216] at e_2003_14
  have e_2003_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_15
  simp only [← c122217, ← c122218, k2, ← c122219] at e_2003_15
  have hr := e_2003_15
  simp only [e_2003_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14787)) := by
  have c122220 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122221 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122222 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122223 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122224 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122225 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_16
  simp only [← c122220, ← c122221, ← c122222] at e_2003_16
  have e_2003_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_17
  simp only [← c122223, ← c122224, k2, ← c122225] at e_2003_17
  have hr := e_2003_17
  simp only [e_2003_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2003 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14788)) := by
  have c122226 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122227 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122228 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122229 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122230 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122231 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2003_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_18
  simp only [← c122226, ← c122227, ← c122228] at e_2003_18
  have e_2003_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2003 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2003_19
  simp only [← c122229, ← c122230, k2, ← c122231] at e_2003_19
  have hr := e_2003_19
  simp only [e_2003_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14789)) := by
  have c122232 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122233 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122234 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122235 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122236 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122237 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_0
  simp only [← c122232, ← c122233, ← c122234] at e_2004_0
  have e_2004_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_1
  simp only [← c122235, ← c122236, k2, ← c122237] at e_2004_1
  have hr := e_2004_1
  simp only [e_2004_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14790)) := by
  have c122238 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122239 := (publicBatchWrapper89_copies3819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122240 := (publicBatchWrapper89_copies3820 a h).1
  have c122241 := (publicBatchWrapper89_copies3820 a h).2.1
  have c122242 := (publicBatchWrapper89_copies3820 a h).2.2.1
  have c122243 := (publicBatchWrapper89_copies3820 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_2
  simp only [← c122238, ← c122239, ← c122240] at e_2004_2
  have e_2004_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_3
  simp only [← c122241, ← c122242, k2, ← c122243] at e_2004_3
  have hr := e_2004_3
  simp only [e_2004_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14791)) := by
  have c122244 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.1
  have c122245 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.1
  have c122246 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.1
  have c122247 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.1
  have c122248 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.1
  have c122249 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_4
  simp only [← c122244, ← c122245, ← c122246] at e_2004_4
  have e_2004_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_5
  simp only [← c122247, ← c122248, k2, ← c122249] at e_2004_5
  have hr := e_2004_5
  simp only [e_2004_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 31) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14792)) := by
  have c122250 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122251 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122252 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122253 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122254 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122255 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_6
  simp only [← c122250, ← c122251, ← c122252] at e_2004_6
  have e_2004_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_7
  simp only [← c122253, ← c122254, k2, ← c122255] at e_2004_7
  have hr := e_2004_7
  simp only [e_2004_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 39) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14793)) := by
  have c122256 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122257 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122258 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122259 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122260 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122261 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_8
  simp only [← c122256, ← c122257, ← c122258] at e_2004_8
  have e_2004_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_9
  simp only [← c122259, ← c122260, k2, ← c122261] at e_2004_9
  have hr := e_2004_9
  simp only [e_2004_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 47) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14794)) := by
  have c122262 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122263 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122264 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122265 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122266 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122267 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_10
  simp only [← c122262, ← c122263, ← c122264] at e_2004_10
  have e_2004_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_11
  simp only [← c122265, ← c122266, k2, ← c122267] at e_2004_11
  have hr := e_2004_11
  simp only [e_2004_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 55) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14795)) := by
  have c122268 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122269 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122270 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122271 := (publicBatchWrapper89_copies3820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122272 := (publicBatchWrapper89_copies3821 a h).1
  have c122273 := (publicBatchWrapper89_copies3821 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_12
  simp only [← c122268, ← c122269, ← c122270] at e_2004_12
  have e_2004_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_13
  simp only [← c122271, ← c122272, k2, ← c122273] at e_2004_13
  have hr := e_2004_13
  simp only [e_2004_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 63) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14796)) := by
  have c122274 := (publicBatchWrapper89_copies3821 a h).2.2.1
  have c122275 := (publicBatchWrapper89_copies3821 a h).2.2.2.1
  have c122276 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.1
  have c122277 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.1
  have c122278 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.1
  have c122279 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_14
  simp only [← c122274, ← c122275, ← c122276] at e_2004_14
  have e_2004_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_15
  simp only [← c122277, ← c122278, k2, ← c122279] at e_2004_15
  have hr := e_2004_15
  simp only [e_2004_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 71) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14797)) := by
  have c122280 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.1
  have c122281 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.1
  have c122282 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122283 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122284 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122285 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_16
  simp only [← c122280, ← c122281, ← c122282] at e_2004_16
  have e_2004_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_17
  simp only [← c122283, ← c122284, k2, ← c122285] at e_2004_17
  have hr := e_2004_17
  simp only [e_2004_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2004 79) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14798)) := by
  have c122286 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122287 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122288 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122289 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122290 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122291 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2004_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_18
  simp only [← c122286, ← c122287, ← c122288] at e_2004_18
  have e_2004_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2004 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2004_19
  simp only [← c122289, ← c122290, k2, ← c122291] at e_2004_19
  have hr := e_2004_19
  simp only [e_2004_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 7) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14799)) := by
  have c122292 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122293 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122294 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122295 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122296 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122297 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_0
  simp only [← c122292, ← c122293, ← c122294] at e_2005_0
  have e_2005_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_1
  simp only [← c122295, ← c122296, k2, ← c122297] at e_2005_1
  have hr := e_2005_1
  simp only [e_2005_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 15) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14800)) := by
  have c122298 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122299 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122300 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122301 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122302 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122303 := (publicBatchWrapper89_copies3821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_2
  simp only [← c122298, ← c122299, ← c122300] at e_2005_2
  have e_2005_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_3
  simp only [← c122301, ← c122302, k2, ← c122303] at e_2005_3
  have hr := e_2005_3
  simp only [e_2005_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 23) = bselect (a (.wire 71 71)) (a (.virt 200)) (a (.virt 14801)) := by
  have c122304 := (publicBatchWrapper89_copies3822 a h).1
  have c122305 := (publicBatchWrapper89_copies3822 a h).2.1
  have c122306 := (publicBatchWrapper89_copies3822 a h).2.2.1
  have c122307 := (publicBatchWrapper89_copies3822 a h).2.2.2.1
  have c122308 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.1
  have c122309 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_4
  simp only [← c122304, ← c122305, ← c122306] at e_2005_4
  have e_2005_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_5
  simp only [← c122307, ← c122308, k2, ← c122309] at e_2005_5
  have hr := e_2005_5
  simp only [e_2005_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14970)) := by
  have c122310 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.1
  have c122311 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.1
  have c122312 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.1
  have c122313 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.1
  have c122314 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122315 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_6
  simp only [← c122310, ← c122311, ← c122312] at e_2005_6
  have e_2005_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_7
  simp only [← c122313, ← c122314, k2, ← c122315] at e_2005_7
  have hr := e_2005_7
  simp only [e_2005_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14971)) := by
  have c122316 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122317 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122318 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122319 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122320 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122321 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_8
  simp only [← c122316, ← c122317, ← c122318] at e_2005_8
  have e_2005_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_9
  simp only [← c122319, ← c122320, k2, ← c122321] at e_2005_9
  have hr := e_2005_9
  simp only [e_2005_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14972)) := by
  have c122322 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122323 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122324 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122325 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122326 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122327 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_10
  simp only [← c122322, ← c122323, ← c122324] at e_2005_10
  have e_2005_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_11
  simp only [← c122325, ← c122326, k2, ← c122327] at e_2005_11
  have hr := e_2005_11
  simp only [e_2005_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14973)) := by
  have c122328 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122329 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122330 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122331 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122332 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122333 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_12
  simp only [← c122328, ← c122329, ← c122330] at e_2005_12
  have e_2005_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_13
  simp only [← c122331, ← c122332, k2, ← c122333] at e_2005_13
  have hr := e_2005_13
  simp only [e_2005_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14974)) := by
  have c122334 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122335 := (publicBatchWrapper89_copies3822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122336 := (publicBatchWrapper89_copies3823 a h).1
  have c122337 := (publicBatchWrapper89_copies3823 a h).2.1
  have c122338 := (publicBatchWrapper89_copies3823 a h).2.2.1
  have c122339 := (publicBatchWrapper89_copies3823 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_14
  simp only [← c122334, ← c122335, ← c122336] at e_2005_14
  have e_2005_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_15
  simp only [← c122337, ← c122338, k2, ← c122339] at e_2005_15
  have hr := e_2005_15
  simp only [e_2005_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14975)) := by
  have c122340 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.1
  have c122341 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.1
  have c122342 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.1
  have c122343 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.1
  have c122344 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.1
  have c122345 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_16
  simp only [← c122340, ← c122341, ← c122342] at e_2005_16
  have e_2005_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_17
  simp only [← c122343, ← c122344, k2, ← c122345] at e_2005_17
  have hr := e_2005_17
  simp only [e_2005_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2005 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14976)) := by
  have c122346 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122347 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122348 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122349 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122350 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122351 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2005_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_18
  simp only [← c122346, ← c122347, ← c122348] at e_2005_18
  have e_2005_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2005 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2005_19
  simp only [← c122349, ← c122350, k2, ← c122351] at e_2005_19
  have hr := e_2005_19
  simp only [e_2005_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14977)) := by
  have c122352 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122353 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122354 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122355 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122356 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122357 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_0
  simp only [← c122352, ← c122353, ← c122354] at e_2006_0
  have e_2006_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_1
  simp only [← c122355, ← c122356, k2, ← c122357] at e_2006_1
  have hr := e_2006_1
  simp only [e_2006_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14978)) := by
  have c122358 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122359 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122360 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122361 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122362 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122363 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_2
  simp only [← c122358, ← c122359, ← c122360] at e_2006_2
  have e_2006_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_3
  simp only [← c122361, ← c122362, k2, ← c122363] at e_2006_3
  have hr := e_2006_3
  simp only [e_2006_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14979)) := by
  have c122364 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122365 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122366 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122367 := (publicBatchWrapper89_copies3823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122368 := (publicBatchWrapper89_copies3824 a h).1
  have c122369 := (publicBatchWrapper89_copies3824 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_4
  simp only [← c122364, ← c122365, ← c122366] at e_2006_4
  have e_2006_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_5
  simp only [← c122367, ← c122368, k2, ← c122369] at e_2006_5
  have hr := e_2006_5
  simp only [e_2006_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14980)) := by
  have c122370 := (publicBatchWrapper89_copies3824 a h).2.2.1
  have c122371 := (publicBatchWrapper89_copies3824 a h).2.2.2.1
  have c122372 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.1
  have c122373 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.1
  have c122374 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.1
  have c122375 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_6
  simp only [← c122370, ← c122371, ← c122372] at e_2006_6
  have e_2006_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_7
  simp only [← c122373, ← c122374, k2, ← c122375] at e_2006_7
  have hr := e_2006_7
  simp only [e_2006_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14981)) := by
  have c122376 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.1
  have c122377 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.1
  have c122378 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122379 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122380 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122381 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_8
  simp only [← c122376, ← c122377, ← c122378] at e_2006_8
  have e_2006_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_9
  simp only [← c122379, ← c122380, k2, ← c122381] at e_2006_9
  have hr := e_2006_9
  simp only [e_2006_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14982)) := by
  have c122382 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122383 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122384 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122385 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122386 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122387 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_10
  simp only [← c122382, ← c122383, ← c122384] at e_2006_10
  have e_2006_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_11
  simp only [← c122385, ← c122386, k2, ← c122387] at e_2006_11
  have hr := e_2006_11
  simp only [e_2006_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14983)) := by
  have c122388 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122389 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122390 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122391 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122392 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122393 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_12
  simp only [← c122388, ← c122389, ← c122390] at e_2006_12
  have e_2006_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_13
  simp only [← c122391, ← c122392, k2, ← c122393] at e_2006_13
  have hr := e_2006_13
  simp only [e_2006_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14984)) := by
  have c122394 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122395 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122396 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122397 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122398 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122399 := (publicBatchWrapper89_copies3824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_14
  simp only [← c122394, ← c122395, ← c122396] at e_2006_14
  have e_2006_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_15
  simp only [← c122397, ← c122398, k2, ← c122399] at e_2006_15
  have hr := e_2006_15
  simp only [e_2006_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14985)) := by
  have c122400 := (publicBatchWrapper89_copies3825 a h).1
  have c122401 := (publicBatchWrapper89_copies3825 a h).2.1
  have c122402 := (publicBatchWrapper89_copies3825 a h).2.2.1
  have c122403 := (publicBatchWrapper89_copies3825 a h).2.2.2.1
  have c122404 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.1
  have c122405 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_16
  simp only [← c122400, ← c122401, ← c122402] at e_2006_16
  have e_2006_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_17
  simp only [← c122403, ← c122404, k2, ← c122405] at e_2006_17
  have hr := e_2006_17
  simp only [e_2006_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2006 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14986)) := by
  have c122406 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.1
  have c122407 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.1
  have c122408 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.1
  have c122409 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.1
  have c122410 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122411 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2006_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_18
  simp only [← c122406, ← c122407, ← c122408] at e_2006_18
  have e_2006_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2006 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2006_19
  simp only [← c122409, ← c122410, k2, ← c122411] at e_2006_19
  have hr := e_2006_19
  simp only [e_2006_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14987)) := by
  have c122412 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122413 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122414 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122415 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122416 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122417 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_0
  simp only [← c122412, ← c122413, ← c122414] at e_2007_0
  have e_2007_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_1
  simp only [← c122415, ← c122416, k2, ← c122417] at e_2007_1
  have hr := e_2007_1
  simp only [e_2007_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14988)) := by
  have c122418 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122419 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122420 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122421 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122422 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122423 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_2
  simp only [← c122418, ← c122419, ← c122420] at e_2007_2
  have e_2007_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_3
  simp only [← c122421, ← c122422, k2, ← c122423] at e_2007_3
  have hr := e_2007_3
  simp only [e_2007_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14989)) := by
  have c122424 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122425 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122426 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122427 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122428 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122429 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_4
  simp only [← c122424, ← c122425, ← c122426] at e_2007_4
  have e_2007_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_5
  simp only [← c122427, ← c122428, k2, ← c122429] at e_2007_5
  have hr := e_2007_5
  simp only [e_2007_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14990)) := by
  have c122430 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122431 := (publicBatchWrapper89_copies3825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122432 := (publicBatchWrapper89_copies3826 a h).1
  have c122433 := (publicBatchWrapper89_copies3826 a h).2.1
  have c122434 := (publicBatchWrapper89_copies3826 a h).2.2.1
  have c122435 := (publicBatchWrapper89_copies3826 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_6
  simp only [← c122430, ← c122431, ← c122432] at e_2007_6
  have e_2007_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_7
  simp only [← c122433, ← c122434, k2, ← c122435] at e_2007_7
  have hr := e_2007_7
  simp only [e_2007_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14991)) := by
  have c122436 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.1
  have c122437 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.1
  have c122438 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.1
  have c122439 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.1
  have c122440 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.1
  have c122441 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_8
  simp only [← c122436, ← c122437, ← c122438] at e_2007_8
  have e_2007_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_9
  simp only [← c122439, ← c122440, k2, ← c122441] at e_2007_9
  have hr := e_2007_9
  simp only [e_2007_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 47) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14992)) := by
  have c122442 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122443 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122444 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122445 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122446 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122447 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_10
  simp only [← c122442, ← c122443, ← c122444] at e_2007_10
  have e_2007_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_11
  simp only [← c122445, ← c122446, k2, ← c122447] at e_2007_11
  have hr := e_2007_11
  simp only [e_2007_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 55) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14993)) := by
  have c122448 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122449 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122450 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122451 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122452 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122453 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_12
  simp only [← c122448, ← c122449, ← c122450] at e_2007_12
  have e_2007_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_13
  simp only [← c122451, ← c122452, k2, ← c122453] at e_2007_13
  have hr := e_2007_13
  simp only [e_2007_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 63) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14994)) := by
  have c122454 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122455 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122456 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122457 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122458 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122459 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_14
  simp only [← c122454, ← c122455, ← c122456] at e_2007_14
  have e_2007_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_15
  simp only [← c122457, ← c122458, k2, ← c122459] at e_2007_15
  have hr := e_2007_15
  simp only [e_2007_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 71) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14995)) := by
  have c122460 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122461 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122462 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122463 := (publicBatchWrapper89_copies3826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122464 := (publicBatchWrapper89_copies3827 a h).1
  have c122465 := (publicBatchWrapper89_copies3827 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_16
  simp only [← c122460, ← c122461, ← c122462] at e_2007_16
  have e_2007_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_17
  simp only [← c122463, ← c122464, k2, ← c122465] at e_2007_17
  have hr := e_2007_17
  simp only [e_2007_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2007 79) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14996)) := by
  have c122466 := (publicBatchWrapper89_copies3827 a h).2.2.1
  have c122467 := (publicBatchWrapper89_copies3827 a h).2.2.2.1
  have c122468 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.1
  have c122469 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.1
  have c122470 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.1
  have c122471 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2007_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_18
  simp only [← c122466, ← c122467, ← c122468] at e_2007_18
  have e_2007_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2007 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2007_19
  simp only [← c122469, ← c122470, k2, ← c122471] at e_2007_19
  have hr := e_2007_19
  simp only [e_2007_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 7) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14997)) := by
  have c122472 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.1
  have c122473 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.1
  have c122474 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122475 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122476 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122477 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_0
  simp only [← c122472, ← c122473, ← c122474] at e_2008_0
  have e_2008_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_1
  simp only [← c122475, ← c122476, k2, ← c122477] at e_2008_1
  have hr := e_2008_1
  simp only [e_2008_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 15) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14998)) := by
  have c122478 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122479 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122480 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122481 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122482 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122483 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_2
  simp only [← c122478, ← c122479, ← c122480] at e_2008_2
  have e_2008_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_3
  simp only [← c122481, ← c122482, k2, ← c122483] at e_2008_3
  have hr := e_2008_3
  simp only [e_2008_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 23) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 14999)) := by
  have c122484 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122485 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122486 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122487 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122488 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122489 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_4
  simp only [← c122484, ← c122485, ← c122486] at e_2008_4
  have e_2008_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_5
  simp only [← c122487, ← c122488, k2, ← c122489] at e_2008_5
  have hr := e_2008_5
  simp only [e_2008_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 31) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 15000)) := by
  have c122490 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122491 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122492 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122493 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122494 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122495 := (publicBatchWrapper89_copies3827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_6
  simp only [← c122490, ← c122491, ← c122492] at e_2008_6
  have e_2008_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_7
  simp only [← c122493, ← c122494, k2, ← c122495] at e_2008_7
  have hr := e_2008_7
  simp only [e_2008_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 39) = bselect (a (.wire 72 35)) (a (.virt 200)) (a (.virt 15001)) := by
  have c122496 := (publicBatchWrapper89_copies3828 a h).1
  have c122497 := (publicBatchWrapper89_copies3828 a h).2.1
  have c122498 := (publicBatchWrapper89_copies3828 a h).2.2.1
  have c122499 := (publicBatchWrapper89_copies3828 a h).2.2.2.1
  have c122500 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.1
  have c122501 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_8
  simp only [← c122496, ← c122497, ← c122498] at e_2008_8
  have e_2008_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_9
  simp only [← c122499, ← c122500, k2, ← c122501] at e_2008_9
  have hr := e_2008_9
  simp only [e_2008_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15170)) := by
  have c122502 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.1
  have c122503 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.1
  have c122504 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.1
  have c122505 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.1
  have c122506 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122507 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_10
  simp only [← c122502, ← c122503, ← c122504] at e_2008_10
  have e_2008_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_11
  simp only [← c122505, ← c122506, k2, ← c122507] at e_2008_11
  have hr := e_2008_11
  simp only [e_2008_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15171)) := by
  have c122508 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122509 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122510 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122511 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122512 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122513 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_12
  simp only [← c122508, ← c122509, ← c122510] at e_2008_12
  have e_2008_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_13
  simp only [← c122511, ← c122512, k2, ← c122513] at e_2008_13
  have hr := e_2008_13
  simp only [e_2008_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15172)) := by
  have c122514 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122515 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122516 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122517 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122518 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122519 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_14
  simp only [← c122514, ← c122515, ← c122516] at e_2008_14
  have e_2008_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_15
  simp only [← c122517, ← c122518, k2, ← c122519] at e_2008_15
  have hr := e_2008_15
  simp only [e_2008_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15173)) := by
  have c122520 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122521 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122522 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122523 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122524 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122525 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_16
  simp only [← c122520, ← c122521, ← c122522] at e_2008_16
  have e_2008_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_17
  simp only [← c122523, ← c122524, k2, ← c122525] at e_2008_17
  have hr := e_2008_17
  simp only [e_2008_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2008 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15174)) := by
  have c122526 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122527 := (publicBatchWrapper89_copies3828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122528 := (publicBatchWrapper89_copies3829 a h).1
  have c122529 := (publicBatchWrapper89_copies3829 a h).2.1
  have c122530 := (publicBatchWrapper89_copies3829 a h).2.2.1
  have c122531 := (publicBatchWrapper89_copies3829 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2008_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_18
  simp only [← c122526, ← c122527, ← c122528] at e_2008_18
  have e_2008_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2008 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2008_19
  simp only [← c122529, ← c122530, k2, ← c122531] at e_2008_19
  have hr := e_2008_19
  simp only [e_2008_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15175)) := by
  have c122532 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.1
  have c122533 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.1
  have c122534 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.1
  have c122535 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.1
  have c122536 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.1
  have c122537 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_0
  simp only [← c122532, ← c122533, ← c122534] at e_2009_0
  have e_2009_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_1
  simp only [← c122535, ← c122536, k2, ← c122537] at e_2009_1
  have hr := e_2009_1
  simp only [e_2009_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15176)) := by
  have c122538 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122539 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122540 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122541 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122542 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122543 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_2
  simp only [← c122538, ← c122539, ← c122540] at e_2009_2
  have e_2009_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_3
  simp only [← c122541, ← c122542, k2, ← c122543] at e_2009_3
  have hr := e_2009_3
  simp only [e_2009_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15177)) := by
  have c122544 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122545 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122546 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122547 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122548 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122549 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_4
  simp only [← c122544, ← c122545, ← c122546] at e_2009_4
  have e_2009_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_5
  simp only [← c122547, ← c122548, k2, ← c122549] at e_2009_5
  have hr := e_2009_5
  simp only [e_2009_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15178)) := by
  have c122550 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122551 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122552 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122553 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122554 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122555 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_6
  simp only [← c122550, ← c122551, ← c122552] at e_2009_6
  have e_2009_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_7
  simp only [← c122553, ← c122554, k2, ← c122555] at e_2009_7
  have hr := e_2009_7
  simp only [e_2009_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15179)) := by
  have c122556 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122557 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122558 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122559 := (publicBatchWrapper89_copies3829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122560 := (publicBatchWrapper89_copies3830 a h).1
  have c122561 := (publicBatchWrapper89_copies3830 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_8
  simp only [← c122556, ← c122557, ← c122558] at e_2009_8
  have e_2009_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_9
  simp only [← c122559, ← c122560, k2, ← c122561] at e_2009_9
  have hr := e_2009_9
  simp only [e_2009_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15180)) := by
  have c122562 := (publicBatchWrapper89_copies3830 a h).2.2.1
  have c122563 := (publicBatchWrapper89_copies3830 a h).2.2.2.1
  have c122564 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.1
  have c122565 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.1
  have c122566 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.1
  have c122567 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_10
  simp only [← c122562, ← c122563, ← c122564] at e_2009_10
  have e_2009_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_11
  simp only [← c122565, ← c122566, k2, ← c122567] at e_2009_11
  have hr := e_2009_11
  simp only [e_2009_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15181)) := by
  have c122568 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.1
  have c122569 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.1
  have c122570 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122571 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122572 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122573 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_12
  simp only [← c122568, ← c122569, ← c122570] at e_2009_12
  have e_2009_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_13
  simp only [← c122571, ← c122572, k2, ← c122573] at e_2009_13
  have hr := e_2009_13
  simp only [e_2009_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15182)) := by
  have c122574 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122575 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122576 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122577 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122578 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122579 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_14
  simp only [← c122574, ← c122575, ← c122576] at e_2009_14
  have e_2009_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_15
  simp only [← c122577, ← c122578, k2, ← c122579] at e_2009_15
  have hr := e_2009_15
  simp only [e_2009_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15183)) := by
  have c122580 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122581 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122582 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122583 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122584 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122585 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_16
  simp only [← c122580, ← c122581, ← c122582] at e_2009_16
  have e_2009_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_17
  simp only [← c122583, ← c122584, k2, ← c122585] at e_2009_17
  have hr := e_2009_17
  simp only [e_2009_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2009 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15184)) := by
  have c122586 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122587 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122588 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122589 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122590 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122591 := (publicBatchWrapper89_copies3830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2009_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_18
  simp only [← c122586, ← c122587, ← c122588] at e_2009_18
  have e_2009_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2009 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2009_19
  simp only [← c122589, ← c122590, k2, ← c122591] at e_2009_19
  have hr := e_2009_19
  simp only [e_2009_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15185)) := by
  have c122592 := (publicBatchWrapper89_copies3831 a h).1
  have c122593 := (publicBatchWrapper89_copies3831 a h).2.1
  have c122594 := (publicBatchWrapper89_copies3831 a h).2.2.1
  have c122595 := (publicBatchWrapper89_copies3831 a h).2.2.2.1
  have c122596 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.1
  have c122597 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_0
  simp only [← c122592, ← c122593, ← c122594] at e_2010_0
  have e_2010_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_1
  simp only [← c122595, ← c122596, k2, ← c122597] at e_2010_1
  have hr := e_2010_1
  simp only [e_2010_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15186)) := by
  have c122598 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.1
  have c122599 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.1
  have c122600 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.1
  have c122601 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.1
  have c122602 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122603 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_2
  simp only [← c122598, ← c122599, ← c122600] at e_2010_2
  have e_2010_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_3
  simp only [← c122601, ← c122602, k2, ← c122603] at e_2010_3
  have hr := e_2010_3
  simp only [e_2010_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15187)) := by
  have c122604 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122605 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122606 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122607 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122608 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122609 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_4
  simp only [← c122604, ← c122605, ← c122606] at e_2010_4
  have e_2010_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_5
  simp only [← c122607, ← c122608, k2, ← c122609] at e_2010_5
  have hr := e_2010_5
  simp only [e_2010_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15188)) := by
  have c122610 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122611 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122612 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122613 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122614 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122615 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_6
  simp only [← c122610, ← c122611, ← c122612] at e_2010_6
  have e_2010_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_7
  simp only [← c122613, ← c122614, k2, ← c122615] at e_2010_7
  have hr := e_2010_7
  simp only [e_2010_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15189)) := by
  have c122616 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122617 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122618 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122619 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122620 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122621 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_8
  simp only [← c122616, ← c122617, ← c122618] at e_2010_8
  have e_2010_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_9
  simp only [← c122619, ← c122620, k2, ← c122621] at e_2010_9
  have hr := e_2010_9
  simp only [e_2010_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15190)) := by
  have c122622 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122623 := (publicBatchWrapper89_copies3831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122624 := (publicBatchWrapper89_copies3832 a h).1
  have c122625 := (publicBatchWrapper89_copies3832 a h).2.1
  have c122626 := (publicBatchWrapper89_copies3832 a h).2.2.1
  have c122627 := (publicBatchWrapper89_copies3832 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_10
  simp only [← c122622, ← c122623, ← c122624] at e_2010_10
  have e_2010_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_11
  simp only [← c122625, ← c122626, k2, ← c122627] at e_2010_11
  have hr := e_2010_11
  simp only [e_2010_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15191)) := by
  have c122628 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.1
  have c122629 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.1
  have c122630 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.1
  have c122631 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.1
  have c122632 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.1
  have c122633 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_12
  simp only [← c122628, ← c122629, ← c122630] at e_2010_12
  have e_2010_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_13
  simp only [← c122631, ← c122632, k2, ← c122633] at e_2010_13
  have hr := e_2010_13
  simp only [e_2010_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15192)) := by
  have c122634 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122635 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122636 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122637 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122638 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122639 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_14
  simp only [← c122634, ← c122635, ← c122636] at e_2010_14
  have e_2010_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_15
  simp only [← c122637, ← c122638, k2, ← c122639] at e_2010_15
  have hr := e_2010_15
  simp only [e_2010_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15193)) := by
  have c122640 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122641 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122642 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122643 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122644 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122645 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_16
  simp only [← c122640, ← c122641, ← c122642] at e_2010_16
  have e_2010_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_17
  simp only [← c122643, ← c122644, k2, ← c122645] at e_2010_17
  have hr := e_2010_17
  simp only [e_2010_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2010 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15194)) := by
  have c122646 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122647 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122648 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122649 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122650 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122651 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2010_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_18
  simp only [← c122646, ← c122647, ← c122648] at e_2010_18
  have e_2010_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2010 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2010_19
  simp only [← c122649, ← c122650, k2, ← c122651] at e_2010_19
  have hr := e_2010_19
  simp only [e_2010_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15195)) := by
  have c122652 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122653 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122654 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122655 := (publicBatchWrapper89_copies3832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122656 := (publicBatchWrapper89_copies3833 a h).1
  have c122657 := (publicBatchWrapper89_copies3833 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_0
  simp only [← c122652, ← c122653, ← c122654] at e_2011_0
  have e_2011_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_1
  simp only [← c122655, ← c122656, k2, ← c122657] at e_2011_1
  have hr := e_2011_1
  simp only [e_2011_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15196)) := by
  have c122658 := (publicBatchWrapper89_copies3833 a h).2.2.1
  have c122659 := (publicBatchWrapper89_copies3833 a h).2.2.2.1
  have c122660 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.1
  have c122661 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.1
  have c122662 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.1
  have c122663 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_2
  simp only [← c122658, ← c122659, ← c122660] at e_2011_2
  have e_2011_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_3
  simp only [← c122661, ← c122662, k2, ← c122663] at e_2011_3
  have hr := e_2011_3
  simp only [e_2011_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15197)) := by
  have c122664 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.1
  have c122665 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.1
  have c122666 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122667 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122668 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122669 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_4
  simp only [← c122664, ← c122665, ← c122666] at e_2011_4
  have e_2011_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_5
  simp only [← c122667, ← c122668, k2, ← c122669] at e_2011_5
  have hr := e_2011_5
  simp only [e_2011_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15198)) := by
  have c122670 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122671 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122672 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122673 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122674 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122675 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_6
  simp only [← c122670, ← c122671, ← c122672] at e_2011_6
  have e_2011_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_7
  simp only [← c122673, ← c122674, k2, ← c122675] at e_2011_7
  have hr := e_2011_7
  simp only [e_2011_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15199)) := by
  have c122676 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122677 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122678 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122679 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122680 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122681 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_8
  simp only [← c122676, ← c122677, ← c122678] at e_2011_8
  have e_2011_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_9
  simp only [← c122679, ← c122680, k2, ← c122681] at e_2011_9
  have hr := e_2011_9
  simp only [e_2011_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15200)) := by
  have c122682 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122683 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122684 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122685 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122686 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122687 := (publicBatchWrapper89_copies3833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_10
  simp only [← c122682, ← c122683, ← c122684] at e_2011_10
  have e_2011_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_11
  simp only [← c122685, ← c122686, k2, ← c122687] at e_2011_11
  have hr := e_2011_11
  simp only [e_2011_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15201)) := by
  have c122688 := (publicBatchWrapper89_copies3834 a h).1
  have c122689 := (publicBatchWrapper89_copies3834 a h).2.1
  have c122690 := (publicBatchWrapper89_copies3834 a h).2.2.1
  have c122691 := (publicBatchWrapper89_copies3834 a h).2.2.2.1
  have c122692 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.1
  have c122693 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_12
  simp only [← c122688, ← c122689, ← c122690] at e_2011_12
  have e_2011_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_13
  simp only [← c122691, ← c122692, k2, ← c122693] at e_2011_13
  have hr := e_2011_13
  simp only [e_2011_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15370)) := by
  have c122694 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.1
  have c122695 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.1
  have c122696 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.1
  have c122697 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.1
  have c122698 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122699 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_14
  simp only [← c122694, ← c122695, ← c122696] at e_2011_14
  have e_2011_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_15
  simp only [← c122697, ← c122698, k2, ← c122699] at e_2011_15
  have hr := e_2011_15
  simp only [e_2011_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15371)) := by
  have c122700 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122701 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122702 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122703 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122704 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122705 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_16
  simp only [← c122700, ← c122701, ← c122702] at e_2011_16
  have e_2011_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_17
  simp only [← c122703, ← c122704, k2, ← c122705] at e_2011_17
  have hr := e_2011_17
  simp only [e_2011_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2011 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15372)) := by
  have c122706 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122707 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122708 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122709 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122710 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122711 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2011_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_18
  simp only [← c122706, ← c122707, ← c122708] at e_2011_18
  have e_2011_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2011 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2011_19
  simp only [← c122709, ← c122710, k2, ← c122711] at e_2011_19
  have hr := e_2011_19
  simp only [e_2011_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15373)) := by
  have c122712 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122713 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122714 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122715 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122716 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122717 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_0
  simp only [← c122712, ← c122713, ← c122714] at e_2012_0
  have e_2012_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_1
  simp only [← c122715, ← c122716, k2, ← c122717] at e_2012_1
  have hr := e_2012_1
  simp only [e_2012_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15374)) := by
  have c122718 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122719 := (publicBatchWrapper89_copies3834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122720 := (publicBatchWrapper89_copies3835 a h).1
  have c122721 := (publicBatchWrapper89_copies3835 a h).2.1
  have c122722 := (publicBatchWrapper89_copies3835 a h).2.2.1
  have c122723 := (publicBatchWrapper89_copies3835 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_2
  simp only [← c122718, ← c122719, ← c122720] at e_2012_2
  have e_2012_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_3
  simp only [← c122721, ← c122722, k2, ← c122723] at e_2012_3
  have hr := e_2012_3
  simp only [e_2012_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15375)) := by
  have c122724 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.1
  have c122725 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.1
  have c122726 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.1
  have c122727 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.1
  have c122728 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.1
  have c122729 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_4
  simp only [← c122724, ← c122725, ← c122726] at e_2012_4
  have e_2012_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_5
  simp only [← c122727, ← c122728, k2, ← c122729] at e_2012_5
  have hr := e_2012_5
  simp only [e_2012_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15376)) := by
  have c122730 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122731 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122732 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122733 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122734 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122735 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_6
  simp only [← c122730, ← c122731, ← c122732] at e_2012_6
  have e_2012_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_7
  simp only [← c122733, ← c122734, k2, ← c122735] at e_2012_7
  have hr := e_2012_7
  simp only [e_2012_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15377)) := by
  have c122736 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122737 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122738 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122739 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122740 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122741 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_8
  simp only [← c122736, ← c122737, ← c122738] at e_2012_8
  have e_2012_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_9
  simp only [← c122739, ← c122740, k2, ← c122741] at e_2012_9
  have hr := e_2012_9
  simp only [e_2012_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15378)) := by
  have c122742 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122743 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122744 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122745 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122746 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122747 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_10
  simp only [← c122742, ← c122743, ← c122744] at e_2012_10
  have e_2012_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_11
  simp only [← c122745, ← c122746, k2, ← c122747] at e_2012_11
  have hr := e_2012_11
  simp only [e_2012_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15379)) := by
  have c122748 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122749 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122750 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122751 := (publicBatchWrapper89_copies3835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122752 := (publicBatchWrapper89_copies3836 a h).1
  have c122753 := (publicBatchWrapper89_copies3836 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_12
  simp only [← c122748, ← c122749, ← c122750] at e_2012_12
  have e_2012_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_13
  simp only [← c122751, ← c122752, k2, ← c122753] at e_2012_13
  have hr := e_2012_13
  simp only [e_2012_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15380)) := by
  have c122754 := (publicBatchWrapper89_copies3836 a h).2.2.1
  have c122755 := (publicBatchWrapper89_copies3836 a h).2.2.2.1
  have c122756 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.1
  have c122757 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.1
  have c122758 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.1
  have c122759 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_14
  simp only [← c122754, ← c122755, ← c122756] at e_2012_14
  have e_2012_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_15
  simp only [← c122757, ← c122758, k2, ← c122759] at e_2012_15
  have hr := e_2012_15
  simp only [e_2012_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15381)) := by
  have c122760 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.1
  have c122761 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.1
  have c122762 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122763 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122764 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122765 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_16
  simp only [← c122760, ← c122761, ← c122762] at e_2012_16
  have e_2012_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_17
  simp only [← c122763, ← c122764, k2, ← c122765] at e_2012_17
  have hr := e_2012_17
  simp only [e_2012_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2012 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15382)) := by
  have c122766 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122767 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122768 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122769 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122770 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122771 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2012_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_18
  simp only [← c122766, ← c122767, ← c122768] at e_2012_18
  have e_2012_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2012 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2012_19
  simp only [← c122769, ← c122770, k2, ← c122771] at e_2012_19
  have hr := e_2012_19
  simp only [e_2012_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15383)) := by
  have c122772 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122773 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122774 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122775 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122776 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122777 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_0
  simp only [← c122772, ← c122773, ← c122774] at e_2013_0
  have e_2013_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_1
  simp only [← c122775, ← c122776, k2, ← c122777] at e_2013_1
  have hr := e_2013_1
  simp only [e_2013_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15384)) := by
  have c122778 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122779 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122780 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122781 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122782 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122783 := (publicBatchWrapper89_copies3836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_2
  simp only [← c122778, ← c122779, ← c122780] at e_2013_2
  have e_2013_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_3
  simp only [← c122781, ← c122782, k2, ← c122783] at e_2013_3
  have hr := e_2013_3
  simp only [e_2013_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15385)) := by
  have c122784 := (publicBatchWrapper89_copies3837 a h).1
  have c122785 := (publicBatchWrapper89_copies3837 a h).2.1
  have c122786 := (publicBatchWrapper89_copies3837 a h).2.2.1
  have c122787 := (publicBatchWrapper89_copies3837 a h).2.2.2.1
  have c122788 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.1
  have c122789 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_4
  simp only [← c122784, ← c122785, ← c122786] at e_2013_4
  have e_2013_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_5
  simp only [← c122787, ← c122788, k2, ← c122789] at e_2013_5
  have hr := e_2013_5
  simp only [e_2013_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15386)) := by
  have c122790 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.1
  have c122791 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.1
  have c122792 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.1
  have c122793 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.1
  have c122794 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122795 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_6
  simp only [← c122790, ← c122791, ← c122792] at e_2013_6
  have e_2013_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_7
  simp only [← c122793, ← c122794, k2, ← c122795] at e_2013_7
  have hr := e_2013_7
  simp only [e_2013_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15387)) := by
  have c122796 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122797 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122798 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122799 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122800 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122801 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_8
  simp only [← c122796, ← c122797, ← c122798] at e_2013_8
  have e_2013_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_9
  simp only [← c122799, ← c122800, k2, ← c122801] at e_2013_9
  have hr := e_2013_9
  simp only [e_2013_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15388)) := by
  have c122802 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122803 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122804 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122805 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122806 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122807 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_10
  simp only [← c122802, ← c122803, ← c122804] at e_2013_10
  have e_2013_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_11
  simp only [← c122805, ← c122806, k2, ← c122807] at e_2013_11
  have hr := e_2013_11
  simp only [e_2013_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15389)) := by
  have c122808 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122809 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122810 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122811 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122812 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122813 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_12
  simp only [← c122808, ← c122809, ← c122810] at e_2013_12
  have e_2013_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_13
  simp only [← c122811, ← c122812, k2, ← c122813] at e_2013_13
  have hr := e_2013_13
  simp only [e_2013_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15390)) := by
  have c122814 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122815 := (publicBatchWrapper89_copies3837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122816 := (publicBatchWrapper89_copies3838 a h).1
  have c122817 := (publicBatchWrapper89_copies3838 a h).2.1
  have c122818 := (publicBatchWrapper89_copies3838 a h).2.2.1
  have c122819 := (publicBatchWrapper89_copies3838 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_14
  simp only [← c122814, ← c122815, ← c122816] at e_2013_14
  have e_2013_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_15
  simp only [← c122817, ← c122818, k2, ← c122819] at e_2013_15
  have hr := e_2013_15
  simp only [e_2013_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15391)) := by
  have c122820 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.1
  have c122821 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.1
  have c122822 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.1
  have c122823 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.1
  have c122824 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.1
  have c122825 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_16
  simp only [← c122820, ← c122821, ← c122822] at e_2013_16
  have e_2013_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_17
  simp only [← c122823, ← c122824, k2, ← c122825] at e_2013_17
  have hr := e_2013_17
  simp only [e_2013_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2013 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15392)) := by
  have c122826 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122827 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122828 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122829 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122830 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122831 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2013_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_18
  simp only [← c122826, ← c122827, ← c122828] at e_2013_18
  have e_2013_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2013 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2013_19
  simp only [← c122829, ← c122830, k2, ← c122831] at e_2013_19
  have hr := e_2013_19
  simp only [e_2013_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15393)) := by
  have c122832 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122833 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122834 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122835 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122836 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122837 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_0
  simp only [← c122832, ← c122833, ← c122834] at e_2014_0
  have e_2014_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_1
  simp only [← c122835, ← c122836, k2, ← c122837] at e_2014_1
  have hr := e_2014_1
  simp only [e_2014_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15394)) := by
  have c122838 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122839 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122840 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122841 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122842 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122843 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_2
  simp only [← c122838, ← c122839, ← c122840] at e_2014_2
  have e_2014_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_3
  simp only [← c122841, ← c122842, k2, ← c122843] at e_2014_3
  have hr := e_2014_3
  simp only [e_2014_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15395)) := by
  have c122844 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122845 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122846 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122847 := (publicBatchWrapper89_copies3838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122848 := (publicBatchWrapper89_copies3839 a h).1
  have c122849 := (publicBatchWrapper89_copies3839 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_4
  simp only [← c122844, ← c122845, ← c122846] at e_2014_4
  have e_2014_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_5
  simp only [← c122847, ← c122848, k2, ← c122849] at e_2014_5
  have hr := e_2014_5
  simp only [e_2014_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15396)) := by
  have c122850 := (publicBatchWrapper89_copies3839 a h).2.2.1
  have c122851 := (publicBatchWrapper89_copies3839 a h).2.2.2.1
  have c122852 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.1
  have c122853 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.1
  have c122854 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.1
  have c122855 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_6
  simp only [← c122850, ← c122851, ← c122852] at e_2014_6
  have e_2014_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_7
  simp only [← c122853, ← c122854, k2, ← c122855] at e_2014_7
  have hr := e_2014_7
  simp only [e_2014_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15397)) := by
  have c122856 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.1
  have c122857 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.1
  have c122858 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122859 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122860 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122861 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_8
  simp only [← c122856, ← c122857, ← c122858] at e_2014_8
  have e_2014_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_9
  simp only [← c122859, ← c122860, k2, ← c122861] at e_2014_9
  have hr := e_2014_9
  simp only [e_2014_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15398)) := by
  have c122862 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122863 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122864 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122865 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122866 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122867 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_10
  simp only [← c122862, ← c122863, ← c122864] at e_2014_10
  have e_2014_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_11
  simp only [← c122865, ← c122866, k2, ← c122867] at e_2014_11
  have hr := e_2014_11
  simp only [e_2014_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15399)) := by
  have c122868 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122869 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122870 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122871 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122872 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122873 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_12
  simp only [← c122868, ← c122869, ← c122870] at e_2014_12
  have e_2014_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_13
  simp only [← c122871, ← c122872, k2, ← c122873] at e_2014_13
  have hr := e_2014_13
  simp only [e_2014_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15400)) := by
  have c122874 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122875 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122876 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122877 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122878 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122879 := (publicBatchWrapper89_copies3839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_14
  simp only [← c122874, ← c122875, ← c122876] at e_2014_14
  have e_2014_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_15
  simp only [← c122877, ← c122878, k2, ← c122879] at e_2014_15
  have hr := e_2014_15
  simp only [e_2014_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15401)) := by
  have c122880 := (publicBatchWrapper89_copies3840 a h).1
  have c122881 := (publicBatchWrapper89_copies3840 a h).2.1
  have c122882 := (publicBatchWrapper89_copies3840 a h).2.2.1
  have c122883 := (publicBatchWrapper89_copies3840 a h).2.2.2.1
  have c122884 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.1
  have c122885 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_16
  simp only [← c122880, ← c122881, ← c122882] at e_2014_16
  have e_2014_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_17
  simp only [← c122883, ← c122884, k2, ← c122885] at e_2014_17
  have hr := e_2014_17
  simp only [e_2014_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2014 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15570)) := by
  have c122886 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.1
  have c122887 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.1
  have c122888 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.1
  have c122889 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.1
  have c122890 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122891 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2014_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_18
  simp only [← c122886, ← c122887, ← c122888] at e_2014_18
  have e_2014_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2014 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2014_19
  simp only [← c122889, ← c122890, k2, ← c122891] at e_2014_19
  have hr := e_2014_19
  simp only [e_2014_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15571)) := by
  have c122892 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122893 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122894 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122895 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122896 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122897 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_0
  simp only [← c122892, ← c122893, ← c122894] at e_2015_0
  have e_2015_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_1
  simp only [← c122895, ← c122896, k2, ← c122897] at e_2015_1
  have hr := e_2015_1
  simp only [e_2015_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15572)) := by
  have c122898 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122899 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122900 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122901 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122902 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122903 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_2
  simp only [← c122898, ← c122899, ← c122900] at e_2015_2
  have e_2015_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_3
  simp only [← c122901, ← c122902, k2, ← c122903] at e_2015_3
  have hr := e_2015_3
  simp only [e_2015_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15573)) := by
  have c122904 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122905 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122906 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122907 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122908 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122909 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_4
  simp only [← c122904, ← c122905, ← c122906] at e_2015_4
  have e_2015_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_5
  simp only [← c122907, ← c122908, k2, ← c122909] at e_2015_5
  have hr := e_2015_5
  simp only [e_2015_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15574)) := by
  have c122910 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122911 := (publicBatchWrapper89_copies3840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122912 := (publicBatchWrapper89_copies3841 a h).1
  have c122913 := (publicBatchWrapper89_copies3841 a h).2.1
  have c122914 := (publicBatchWrapper89_copies3841 a h).2.2.1
  have c122915 := (publicBatchWrapper89_copies3841 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_6
  simp only [← c122910, ← c122911, ← c122912] at e_2015_6
  have e_2015_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_7
  simp only [← c122913, ← c122914, k2, ← c122915] at e_2015_7
  have hr := e_2015_7
  simp only [e_2015_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15575)) := by
  have c122916 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.1
  have c122917 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.1
  have c122918 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.1
  have c122919 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.1
  have c122920 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.1
  have c122921 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_8
  simp only [← c122916, ← c122917, ← c122918] at e_2015_8
  have e_2015_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_9
  simp only [← c122919, ← c122920, k2, ← c122921] at e_2015_9
  have hr := e_2015_9
  simp only [e_2015_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15576)) := by
  have c122922 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122923 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122924 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122925 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122926 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122927 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_10
  simp only [← c122922, ← c122923, ← c122924] at e_2015_10
  have e_2015_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_11
  simp only [← c122925, ← c122926, k2, ← c122927] at e_2015_11
  have hr := e_2015_11
  simp only [e_2015_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15577)) := by
  have c122928 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122929 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122930 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122931 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122932 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122933 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_12
  simp only [← c122928, ← c122929, ← c122930] at e_2015_12
  have e_2015_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_13
  simp only [← c122931, ← c122932, k2, ← c122933] at e_2015_13
  have hr := e_2015_13
  simp only [e_2015_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15578)) := by
  have c122934 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122935 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122936 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122937 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122938 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122939 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_14
  simp only [← c122934, ← c122935, ← c122936] at e_2015_14
  have e_2015_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_15
  simp only [← c122937, ← c122938, k2, ← c122939] at e_2015_15
  have hr := e_2015_15
  simp only [e_2015_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15579)) := by
  have c122940 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122941 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122942 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122943 := (publicBatchWrapper89_copies3841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c122944 := (publicBatchWrapper89_copies3842 a h).1
  have c122945 := (publicBatchWrapper89_copies3842 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_16
  simp only [← c122940, ← c122941, ← c122942] at e_2015_16
  have e_2015_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_17
  simp only [← c122943, ← c122944, k2, ← c122945] at e_2015_17
  have hr := e_2015_17
  simp only [e_2015_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2015 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15580)) := by
  have c122946 := (publicBatchWrapper89_copies3842 a h).2.2.1
  have c122947 := (publicBatchWrapper89_copies3842 a h).2.2.2.1
  have c122948 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.1
  have c122949 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.1
  have c122950 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.1
  have c122951 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2015_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_18
  simp only [← c122946, ← c122947, ← c122948] at e_2015_18
  have e_2015_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2015 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2015_19
  simp only [← c122949, ← c122950, k2, ← c122951] at e_2015_19
  have hr := e_2015_19
  simp only [e_2015_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15581)) := by
  have c122952 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.1
  have c122953 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.1
  have c122954 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122955 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c122956 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122957 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_0
  simp only [← c122952, ← c122953, ← c122954] at e_2016_0
  have e_2016_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_1
  simp only [← c122955, ← c122956, k2, ← c122957] at e_2016_1
  have hr := e_2016_1
  simp only [e_2016_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15582)) := by
  have c122958 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122959 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122960 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122961 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122962 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122963 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_2
  simp only [← c122958, ← c122959, ← c122960] at e_2016_2
  have e_2016_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_3
  simp only [← c122961, ← c122962, k2, ← c122963] at e_2016_3
  have hr := e_2016_3
  simp only [e_2016_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15583)) := by
  have c122964 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122965 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122966 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122967 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122968 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122969 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_4
  simp only [← c122964, ← c122965, ← c122966] at e_2016_4
  have e_2016_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_5
  simp only [← c122967, ← c122968, k2, ← c122969] at e_2016_5
  have hr := e_2016_5
  simp only [e_2016_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15584)) := by
  have c122970 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122971 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122972 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122973 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122974 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122975 := (publicBatchWrapper89_copies3842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_6
  simp only [← c122970, ← c122971, ← c122972] at e_2016_6
  have e_2016_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_7
  simp only [← c122973, ← c122974, k2, ← c122975] at e_2016_7
  have hr := e_2016_7
  simp only [e_2016_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15585)) := by
  have c122976 := (publicBatchWrapper89_copies3843 a h).1
  have c122977 := (publicBatchWrapper89_copies3843 a h).2.1
  have c122978 := (publicBatchWrapper89_copies3843 a h).2.2.1
  have c122979 := (publicBatchWrapper89_copies3843 a h).2.2.2.1
  have c122980 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.1
  have c122981 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_8
  simp only [← c122976, ← c122977, ← c122978] at e_2016_8
  have e_2016_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_9
  simp only [← c122979, ← c122980, k2, ← c122981] at e_2016_9
  have hr := e_2016_9
  simp only [e_2016_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15586)) := by
  have c122982 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.1
  have c122983 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.1
  have c122984 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.1
  have c122985 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.1
  have c122986 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.1
  have c122987 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_10
  simp only [← c122982, ← c122983, ← c122984] at e_2016_10
  have e_2016_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_11
  simp only [← c122985, ← c122986, k2, ← c122987] at e_2016_11
  have hr := e_2016_11
  simp only [e_2016_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15587)) := by
  have c122988 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122989 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122990 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122991 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122992 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122993 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_12
  simp only [← c122988, ← c122989, ← c122990] at e_2016_12
  have e_2016_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_13
  simp only [← c122991, ← c122992, k2, ← c122993] at e_2016_13
  have hr := e_2016_13
  simp only [e_2016_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15588)) := by
  have c122994 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122995 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122996 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122997 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122998 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c122999 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_14
  simp only [← c122994, ← c122995, ← c122996] at e_2016_14
  have e_2016_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_15
  simp only [← c122997, ← c122998, k2, ← c122999] at e_2016_15
  have hr := e_2016_15
  simp only [e_2016_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15589)) := by
  have c123000 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123001 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123002 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123003 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123004 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123005 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_16
  simp only [← c123000, ← c123001, ← c123002] at e_2016_16
  have e_2016_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_17
  simp only [← c123003, ← c123004, k2, ← c123005] at e_2016_17
  have hr := e_2016_17
  simp only [e_2016_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2016 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15590)) := by
  have c123006 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123007 := (publicBatchWrapper89_copies3843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c123008 := (publicBatchWrapper89_copies3844 a h).1
  have c123009 := (publicBatchWrapper89_copies3844 a h).2.1
  have c123010 := (publicBatchWrapper89_copies3844 a h).2.2.1
  have c123011 := (publicBatchWrapper89_copies3844 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2016_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_18
  simp only [← c123006, ← c123007, ← c123008] at e_2016_18
  have e_2016_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row2016 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2016_19
  simp only [← c123009, ← c123010, k2, ← c123011] at e_2016_19
  have hr := e_2016_19
  simp only [e_2016_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15591)) := by
  have c123012 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.1
  have c123013 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.1
  have c123014 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.1
  have c123015 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.1
  have c123016 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.1
  have c123017 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_0
  simp only [← c123012, ← c123013, ← c123014] at e_2017_0
  have e_2017_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_1
  simp only [← c123015, ← c123016, k2, ← c123017] at e_2017_1
  have hr := e_2017_1
  simp only [e_2017_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2017 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15592)) := by
  have c123018 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.1
  have c123019 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c123020 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123021 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123022 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c123023 := (publicBatchWrapper89_copies3844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2017_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_2
  simp only [← c123018, ← c123019, ← c123020] at e_2017_2
  have e_2017_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row2017 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2017_3
  simp only [← c123021, ← c123022, k2, ← c123023] at e_2017_3
  have hr := e_2017_3
  simp only [e_2017_2] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
