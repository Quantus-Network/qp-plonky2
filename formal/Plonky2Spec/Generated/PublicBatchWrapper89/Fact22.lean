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

theorem publicBatchWrapper89_f16991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1741 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17455)) := by
  have c106452 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106453 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106454 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106455 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106456 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106457 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1741_0 := arithEq_of_rows h (row := 1741) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_0
  simp only [← c106452, ← c106453, ← c106454] at e_1741_0
  have e_1741_1 := arithEq_of_rows h (row := 1741) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_1
  simp only [← c106455, ← c106456, k2, ← c106457] at e_1741_1
  have hr := e_1741_1
  simp only [e_1741_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1741 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17456)) := by
  have c106458 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106459 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106460 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106461 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106462 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106463 := (publicBatchWrapper89_copies3326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1741_2 := arithEq_of_rows h (row := 1741) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_2
  simp only [← c106458, ← c106459, ← c106460] at e_1741_2
  have e_1741_3 := arithEq_of_rows h (row := 1741) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_3
  simp only [← c106461, ← c106462, k2, ← c106463] at e_1741_3
  have hr := e_1741_3
  simp only [e_1741_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1741 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17457)) := by
  have c106464 := (publicBatchWrapper89_copies3327 a h).1
  have c106465 := (publicBatchWrapper89_copies3327 a h).2.1
  have c106466 := (publicBatchWrapper89_copies3327 a h).2.2.1
  have c106467 := (publicBatchWrapper89_copies3327 a h).2.2.2.1
  have c106468 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.1
  have c106469 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1741_4 := arithEq_of_rows h (row := 1741) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_4
  simp only [← c106464, ← c106465, ← c106466] at e_1741_4
  have e_1741_5 := arithEq_of_rows h (row := 1741) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_5
  simp only [← c106467, ← c106468, k2, ← c106469] at e_1741_5
  have hr := e_1741_5
  simp only [e_1741_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1741 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17458)) := by
  have c106470 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.1
  have c106471 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.1
  have c106472 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.1
  have c106473 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.1
  have c106474 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106475 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1741_6 := arithEq_of_rows h (row := 1741) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_6
  simp only [← c106470, ← c106471, ← c106472] at e_1741_6
  have e_1741_7 := arithEq_of_rows h (row := 1741) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_7
  simp only [← c106473, ← c106474, k2, ← c106475] at e_1741_7
  have hr := e_1741_7
  simp only [e_1741_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1741 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17459)) := by
  have c106476 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106477 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106478 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106479 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106480 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106481 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1741_8 := arithEq_of_rows h (row := 1741) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_8
  simp only [← c106476, ← c106477, ← c106478] at e_1741_8
  have e_1741_9 := arithEq_of_rows h (row := 1741) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_9
  simp only [← c106479, ← c106480, k2, ← c106481] at e_1741_9
  have hr := e_1741_9
  simp only [e_1741_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1741 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17460)) := by
  have c106482 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106483 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106484 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106485 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106486 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106487 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1741_10 := arithEq_of_rows h (row := 1741) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_10
  simp only [← c106482, ← c106483, ← c106484] at e_1741_10
  have e_1741_11 := arithEq_of_rows h (row := 1741) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_11
  simp only [← c106485, ← c106486, k2, ← c106487] at e_1741_11
  have hr := e_1741_11
  simp only [e_1741_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1741 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17461)) := by
  have c106488 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106489 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106490 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106491 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106492 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106493 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1741_12 := arithEq_of_rows h (row := 1741) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_12
  simp only [← c106488, ← c106489, ← c106490] at e_1741_12
  have e_1741_13 := arithEq_of_rows h (row := 1741) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_13
  simp only [← c106491, ← c106492, k2, ← c106493] at e_1741_13
  have hr := e_1741_13
  simp only [e_1741_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1741 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17462)) := by
  have c106494 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106495 := (publicBatchWrapper89_copies3327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106496 := (publicBatchWrapper89_copies3328 a h).1
  have c106497 := (publicBatchWrapper89_copies3328 a h).2.1
  have c106498 := (publicBatchWrapper89_copies3328 a h).2.2.1
  have c106499 := (publicBatchWrapper89_copies3328 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1741_14 := arithEq_of_rows h (row := 1741) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_14
  simp only [← c106494, ← c106495, ← c106496] at e_1741_14
  have e_1741_15 := arithEq_of_rows h (row := 1741) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_15
  simp only [← c106497, ← c106498, k2, ← c106499] at e_1741_15
  have hr := e_1741_15
  simp only [e_1741_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f16999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1741 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17463)) := by
  have c106500 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.1
  have c106501 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.1
  have c106502 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.1
  have c106503 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.1
  have c106504 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.1
  have c106505 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1741_16 := arithEq_of_rows h (row := 1741) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_16
  simp only [← c106500, ← c106501, ← c106502] at e_1741_16
  have e_1741_17 := arithEq_of_rows h (row := 1741) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_17
  simp only [← c106503, ← c106504, k2, ← c106505] at e_1741_17
  have hr := e_1741_17
  simp only [e_1741_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1741 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17464)) := by
  have c106506 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106507 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106508 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106509 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106510 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106511 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1741_18 := arithEq_of_rows h (row := 1741) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_18
  simp only [← c106506, ← c106507, ← c106508] at e_1741_18
  have e_1741_19 := arithEq_of_rows h (row := 1741) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_19
  simp only [← c106509, ← c106510, k2, ← c106511] at e_1741_19
  have hr := e_1741_19
  simp only [e_1741_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1742 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17465)) := by
  have c106512 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106513 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106514 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106515 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106516 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106517 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1742_0 := arithEq_of_rows h (row := 1742) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_0
  simp only [← c106512, ← c106513, ← c106514] at e_1742_0
  have e_1742_1 := arithEq_of_rows h (row := 1742) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_1
  simp only [← c106515, ← c106516, k2, ← c106517] at e_1742_1
  have hr := e_1742_1
  simp only [e_1742_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1742 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17466)) := by
  have c106518 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106519 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106520 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106521 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106522 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106523 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1742_2 := arithEq_of_rows h (row := 1742) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_2
  simp only [← c106518, ← c106519, ← c106520] at e_1742_2
  have e_1742_3 := arithEq_of_rows h (row := 1742) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_3
  simp only [← c106521, ← c106522, k2, ← c106523] at e_1742_3
  have hr := e_1742_3
  simp only [e_1742_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1742 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17467)) := by
  have c106524 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106525 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106526 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106527 := (publicBatchWrapper89_copies3328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106528 := (publicBatchWrapper89_copies3329 a h).1
  have c106529 := (publicBatchWrapper89_copies3329 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1742_4 := arithEq_of_rows h (row := 1742) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_4
  simp only [← c106524, ← c106525, ← c106526] at e_1742_4
  have e_1742_5 := arithEq_of_rows h (row := 1742) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_5
  simp only [← c106527, ← c106528, k2, ← c106529] at e_1742_5
  have hr := e_1742_5
  simp only [e_1742_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1742 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17468)) := by
  have c106530 := (publicBatchWrapper89_copies3329 a h).2.2.1
  have c106531 := (publicBatchWrapper89_copies3329 a h).2.2.2.1
  have c106532 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.1
  have c106533 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.1
  have c106534 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.1
  have c106535 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1742_6 := arithEq_of_rows h (row := 1742) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_6
  simp only [← c106530, ← c106531, ← c106532] at e_1742_6
  have e_1742_7 := arithEq_of_rows h (row := 1742) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_7
  simp only [← c106533, ← c106534, k2, ← c106535] at e_1742_7
  have hr := e_1742_7
  simp only [e_1742_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1742 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17469)) := by
  have c106536 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.1
  have c106537 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.1
  have c106538 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106539 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106540 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106541 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1742_8 := arithEq_of_rows h (row := 1742) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_8
  simp only [← c106536, ← c106537, ← c106538] at e_1742_8
  have e_1742_9 := arithEq_of_rows h (row := 1742) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_9
  simp only [← c106539, ← c106540, k2, ← c106541] at e_1742_9
  have hr := e_1742_9
  simp only [e_1742_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1742 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17470)) := by
  have c106542 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106543 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106544 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106545 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106546 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106547 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1742_10 := arithEq_of_rows h (row := 1742) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_10
  simp only [← c106542, ← c106543, ← c106544] at e_1742_10
  have e_1742_11 := arithEq_of_rows h (row := 1742) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_11
  simp only [← c106545, ← c106546, k2, ← c106547] at e_1742_11
  have hr := e_1742_11
  simp only [e_1742_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1742 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17471)) := by
  have c106548 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106549 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106550 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106551 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106552 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106553 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1742_12 := arithEq_of_rows h (row := 1742) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_12
  simp only [← c106548, ← c106549, ← c106550] at e_1742_12
  have e_1742_13 := arithEq_of_rows h (row := 1742) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_13
  simp only [← c106551, ← c106552, k2, ← c106553] at e_1742_13
  have hr := e_1742_13
  simp only [e_1742_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1742 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17472)) := by
  have c106554 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106555 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106556 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106557 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106558 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106559 := (publicBatchWrapper89_copies3329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1742_14 := arithEq_of_rows h (row := 1742) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_14
  simp only [← c106554, ← c106555, ← c106556] at e_1742_14
  have e_1742_15 := arithEq_of_rows h (row := 1742) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_15
  simp only [← c106557, ← c106558, k2, ← c106559] at e_1742_15
  have hr := e_1742_15
  simp only [e_1742_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1742 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17473)) := by
  have c106560 := (publicBatchWrapper89_copies3330 a h).1
  have c106561 := (publicBatchWrapper89_copies3330 a h).2.1
  have c106562 := (publicBatchWrapper89_copies3330 a h).2.2.1
  have c106563 := (publicBatchWrapper89_copies3330 a h).2.2.2.1
  have c106564 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.1
  have c106565 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1742_16 := arithEq_of_rows h (row := 1742) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_16
  simp only [← c106560, ← c106561, ← c106562] at e_1742_16
  have e_1742_17 := arithEq_of_rows h (row := 1742) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_17
  simp only [← c106563, ← c106564, k2, ← c106565] at e_1742_17
  have hr := e_1742_17
  simp only [e_1742_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1742 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17474)) := by
  have c106566 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.1
  have c106567 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.1
  have c106568 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.1
  have c106569 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.1
  have c106570 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106571 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1742_18 := arithEq_of_rows h (row := 1742) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_18
  simp only [← c106566, ← c106567, ← c106568] at e_1742_18
  have e_1742_19 := arithEq_of_rows h (row := 1742) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1742_19
  simp only [← c106569, ← c106570, k2, ← c106571] at e_1742_19
  have hr := e_1742_19
  simp only [e_1742_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1743 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17475)) := by
  have c106572 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106573 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106574 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106575 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106576 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106577 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1743_0 := arithEq_of_rows h (row := 1743) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_0
  simp only [← c106572, ← c106573, ← c106574] at e_1743_0
  have e_1743_1 := arithEq_of_rows h (row := 1743) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_1
  simp only [← c106575, ← c106576, k2, ← c106577] at e_1743_1
  have hr := e_1743_1
  simp only [e_1743_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1743 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17476)) := by
  have c106578 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106579 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106580 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106581 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106582 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106583 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1743_2 := arithEq_of_rows h (row := 1743) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_2
  simp only [← c106578, ← c106579, ← c106580] at e_1743_2
  have e_1743_3 := arithEq_of_rows h (row := 1743) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_3
  simp only [← c106581, ← c106582, k2, ← c106583] at e_1743_3
  have hr := e_1743_3
  simp only [e_1743_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1743 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17477)) := by
  have c106584 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106585 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106586 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106587 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106588 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106589 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1743_4 := arithEq_of_rows h (row := 1743) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_4
  simp only [← c106584, ← c106585, ← c106586] at e_1743_4
  have e_1743_5 := arithEq_of_rows h (row := 1743) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_5
  simp only [← c106587, ← c106588, k2, ← c106589] at e_1743_5
  have hr := e_1743_5
  simp only [e_1743_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1743 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17478)) := by
  have c106590 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106591 := (publicBatchWrapper89_copies3330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106592 := (publicBatchWrapper89_copies3331 a h).1
  have c106593 := (publicBatchWrapper89_copies3331 a h).2.1
  have c106594 := (publicBatchWrapper89_copies3331 a h).2.2.1
  have c106595 := (publicBatchWrapper89_copies3331 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1743_6 := arithEq_of_rows h (row := 1743) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_6
  simp only [← c106590, ← c106591, ← c106592] at e_1743_6
  have e_1743_7 := arithEq_of_rows h (row := 1743) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_7
  simp only [← c106593, ← c106594, k2, ← c106595] at e_1743_7
  have hr := e_1743_7
  simp only [e_1743_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1743 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17479)) := by
  have c106596 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.1
  have c106597 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.1
  have c106598 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.1
  have c106599 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.1
  have c106600 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.1
  have c106601 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1743_8 := arithEq_of_rows h (row := 1743) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_8
  simp only [← c106596, ← c106597, ← c106598] at e_1743_8
  have e_1743_9 := arithEq_of_rows h (row := 1743) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_9
  simp only [← c106599, ← c106600, k2, ← c106601] at e_1743_9
  have hr := e_1743_9
  simp only [e_1743_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1743 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17480)) := by
  have c106602 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106603 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106604 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106605 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106606 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106607 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1743_10 := arithEq_of_rows h (row := 1743) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_10
  simp only [← c106602, ← c106603, ← c106604] at e_1743_10
  have e_1743_11 := arithEq_of_rows h (row := 1743) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_11
  simp only [← c106605, ← c106606, k2, ← c106607] at e_1743_11
  have hr := e_1743_11
  simp only [e_1743_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1743 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17481)) := by
  have c106608 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106609 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106610 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106611 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106612 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106613 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1743_12 := arithEq_of_rows h (row := 1743) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_12
  simp only [← c106608, ← c106609, ← c106610] at e_1743_12
  have e_1743_13 := arithEq_of_rows h (row := 1743) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_13
  simp only [← c106611, ← c106612, k2, ← c106613] at e_1743_13
  have hr := e_1743_13
  simp only [e_1743_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1743 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17482)) := by
  have c106614 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106615 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106616 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106617 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106618 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106619 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1743_14 := arithEq_of_rows h (row := 1743) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_14
  simp only [← c106614, ← c106615, ← c106616] at e_1743_14
  have e_1743_15 := arithEq_of_rows h (row := 1743) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_15
  simp only [← c106617, ← c106618, k2, ← c106619] at e_1743_15
  have hr := e_1743_15
  simp only [e_1743_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1743 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17483)) := by
  have c106620 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106621 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106622 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106623 := (publicBatchWrapper89_copies3331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106624 := (publicBatchWrapper89_copies3332 a h).1
  have c106625 := (publicBatchWrapper89_copies3332 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1743_16 := arithEq_of_rows h (row := 1743) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_16
  simp only [← c106620, ← c106621, ← c106622] at e_1743_16
  have e_1743_17 := arithEq_of_rows h (row := 1743) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_17
  simp only [← c106623, ← c106624, k2, ← c106625] at e_1743_17
  have hr := e_1743_17
  simp only [e_1743_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1743 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17484)) := by
  have c106626 := (publicBatchWrapper89_copies3332 a h).2.2.1
  have c106627 := (publicBatchWrapper89_copies3332 a h).2.2.2.1
  have c106628 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.1
  have c106629 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.1
  have c106630 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.1
  have c106631 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1743_18 := arithEq_of_rows h (row := 1743) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_18
  simp only [← c106626, ← c106627, ← c106628] at e_1743_18
  have e_1743_19 := arithEq_of_rows h (row := 1743) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1743_19
  simp only [← c106629, ← c106630, k2, ← c106631] at e_1743_19
  have hr := e_1743_19
  simp only [e_1743_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1744 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17485)) := by
  have c106632 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.1
  have c106633 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.1
  have c106634 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106635 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106636 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106637 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1744_0 := arithEq_of_rows h (row := 1744) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_0
  simp only [← c106632, ← c106633, ← c106634] at e_1744_0
  have e_1744_1 := arithEq_of_rows h (row := 1744) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_1
  simp only [← c106635, ← c106636, k2, ← c106637] at e_1744_1
  have hr := e_1744_1
  simp only [e_1744_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1744 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17486)) := by
  have c106638 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106639 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106640 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106641 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106642 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106643 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1744_2 := arithEq_of_rows h (row := 1744) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_2
  simp only [← c106638, ← c106639, ← c106640] at e_1744_2
  have e_1744_3 := arithEq_of_rows h (row := 1744) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_3
  simp only [← c106641, ← c106642, k2, ← c106643] at e_1744_3
  have hr := e_1744_3
  simp only [e_1744_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1744 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17487)) := by
  have c106644 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106645 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106646 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106647 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106648 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106649 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1744_4 := arithEq_of_rows h (row := 1744) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_4
  simp only [← c106644, ← c106645, ← c106646] at e_1744_4
  have e_1744_5 := arithEq_of_rows h (row := 1744) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_5
  simp only [← c106647, ← c106648, k2, ← c106649] at e_1744_5
  have hr := e_1744_5
  simp only [e_1744_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1744 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17488)) := by
  have c106650 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106651 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106652 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106653 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106654 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106655 := (publicBatchWrapper89_copies3332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1744_6 := arithEq_of_rows h (row := 1744) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_6
  simp only [← c106650, ← c106651, ← c106652] at e_1744_6
  have e_1744_7 := arithEq_of_rows h (row := 1744) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_7
  simp only [← c106653, ← c106654, k2, ← c106655] at e_1744_7
  have hr := e_1744_7
  simp only [e_1744_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1744 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17489)) := by
  have c106656 := (publicBatchWrapper89_copies3333 a h).1
  have c106657 := (publicBatchWrapper89_copies3333 a h).2.1
  have c106658 := (publicBatchWrapper89_copies3333 a h).2.2.1
  have c106659 := (publicBatchWrapper89_copies3333 a h).2.2.2.1
  have c106660 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.1
  have c106661 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1744_8 := arithEq_of_rows h (row := 1744) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_8
  simp only [← c106656, ← c106657, ← c106658] at e_1744_8
  have e_1744_9 := arithEq_of_rows h (row := 1744) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_9
  simp only [← c106659, ← c106660, k2, ← c106661] at e_1744_9
  have hr := e_1744_9
  simp only [e_1744_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1744 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17490)) := by
  have c106662 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.1
  have c106663 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.1
  have c106664 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.1
  have c106665 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.1
  have c106666 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106667 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1744_10 := arithEq_of_rows h (row := 1744) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_10
  simp only [← c106662, ← c106663, ← c106664] at e_1744_10
  have e_1744_11 := arithEq_of_rows h (row := 1744) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_11
  simp only [← c106665, ← c106666, k2, ← c106667] at e_1744_11
  have hr := e_1744_11
  simp only [e_1744_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1744 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17491)) := by
  have c106668 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106669 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106670 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106671 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106672 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106673 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1744_12 := arithEq_of_rows h (row := 1744) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_12
  simp only [← c106668, ← c106669, ← c106670] at e_1744_12
  have e_1744_13 := arithEq_of_rows h (row := 1744) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_13
  simp only [← c106671, ← c106672, k2, ← c106673] at e_1744_13
  have hr := e_1744_13
  simp only [e_1744_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1744 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17492)) := by
  have c106674 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106675 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106676 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106677 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106678 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106679 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1744_14 := arithEq_of_rows h (row := 1744) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_14
  simp only [← c106674, ← c106675, ← c106676] at e_1744_14
  have e_1744_15 := arithEq_of_rows h (row := 1744) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_15
  simp only [← c106677, ← c106678, k2, ← c106679] at e_1744_15
  have hr := e_1744_15
  simp only [e_1744_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1744 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17493)) := by
  have c106680 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106681 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106682 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106683 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106684 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106685 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1744_16 := arithEq_of_rows h (row := 1744) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_16
  simp only [← c106680, ← c106681, ← c106682] at e_1744_16
  have e_1744_17 := arithEq_of_rows h (row := 1744) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_17
  simp only [← c106683, ← c106684, k2, ← c106685] at e_1744_17
  have hr := e_1744_17
  simp only [e_1744_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1744 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17494)) := by
  have c106686 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106687 := (publicBatchWrapper89_copies3333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106688 := (publicBatchWrapper89_copies3334 a h).1
  have c106689 := (publicBatchWrapper89_copies3334 a h).2.1
  have c106690 := (publicBatchWrapper89_copies3334 a h).2.2.1
  have c106691 := (publicBatchWrapper89_copies3334 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1744_18 := arithEq_of_rows h (row := 1744) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_18
  simp only [← c106686, ← c106687, ← c106688] at e_1744_18
  have e_1744_19 := arithEq_of_rows h (row := 1744) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1744_19
  simp only [← c106689, ← c106690, k2, ← c106691] at e_1744_19
  have hr := e_1744_19
  simp only [e_1744_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1745 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17495)) := by
  have c106692 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.1
  have c106693 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.1
  have c106694 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.1
  have c106695 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.1
  have c106696 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.1
  have c106697 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1745_0 := arithEq_of_rows h (row := 1745) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_0
  simp only [← c106692, ← c106693, ← c106694] at e_1745_0
  have e_1745_1 := arithEq_of_rows h (row := 1745) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_1
  simp only [← c106695, ← c106696, k2, ← c106697] at e_1745_1
  have hr := e_1745_1
  simp only [e_1745_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1745 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17496)) := by
  have c106698 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106699 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106700 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106701 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106702 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106703 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1745_2 := arithEq_of_rows h (row := 1745) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_2
  simp only [← c106698, ← c106699, ← c106700] at e_1745_2
  have e_1745_3 := arithEq_of_rows h (row := 1745) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_3
  simp only [← c106701, ← c106702, k2, ← c106703] at e_1745_3
  have hr := e_1745_3
  simp only [e_1745_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1745 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17497)) := by
  have c106704 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106705 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106706 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106707 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106708 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106709 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1745_4 := arithEq_of_rows h (row := 1745) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_4
  simp only [← c106704, ← c106705, ← c106706] at e_1745_4
  have e_1745_5 := arithEq_of_rows h (row := 1745) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_5
  simp only [← c106707, ← c106708, k2, ← c106709] at e_1745_5
  have hr := e_1745_5
  simp only [e_1745_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1745 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17498)) := by
  have c106710 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106711 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106712 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106713 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106714 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106715 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1745_6 := arithEq_of_rows h (row := 1745) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_6
  simp only [← c106710, ← c106711, ← c106712] at e_1745_6
  have e_1745_7 := arithEq_of_rows h (row := 1745) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_7
  simp only [← c106713, ← c106714, k2, ← c106715] at e_1745_7
  have hr := e_1745_7
  simp only [e_1745_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1745 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17499)) := by
  have c106716 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106717 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106718 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106719 := (publicBatchWrapper89_copies3334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106720 := (publicBatchWrapper89_copies3335 a h).1
  have c106721 := (publicBatchWrapper89_copies3335 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1745_8 := arithEq_of_rows h (row := 1745) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_8
  simp only [← c106716, ← c106717, ← c106718] at e_1745_8
  have e_1745_9 := arithEq_of_rows h (row := 1745) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_9
  simp only [← c106719, ← c106720, k2, ← c106721] at e_1745_9
  have hr := e_1745_9
  simp only [e_1745_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1745 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17500)) := by
  have c106722 := (publicBatchWrapper89_copies3335 a h).2.2.1
  have c106723 := (publicBatchWrapper89_copies3335 a h).2.2.2.1
  have c106724 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.1
  have c106725 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.1
  have c106726 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.1
  have c106727 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1745_10 := arithEq_of_rows h (row := 1745) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_10
  simp only [← c106722, ← c106723, ← c106724] at e_1745_10
  have e_1745_11 := arithEq_of_rows h (row := 1745) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_11
  simp only [← c106725, ← c106726, k2, ← c106727] at e_1745_11
  have hr := e_1745_11
  simp only [e_1745_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1745 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17501)) := by
  have c106728 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.1
  have c106729 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.1
  have c106730 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106731 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106732 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106733 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1745_12 := arithEq_of_rows h (row := 1745) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_12
  simp only [← c106728, ← c106729, ← c106730] at e_1745_12
  have e_1745_13 := arithEq_of_rows h (row := 1745) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_13
  simp only [← c106731, ← c106732, k2, ← c106733] at e_1745_13
  have hr := e_1745_13
  simp only [e_1745_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1745 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17502)) := by
  have c106734 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106735 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106736 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106737 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106738 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106739 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1745_14 := arithEq_of_rows h (row := 1745) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_14
  simp only [← c106734, ← c106735, ← c106736] at e_1745_14
  have e_1745_15 := arithEq_of_rows h (row := 1745) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_15
  simp only [← c106737, ← c106738, k2, ← c106739] at e_1745_15
  have hr := e_1745_15
  simp only [e_1745_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1745 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17503)) := by
  have c106740 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106741 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106742 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106743 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106744 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106745 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1745_16 := arithEq_of_rows h (row := 1745) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_16
  simp only [← c106740, ← c106741, ← c106742] at e_1745_16
  have e_1745_17 := arithEq_of_rows h (row := 1745) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_17
  simp only [← c106743, ← c106744, k2, ← c106745] at e_1745_17
  have hr := e_1745_17
  simp only [e_1745_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1745 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17504)) := by
  have c106746 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106747 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106748 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106749 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106750 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106751 := (publicBatchWrapper89_copies3335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1745_18 := arithEq_of_rows h (row := 1745) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_18
  simp only [← c106746, ← c106747, ← c106748] at e_1745_18
  have e_1745_19 := arithEq_of_rows h (row := 1745) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1745_19
  simp only [← c106749, ← c106750, k2, ← c106751] at e_1745_19
  have hr := e_1745_19
  simp only [e_1745_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1746 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17505)) := by
  have c106752 := (publicBatchWrapper89_copies3336 a h).1
  have c106753 := (publicBatchWrapper89_copies3336 a h).2.1
  have c106754 := (publicBatchWrapper89_copies3336 a h).2.2.1
  have c106755 := (publicBatchWrapper89_copies3336 a h).2.2.2.1
  have c106756 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.1
  have c106757 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1746_0 := arithEq_of_rows h (row := 1746) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_0
  simp only [← c106752, ← c106753, ← c106754] at e_1746_0
  have e_1746_1 := arithEq_of_rows h (row := 1746) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_1
  simp only [← c106755, ← c106756, k2, ← c106757] at e_1746_1
  have hr := e_1746_1
  simp only [e_1746_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1746 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17506)) := by
  have c106758 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.1
  have c106759 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.1
  have c106760 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.1
  have c106761 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.1
  have c106762 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106763 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1746_2 := arithEq_of_rows h (row := 1746) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_2
  simp only [← c106758, ← c106759, ← c106760] at e_1746_2
  have e_1746_3 := arithEq_of_rows h (row := 1746) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_3
  simp only [← c106761, ← c106762, k2, ← c106763] at e_1746_3
  have hr := e_1746_3
  simp only [e_1746_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1746 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17507)) := by
  have c106764 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106765 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106766 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106767 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106768 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106769 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1746_4 := arithEq_of_rows h (row := 1746) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_4
  simp only [← c106764, ← c106765, ← c106766] at e_1746_4
  have e_1746_5 := arithEq_of_rows h (row := 1746) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_5
  simp only [← c106767, ← c106768, k2, ← c106769] at e_1746_5
  have hr := e_1746_5
  simp only [e_1746_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1746 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17508)) := by
  have c106770 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106771 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106772 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106773 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106774 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106775 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1746_6 := arithEq_of_rows h (row := 1746) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_6
  simp only [← c106770, ← c106771, ← c106772] at e_1746_6
  have e_1746_7 := arithEq_of_rows h (row := 1746) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_7
  simp only [← c106773, ← c106774, k2, ← c106775] at e_1746_7
  have hr := e_1746_7
  simp only [e_1746_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1746 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17509)) := by
  have c106776 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106777 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106778 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106779 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106780 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106781 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1746_8 := arithEq_of_rows h (row := 1746) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_8
  simp only [← c106776, ← c106777, ← c106778] at e_1746_8
  have e_1746_9 := arithEq_of_rows h (row := 1746) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_9
  simp only [← c106779, ← c106780, k2, ← c106781] at e_1746_9
  have hr := e_1746_9
  simp only [e_1746_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1746 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17510)) := by
  have c106782 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106783 := (publicBatchWrapper89_copies3336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106784 := (publicBatchWrapper89_copies3337 a h).1
  have c106785 := (publicBatchWrapper89_copies3337 a h).2.1
  have c106786 := (publicBatchWrapper89_copies3337 a h).2.2.1
  have c106787 := (publicBatchWrapper89_copies3337 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1746_10 := arithEq_of_rows h (row := 1746) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_10
  simp only [← c106782, ← c106783, ← c106784] at e_1746_10
  have e_1746_11 := arithEq_of_rows h (row := 1746) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_11
  simp only [← c106785, ← c106786, k2, ← c106787] at e_1746_11
  have hr := e_1746_11
  simp only [e_1746_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1746 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17511)) := by
  have c106788 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.1
  have c106789 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.1
  have c106790 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.1
  have c106791 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.1
  have c106792 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.1
  have c106793 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1746_12 := arithEq_of_rows h (row := 1746) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_12
  simp only [← c106788, ← c106789, ← c106790] at e_1746_12
  have e_1746_13 := arithEq_of_rows h (row := 1746) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_13
  simp only [← c106791, ← c106792, k2, ← c106793] at e_1746_13
  have hr := e_1746_13
  simp only [e_1746_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1746 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17512)) := by
  have c106794 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106795 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106796 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106797 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106798 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106799 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1746_14 := arithEq_of_rows h (row := 1746) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_14
  simp only [← c106794, ← c106795, ← c106796] at e_1746_14
  have e_1746_15 := arithEq_of_rows h (row := 1746) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_15
  simp only [← c106797, ← c106798, k2, ← c106799] at e_1746_15
  have hr := e_1746_15
  simp only [e_1746_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1746 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17513)) := by
  have c106800 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106801 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106802 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106803 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106804 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106805 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1746_16 := arithEq_of_rows h (row := 1746) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_16
  simp only [← c106800, ← c106801, ← c106802] at e_1746_16
  have e_1746_17 := arithEq_of_rows h (row := 1746) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_17
  simp only [← c106803, ← c106804, k2, ← c106805] at e_1746_17
  have hr := e_1746_17
  simp only [e_1746_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1746 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17514)) := by
  have c106806 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106807 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106808 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106809 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106810 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106811 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1746_18 := arithEq_of_rows h (row := 1746) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_18
  simp only [← c106806, ← c106807, ← c106808] at e_1746_18
  have e_1746_19 := arithEq_of_rows h (row := 1746) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1746_19
  simp only [← c106809, ← c106810, k2, ← c106811] at e_1746_19
  have hr := e_1746_19
  simp only [e_1746_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1747 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17515)) := by
  have c106812 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106813 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106814 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106815 := (publicBatchWrapper89_copies3337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106816 := (publicBatchWrapper89_copies3338 a h).1
  have c106817 := (publicBatchWrapper89_copies3338 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1747_0 := arithEq_of_rows h (row := 1747) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_0
  simp only [← c106812, ← c106813, ← c106814] at e_1747_0
  have e_1747_1 := arithEq_of_rows h (row := 1747) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_1
  simp only [← c106815, ← c106816, k2, ← c106817] at e_1747_1
  have hr := e_1747_1
  simp only [e_1747_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1747 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17516)) := by
  have c106818 := (publicBatchWrapper89_copies3338 a h).2.2.1
  have c106819 := (publicBatchWrapper89_copies3338 a h).2.2.2.1
  have c106820 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.1
  have c106821 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.1
  have c106822 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.1
  have c106823 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1747_2 := arithEq_of_rows h (row := 1747) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_2
  simp only [← c106818, ← c106819, ← c106820] at e_1747_2
  have e_1747_3 := arithEq_of_rows h (row := 1747) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_3
  simp only [← c106821, ← c106822, k2, ← c106823] at e_1747_3
  have hr := e_1747_3
  simp only [e_1747_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1747 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17517)) := by
  have c106824 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.1
  have c106825 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.1
  have c106826 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106827 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106828 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106829 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1747_4 := arithEq_of_rows h (row := 1747) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_4
  simp only [← c106824, ← c106825, ← c106826] at e_1747_4
  have e_1747_5 := arithEq_of_rows h (row := 1747) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_5
  simp only [← c106827, ← c106828, k2, ← c106829] at e_1747_5
  have hr := e_1747_5
  simp only [e_1747_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1747 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17518)) := by
  have c106830 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106831 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106832 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106833 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106834 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106835 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1747_6 := arithEq_of_rows h (row := 1747) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_6
  simp only [← c106830, ← c106831, ← c106832] at e_1747_6
  have e_1747_7 := arithEq_of_rows h (row := 1747) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_7
  simp only [← c106833, ← c106834, k2, ← c106835] at e_1747_7
  have hr := e_1747_7
  simp only [e_1747_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1747 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17519)) := by
  have c106836 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106837 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106838 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106839 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106840 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106841 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1747_8 := arithEq_of_rows h (row := 1747) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_8
  simp only [← c106836, ← c106837, ← c106838] at e_1747_8
  have e_1747_9 := arithEq_of_rows h (row := 1747) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_9
  simp only [← c106839, ← c106840, k2, ← c106841] at e_1747_9
  have hr := e_1747_9
  simp only [e_1747_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1747 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17520)) := by
  have c106842 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106843 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106844 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106845 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106846 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106847 := (publicBatchWrapper89_copies3338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1747_10 := arithEq_of_rows h (row := 1747) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_10
  simp only [← c106842, ← c106843, ← c106844] at e_1747_10
  have e_1747_11 := arithEq_of_rows h (row := 1747) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_11
  simp only [← c106845, ← c106846, k2, ← c106847] at e_1747_11
  have hr := e_1747_11
  simp only [e_1747_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1747 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17521)) := by
  have c106848 := (publicBatchWrapper89_copies3339 a h).1
  have c106849 := (publicBatchWrapper89_copies3339 a h).2.1
  have c106850 := (publicBatchWrapper89_copies3339 a h).2.2.1
  have c106851 := (publicBatchWrapper89_copies3339 a h).2.2.2.1
  have c106852 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.1
  have c106853 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1747_12 := arithEq_of_rows h (row := 1747) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_12
  simp only [← c106848, ← c106849, ← c106850] at e_1747_12
  have e_1747_13 := arithEq_of_rows h (row := 1747) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_13
  simp only [← c106851, ← c106852, k2, ← c106853] at e_1747_13
  have hr := e_1747_13
  simp only [e_1747_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1747 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17522)) := by
  have c106854 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.1
  have c106855 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.1
  have c106856 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.1
  have c106857 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.1
  have c106858 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106859 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1747_14 := arithEq_of_rows h (row := 1747) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_14
  simp only [← c106854, ← c106855, ← c106856] at e_1747_14
  have e_1747_15 := arithEq_of_rows h (row := 1747) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_15
  simp only [← c106857, ← c106858, k2, ← c106859] at e_1747_15
  have hr := e_1747_15
  simp only [e_1747_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1747 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17523)) := by
  have c106860 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106861 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106862 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106863 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106864 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106865 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1747_16 := arithEq_of_rows h (row := 1747) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_16
  simp only [← c106860, ← c106861, ← c106862] at e_1747_16
  have e_1747_17 := arithEq_of_rows h (row := 1747) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_17
  simp only [← c106863, ← c106864, k2, ← c106865] at e_1747_17
  have hr := e_1747_17
  simp only [e_1747_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1747 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17524)) := by
  have c106866 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106867 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106868 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106869 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106870 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106871 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1747_18 := arithEq_of_rows h (row := 1747) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_18
  simp only [← c106866, ← c106867, ← c106868] at e_1747_18
  have e_1747_19 := arithEq_of_rows h (row := 1747) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1747_19
  simp only [← c106869, ← c106870, k2, ← c106871] at e_1747_19
  have hr := e_1747_19
  simp only [e_1747_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1748 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17525)) := by
  have c106872 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106873 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106874 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106875 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106876 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106877 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1748_0 := arithEq_of_rows h (row := 1748) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_0
  simp only [← c106872, ← c106873, ← c106874] at e_1748_0
  have e_1748_1 := arithEq_of_rows h (row := 1748) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_1
  simp only [← c106875, ← c106876, k2, ← c106877] at e_1748_1
  have hr := e_1748_1
  simp only [e_1748_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1748 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17526)) := by
  have c106878 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106879 := (publicBatchWrapper89_copies3339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106880 := (publicBatchWrapper89_copies3340 a h).1
  have c106881 := (publicBatchWrapper89_copies3340 a h).2.1
  have c106882 := (publicBatchWrapper89_copies3340 a h).2.2.1
  have c106883 := (publicBatchWrapper89_copies3340 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1748_2 := arithEq_of_rows h (row := 1748) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_2
  simp only [← c106878, ← c106879, ← c106880] at e_1748_2
  have e_1748_3 := arithEq_of_rows h (row := 1748) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_3
  simp only [← c106881, ← c106882, k2, ← c106883] at e_1748_3
  have hr := e_1748_3
  simp only [e_1748_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1748 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17527)) := by
  have c106884 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.1
  have c106885 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.1
  have c106886 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.1
  have c106887 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.1
  have c106888 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.1
  have c106889 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1748_4 := arithEq_of_rows h (row := 1748) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_4
  simp only [← c106884, ← c106885, ← c106886] at e_1748_4
  have e_1748_5 := arithEq_of_rows h (row := 1748) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_5
  simp only [← c106887, ← c106888, k2, ← c106889] at e_1748_5
  have hr := e_1748_5
  simp only [e_1748_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1748 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17528)) := by
  have c106890 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106891 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106892 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106893 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106894 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106895 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1748_6 := arithEq_of_rows h (row := 1748) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_6
  simp only [← c106890, ← c106891, ← c106892] at e_1748_6
  have e_1748_7 := arithEq_of_rows h (row := 1748) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_7
  simp only [← c106893, ← c106894, k2, ← c106895] at e_1748_7
  have hr := e_1748_7
  simp only [e_1748_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1748 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17529)) := by
  have c106896 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106897 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106898 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106899 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106900 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106901 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1748_8 := arithEq_of_rows h (row := 1748) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_8
  simp only [← c106896, ← c106897, ← c106898] at e_1748_8
  have e_1748_9 := arithEq_of_rows h (row := 1748) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_9
  simp only [← c106899, ← c106900, k2, ← c106901] at e_1748_9
  have hr := e_1748_9
  simp only [e_1748_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1748 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17530)) := by
  have c106902 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106903 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106904 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106905 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106906 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106907 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1748_10 := arithEq_of_rows h (row := 1748) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_10
  simp only [← c106902, ← c106903, ← c106904] at e_1748_10
  have e_1748_11 := arithEq_of_rows h (row := 1748) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_11
  simp only [← c106905, ← c106906, k2, ← c106907] at e_1748_11
  have hr := e_1748_11
  simp only [e_1748_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1748 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17531)) := by
  have c106908 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106909 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106910 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106911 := (publicBatchWrapper89_copies3340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106912 := (publicBatchWrapper89_copies3341 a h).1
  have c106913 := (publicBatchWrapper89_copies3341 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1748_12 := arithEq_of_rows h (row := 1748) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_12
  simp only [← c106908, ← c106909, ← c106910] at e_1748_12
  have e_1748_13 := arithEq_of_rows h (row := 1748) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_13
  simp only [← c106911, ← c106912, k2, ← c106913] at e_1748_13
  have hr := e_1748_13
  simp only [e_1748_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1748 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17532)) := by
  have c106914 := (publicBatchWrapper89_copies3341 a h).2.2.1
  have c106915 := (publicBatchWrapper89_copies3341 a h).2.2.2.1
  have c106916 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.1
  have c106917 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.1
  have c106918 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.1
  have c106919 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1748_14 := arithEq_of_rows h (row := 1748) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_14
  simp only [← c106914, ← c106915, ← c106916] at e_1748_14
  have e_1748_15 := arithEq_of_rows h (row := 1748) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_15
  simp only [← c106917, ← c106918, k2, ← c106919] at e_1748_15
  have hr := e_1748_15
  simp only [e_1748_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1748 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17533)) := by
  have c106920 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.1
  have c106921 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.1
  have c106922 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106923 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106924 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106925 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1748_16 := arithEq_of_rows h (row := 1748) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_16
  simp only [← c106920, ← c106921, ← c106922] at e_1748_16
  have e_1748_17 := arithEq_of_rows h (row := 1748) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_17
  simp only [← c106923, ← c106924, k2, ← c106925] at e_1748_17
  have hr := e_1748_17
  simp only [e_1748_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1748 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17534)) := by
  have c106926 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106927 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106928 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106929 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106930 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106931 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1748_18 := arithEq_of_rows h (row := 1748) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_18
  simp only [← c106926, ← c106927, ← c106928] at e_1748_18
  have e_1748_19 := arithEq_of_rows h (row := 1748) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1748_19
  simp only [← c106929, ← c106930, k2, ← c106931] at e_1748_19
  have hr := e_1748_19
  simp only [e_1748_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1749 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17535)) := by
  have c106932 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106933 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106934 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106935 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106936 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106937 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1749_0 := arithEq_of_rows h (row := 1749) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_0
  simp only [← c106932, ← c106933, ← c106934] at e_1749_0
  have e_1749_1 := arithEq_of_rows h (row := 1749) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_1
  simp only [← c106935, ← c106936, k2, ← c106937] at e_1749_1
  have hr := e_1749_1
  simp only [e_1749_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1749 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17536)) := by
  have c106938 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106939 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106940 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106941 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106942 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106943 := (publicBatchWrapper89_copies3341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1749_2 := arithEq_of_rows h (row := 1749) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_2
  simp only [← c106938, ← c106939, ← c106940] at e_1749_2
  have e_1749_3 := arithEq_of_rows h (row := 1749) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_3
  simp only [← c106941, ← c106942, k2, ← c106943] at e_1749_3
  have hr := e_1749_3
  simp only [e_1749_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1749 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17537)) := by
  have c106944 := (publicBatchWrapper89_copies3342 a h).1
  have c106945 := (publicBatchWrapper89_copies3342 a h).2.1
  have c106946 := (publicBatchWrapper89_copies3342 a h).2.2.1
  have c106947 := (publicBatchWrapper89_copies3342 a h).2.2.2.1
  have c106948 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.1
  have c106949 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1749_4 := arithEq_of_rows h (row := 1749) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_4
  simp only [← c106944, ← c106945, ← c106946] at e_1749_4
  have e_1749_5 := arithEq_of_rows h (row := 1749) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_5
  simp only [← c106947, ← c106948, k2, ← c106949] at e_1749_5
  have hr := e_1749_5
  simp only [e_1749_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1749 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17538)) := by
  have c106950 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.1
  have c106951 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.1
  have c106952 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.1
  have c106953 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.1
  have c106954 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106955 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1749_6 := arithEq_of_rows h (row := 1749) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_6
  simp only [← c106950, ← c106951, ← c106952] at e_1749_6
  have e_1749_7 := arithEq_of_rows h (row := 1749) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_7
  simp only [← c106953, ← c106954, k2, ← c106955] at e_1749_7
  have hr := e_1749_7
  simp only [e_1749_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1749 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17539)) := by
  have c106956 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106957 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106958 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106959 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106960 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106961 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1749_8 := arithEq_of_rows h (row := 1749) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_8
  simp only [← c106956, ← c106957, ← c106958] at e_1749_8
  have e_1749_9 := arithEq_of_rows h (row := 1749) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_9
  simp only [← c106959, ← c106960, k2, ← c106961] at e_1749_9
  have hr := e_1749_9
  simp only [e_1749_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1749 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17540)) := by
  have c106962 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106963 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106964 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106965 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106966 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106967 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1749_10 := arithEq_of_rows h (row := 1749) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_10
  simp only [← c106962, ← c106963, ← c106964] at e_1749_10
  have e_1749_11 := arithEq_of_rows h (row := 1749) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_11
  simp only [← c106965, ← c106966, k2, ← c106967] at e_1749_11
  have hr := e_1749_11
  simp only [e_1749_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1749 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17541)) := by
  have c106968 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106969 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106970 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106971 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106972 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106973 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1749_12 := arithEq_of_rows h (row := 1749) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_12
  simp only [← c106968, ← c106969, ← c106970] at e_1749_12
  have e_1749_13 := arithEq_of_rows h (row := 1749) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_13
  simp only [← c106971, ← c106972, k2, ← c106973] at e_1749_13
  have hr := e_1749_13
  simp only [e_1749_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1749 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17542)) := by
  have c106974 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106975 := (publicBatchWrapper89_copies3342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106976 := (publicBatchWrapper89_copies3343 a h).1
  have c106977 := (publicBatchWrapper89_copies3343 a h).2.1
  have c106978 := (publicBatchWrapper89_copies3343 a h).2.2.1
  have c106979 := (publicBatchWrapper89_copies3343 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1749_14 := arithEq_of_rows h (row := 1749) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_14
  simp only [← c106974, ← c106975, ← c106976] at e_1749_14
  have e_1749_15 := arithEq_of_rows h (row := 1749) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_15
  simp only [← c106977, ← c106978, k2, ← c106979] at e_1749_15
  have hr := e_1749_15
  simp only [e_1749_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1749 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17543)) := by
  have c106980 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.1
  have c106981 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.1
  have c106982 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.1
  have c106983 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.1
  have c106984 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.1
  have c106985 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1749_16 := arithEq_of_rows h (row := 1749) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_16
  simp only [← c106980, ← c106981, ← c106982] at e_1749_16
  have e_1749_17 := arithEq_of_rows h (row := 1749) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_17
  simp only [← c106983, ← c106984, k2, ← c106985] at e_1749_17
  have hr := e_1749_17
  simp only [e_1749_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1749 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17544)) := by
  have c106986 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106987 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106988 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106989 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106990 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106991 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1749_18 := arithEq_of_rows h (row := 1749) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_18
  simp only [← c106986, ← c106987, ← c106988] at e_1749_18
  have e_1749_19 := arithEq_of_rows h (row := 1749) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1749_19
  simp only [← c106989, ← c106990, k2, ← c106991] at e_1749_19
  have hr := e_1749_19
  simp only [e_1749_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1750 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17545)) := by
  have c106992 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106993 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106994 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106995 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106996 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106997 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1750_0 := arithEq_of_rows h (row := 1750) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_0
  simp only [← c106992, ← c106993, ← c106994] at e_1750_0
  have e_1750_1 := arithEq_of_rows h (row := 1750) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_1
  simp only [← c106995, ← c106996, k2, ← c106997] at e_1750_1
  have hr := e_1750_1
  simp only [e_1750_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1750 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17546)) := by
  have c106998 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106999 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107000 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107001 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107002 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107003 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1750_2 := arithEq_of_rows h (row := 1750) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_2
  simp only [← c106998, ← c106999, ← c107000] at e_1750_2
  have e_1750_3 := arithEq_of_rows h (row := 1750) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_3
  simp only [← c107001, ← c107002, k2, ← c107003] at e_1750_3
  have hr := e_1750_3
  simp only [e_1750_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1750 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17547)) := by
  have c107004 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107005 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107006 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107007 := (publicBatchWrapper89_copies3343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107008 := (publicBatchWrapper89_copies3344 a h).1
  have c107009 := (publicBatchWrapper89_copies3344 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1750_4 := arithEq_of_rows h (row := 1750) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_4
  simp only [← c107004, ← c107005, ← c107006] at e_1750_4
  have e_1750_5 := arithEq_of_rows h (row := 1750) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_5
  simp only [← c107007, ← c107008, k2, ← c107009] at e_1750_5
  have hr := e_1750_5
  simp only [e_1750_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1750 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17548)) := by
  have c107010 := (publicBatchWrapper89_copies3344 a h).2.2.1
  have c107011 := (publicBatchWrapper89_copies3344 a h).2.2.2.1
  have c107012 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.1
  have c107013 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.1
  have c107014 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.1
  have c107015 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1750_6 := arithEq_of_rows h (row := 1750) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_6
  simp only [← c107010, ← c107011, ← c107012] at e_1750_6
  have e_1750_7 := arithEq_of_rows h (row := 1750) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_7
  simp only [← c107013, ← c107014, k2, ← c107015] at e_1750_7
  have hr := e_1750_7
  simp only [e_1750_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1750 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17549)) := by
  have c107016 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.1
  have c107017 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.1
  have c107018 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107019 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107020 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107021 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1750_8 := arithEq_of_rows h (row := 1750) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_8
  simp only [← c107016, ← c107017, ← c107018] at e_1750_8
  have e_1750_9 := arithEq_of_rows h (row := 1750) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_9
  simp only [← c107019, ← c107020, k2, ← c107021] at e_1750_9
  have hr := e_1750_9
  simp only [e_1750_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1750 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17550)) := by
  have c107022 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107023 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107024 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107025 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107026 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107027 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1750_10 := arithEq_of_rows h (row := 1750) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_10
  simp only [← c107022, ← c107023, ← c107024] at e_1750_10
  have e_1750_11 := arithEq_of_rows h (row := 1750) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_11
  simp only [← c107025, ← c107026, k2, ← c107027] at e_1750_11
  have hr := e_1750_11
  simp only [e_1750_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1750 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17551)) := by
  have c107028 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107029 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107030 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107031 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107032 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107033 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1750_12 := arithEq_of_rows h (row := 1750) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_12
  simp only [← c107028, ← c107029, ← c107030] at e_1750_12
  have e_1750_13 := arithEq_of_rows h (row := 1750) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_13
  simp only [← c107031, ← c107032, k2, ← c107033] at e_1750_13
  have hr := e_1750_13
  simp only [e_1750_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1750 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17552)) := by
  have c107034 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107035 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107036 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107037 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107038 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107039 := (publicBatchWrapper89_copies3344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1750_14 := arithEq_of_rows h (row := 1750) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_14
  simp only [← c107034, ← c107035, ← c107036] at e_1750_14
  have e_1750_15 := arithEq_of_rows h (row := 1750) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_15
  simp only [← c107037, ← c107038, k2, ← c107039] at e_1750_15
  have hr := e_1750_15
  simp only [e_1750_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1750 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17553)) := by
  have c107040 := (publicBatchWrapper89_copies3345 a h).1
  have c107041 := (publicBatchWrapper89_copies3345 a h).2.1
  have c107042 := (publicBatchWrapper89_copies3345 a h).2.2.1
  have c107043 := (publicBatchWrapper89_copies3345 a h).2.2.2.1
  have c107044 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.1
  have c107045 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1750_16 := arithEq_of_rows h (row := 1750) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_16
  simp only [← c107040, ← c107041, ← c107042] at e_1750_16
  have e_1750_17 := arithEq_of_rows h (row := 1750) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_17
  simp only [← c107043, ← c107044, k2, ← c107045] at e_1750_17
  have hr := e_1750_17
  simp only [e_1750_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1750 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17554)) := by
  have c107046 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.1
  have c107047 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.1
  have c107048 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.1
  have c107049 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.1
  have c107050 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107051 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1750_18 := arithEq_of_rows h (row := 1750) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_18
  simp only [← c107046, ← c107047, ← c107048] at e_1750_18
  have e_1750_19 := arithEq_of_rows h (row := 1750) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1750_19
  simp only [← c107049, ← c107050, k2, ← c107051] at e_1750_19
  have hr := e_1750_19
  simp only [e_1750_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1751 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17555)) := by
  have c107052 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107053 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107054 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107055 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107056 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107057 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1751_0 := arithEq_of_rows h (row := 1751) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_0
  simp only [← c107052, ← c107053, ← c107054] at e_1751_0
  have e_1751_1 := arithEq_of_rows h (row := 1751) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_1
  simp only [← c107055, ← c107056, k2, ← c107057] at e_1751_1
  have hr := e_1751_1
  simp only [e_1751_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1751 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17556)) := by
  have c107058 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107059 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107060 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107061 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107062 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107063 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1751_2 := arithEq_of_rows h (row := 1751) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_2
  simp only [← c107058, ← c107059, ← c107060] at e_1751_2
  have e_1751_3 := arithEq_of_rows h (row := 1751) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_3
  simp only [← c107061, ← c107062, k2, ← c107063] at e_1751_3
  have hr := e_1751_3
  simp only [e_1751_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1751 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17557)) := by
  have c107064 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107065 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107066 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107067 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107068 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107069 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1751_4 := arithEq_of_rows h (row := 1751) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_4
  simp only [← c107064, ← c107065, ← c107066] at e_1751_4
  have e_1751_5 := arithEq_of_rows h (row := 1751) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_5
  simp only [← c107067, ← c107068, k2, ← c107069] at e_1751_5
  have hr := e_1751_5
  simp only [e_1751_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1751 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17558)) := by
  have c107070 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107071 := (publicBatchWrapper89_copies3345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107072 := (publicBatchWrapper89_copies3346 a h).1
  have c107073 := (publicBatchWrapper89_copies3346 a h).2.1
  have c107074 := (publicBatchWrapper89_copies3346 a h).2.2.1
  have c107075 := (publicBatchWrapper89_copies3346 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1751_6 := arithEq_of_rows h (row := 1751) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_6
  simp only [← c107070, ← c107071, ← c107072] at e_1751_6
  have e_1751_7 := arithEq_of_rows h (row := 1751) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_7
  simp only [← c107073, ← c107074, k2, ← c107075] at e_1751_7
  have hr := e_1751_7
  simp only [e_1751_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1751 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17559)) := by
  have c107076 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.1
  have c107077 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.1
  have c107078 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.1
  have c107079 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.1
  have c107080 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.1
  have c107081 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1751_8 := arithEq_of_rows h (row := 1751) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_8
  simp only [← c107076, ← c107077, ← c107078] at e_1751_8
  have e_1751_9 := arithEq_of_rows h (row := 1751) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_9
  simp only [← c107079, ← c107080, k2, ← c107081] at e_1751_9
  have hr := e_1751_9
  simp only [e_1751_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1751 47) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17560)) := by
  have c107082 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107083 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107084 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107085 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107086 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107087 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1751_10 := arithEq_of_rows h (row := 1751) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_10
  simp only [← c107082, ← c107083, ← c107084] at e_1751_10
  have e_1751_11 := arithEq_of_rows h (row := 1751) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_11
  simp only [← c107085, ← c107086, k2, ← c107087] at e_1751_11
  have hr := e_1751_11
  simp only [e_1751_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1751 55) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17561)) := by
  have c107088 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107089 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107090 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107091 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107092 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107093 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1751_12 := arithEq_of_rows h (row := 1751) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_12
  simp only [← c107088, ← c107089, ← c107090] at e_1751_12
  have e_1751_13 := arithEq_of_rows h (row := 1751) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_13
  simp only [← c107091, ← c107092, k2, ← c107093] at e_1751_13
  have hr := e_1751_13
  simp only [e_1751_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1751 63) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17562)) := by
  have c107094 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107095 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107096 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107097 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107098 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107099 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1751_14 := arithEq_of_rows h (row := 1751) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_14
  simp only [← c107094, ← c107095, ← c107096] at e_1751_14
  have e_1751_15 := arithEq_of_rows h (row := 1751) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_15
  simp only [← c107097, ← c107098, k2, ← c107099] at e_1751_15
  have hr := e_1751_15
  simp only [e_1751_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1751 71) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17563)) := by
  have c107100 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107101 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107102 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107103 := (publicBatchWrapper89_copies3346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107104 := (publicBatchWrapper89_copies3347 a h).1
  have c107105 := (publicBatchWrapper89_copies3347 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1751_16 := arithEq_of_rows h (row := 1751) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_16
  simp only [← c107100, ← c107101, ← c107102] at e_1751_16
  have e_1751_17 := arithEq_of_rows h (row := 1751) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_17
  simp only [← c107103, ← c107104, k2, ← c107105] at e_1751_17
  have hr := e_1751_17
  simp only [e_1751_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1751 79) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17564)) := by
  have c107106 := (publicBatchWrapper89_copies3347 a h).2.2.1
  have c107107 := (publicBatchWrapper89_copies3347 a h).2.2.2.1
  have c107108 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.1
  have c107109 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.1
  have c107110 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.1
  have c107111 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1751_18 := arithEq_of_rows h (row := 1751) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_18
  simp only [← c107106, ← c107107, ← c107108] at e_1751_18
  have e_1751_19 := arithEq_of_rows h (row := 1751) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1751_19
  simp only [← c107109, ← c107110, k2, ← c107111] at e_1751_19
  have hr := e_1751_19
  simp only [e_1751_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1752 7) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17565)) := by
  have c107112 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.1
  have c107113 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.1
  have c107114 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107115 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107116 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107117 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1752_0 := arithEq_of_rows h (row := 1752) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_0
  simp only [← c107112, ← c107113, ← c107114] at e_1752_0
  have e_1752_1 := arithEq_of_rows h (row := 1752) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_1
  simp only [← c107115, ← c107116, k2, ← c107117] at e_1752_1
  have hr := e_1752_1
  simp only [e_1752_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1752 15) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17566)) := by
  have c107118 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107119 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107120 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107121 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107122 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107123 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1752_2 := arithEq_of_rows h (row := 1752) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_2
  simp only [← c107118, ← c107119, ← c107120] at e_1752_2
  have e_1752_3 := arithEq_of_rows h (row := 1752) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_3
  simp only [← c107121, ← c107122, k2, ← c107123] at e_1752_3
  have hr := e_1752_3
  simp only [e_1752_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1752 23) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17567)) := by
  have c107124 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107125 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107126 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107127 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107128 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107129 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1752_4 := arithEq_of_rows h (row := 1752) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_4
  simp only [← c107124, ← c107125, ← c107126] at e_1752_4
  have e_1752_5 := arithEq_of_rows h (row := 1752) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_5
  simp only [← c107127, ← c107128, k2, ← c107129] at e_1752_5
  have hr := e_1752_5
  simp only [e_1752_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1752 31) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17568)) := by
  have c107130 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107131 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107132 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107133 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107134 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107135 := (publicBatchWrapper89_copies3347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1752_6 := arithEq_of_rows h (row := 1752) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_6
  simp only [← c107130, ← c107131, ← c107132] at e_1752_6
  have e_1752_7 := arithEq_of_rows h (row := 1752) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_7
  simp only [← c107133, ← c107134, k2, ← c107135] at e_1752_7
  have hr := e_1752_7
  simp only [e_1752_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1752 39) = bselect (a (.wire 84 47)) (a (.virt 200)) (a (.virt 17569)) := by
  have c107136 := (publicBatchWrapper89_copies3348 a h).1
  have c107137 := (publicBatchWrapper89_copies3348 a h).2.1
  have c107138 := (publicBatchWrapper89_copies3348 a h).2.2.1
  have c107139 := (publicBatchWrapper89_copies3348 a h).2.2.2.1
  have c107140 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.1
  have c107141 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1752_8 := arithEq_of_rows h (row := 1752) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_8
  simp only [← c107136, ← c107137, ← c107138] at e_1752_8
  have e_1752_9 := arithEq_of_rows h (row := 1752) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_9
  simp only [← c107139, ← c107140, k2, ← c107141] at e_1752_9
  have hr := e_1752_9
  simp only [e_1752_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1752 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17610)) := by
  have c107142 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.1
  have c107143 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.1
  have c107144 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.1
  have c107145 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.1
  have c107146 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107147 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1752_10 := arithEq_of_rows h (row := 1752) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_10
  simp only [← c107142, ← c107143, ← c107144] at e_1752_10
  have e_1752_11 := arithEq_of_rows h (row := 1752) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_11
  simp only [← c107145, ← c107146, k2, ← c107147] at e_1752_11
  have hr := e_1752_11
  simp only [e_1752_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1752 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17611)) := by
  have c107148 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107149 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107150 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107151 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107152 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107153 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1752_12 := arithEq_of_rows h (row := 1752) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_12
  simp only [← c107148, ← c107149, ← c107150] at e_1752_12
  have e_1752_13 := arithEq_of_rows h (row := 1752) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_13
  simp only [← c107151, ← c107152, k2, ← c107153] at e_1752_13
  have hr := e_1752_13
  simp only [e_1752_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1752 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17612)) := by
  have c107154 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107155 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107156 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107157 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107158 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107159 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1752_14 := arithEq_of_rows h (row := 1752) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_14
  simp only [← c107154, ← c107155, ← c107156] at e_1752_14
  have e_1752_15 := arithEq_of_rows h (row := 1752) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_15
  simp only [← c107157, ← c107158, k2, ← c107159] at e_1752_15
  have hr := e_1752_15
  simp only [e_1752_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1752 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17613)) := by
  have c107160 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107161 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107162 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107163 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107164 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107165 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1752_16 := arithEq_of_rows h (row := 1752) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_16
  simp only [← c107160, ← c107161, ← c107162] at e_1752_16
  have e_1752_17 := arithEq_of_rows h (row := 1752) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_17
  simp only [← c107163, ← c107164, k2, ← c107165] at e_1752_17
  have hr := e_1752_17
  simp only [e_1752_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1752 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17614)) := by
  have c107166 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107167 := (publicBatchWrapper89_copies3348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107168 := (publicBatchWrapper89_copies3349 a h).1
  have c107169 := (publicBatchWrapper89_copies3349 a h).2.1
  have c107170 := (publicBatchWrapper89_copies3349 a h).2.2.1
  have c107171 := (publicBatchWrapper89_copies3349 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1752_18 := arithEq_of_rows h (row := 1752) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_18
  simp only [← c107166, ← c107167, ← c107168] at e_1752_18
  have e_1752_19 := arithEq_of_rows h (row := 1752) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1752_19
  simp only [← c107169, ← c107170, k2, ← c107171] at e_1752_19
  have hr := e_1752_19
  simp only [e_1752_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1753 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17615)) := by
  have c107172 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.1
  have c107173 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.1
  have c107174 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.1
  have c107175 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.1
  have c107176 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.1
  have c107177 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1753_0 := arithEq_of_rows h (row := 1753) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_0
  simp only [← c107172, ← c107173, ← c107174] at e_1753_0
  have e_1753_1 := arithEq_of_rows h (row := 1753) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_1
  simp only [← c107175, ← c107176, k2, ← c107177] at e_1753_1
  have hr := e_1753_1
  simp only [e_1753_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1753 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17616)) := by
  have c107178 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107179 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107180 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107181 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107182 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107183 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1753_2 := arithEq_of_rows h (row := 1753) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_2
  simp only [← c107178, ← c107179, ← c107180] at e_1753_2
  have e_1753_3 := arithEq_of_rows h (row := 1753) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_3
  simp only [← c107181, ← c107182, k2, ← c107183] at e_1753_3
  have hr := e_1753_3
  simp only [e_1753_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1753 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17617)) := by
  have c107184 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107185 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107186 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107187 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107188 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107189 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1753_4 := arithEq_of_rows h (row := 1753) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_4
  simp only [← c107184, ← c107185, ← c107186] at e_1753_4
  have e_1753_5 := arithEq_of_rows h (row := 1753) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_5
  simp only [← c107187, ← c107188, k2, ← c107189] at e_1753_5
  have hr := e_1753_5
  simp only [e_1753_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1753 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17618)) := by
  have c107190 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107191 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107192 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107193 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107194 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107195 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1753_6 := arithEq_of_rows h (row := 1753) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_6
  simp only [← c107190, ← c107191, ← c107192] at e_1753_6
  have e_1753_7 := arithEq_of_rows h (row := 1753) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_7
  simp only [← c107193, ← c107194, k2, ← c107195] at e_1753_7
  have hr := e_1753_7
  simp only [e_1753_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1753 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17619)) := by
  have c107196 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107197 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107198 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107199 := (publicBatchWrapper89_copies3349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107200 := (publicBatchWrapper89_copies3350 a h).1
  have c107201 := (publicBatchWrapper89_copies3350 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1753_8 := arithEq_of_rows h (row := 1753) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_8
  simp only [← c107196, ← c107197, ← c107198] at e_1753_8
  have e_1753_9 := arithEq_of_rows h (row := 1753) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_9
  simp only [← c107199, ← c107200, k2, ← c107201] at e_1753_9
  have hr := e_1753_9
  simp only [e_1753_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1753 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17620)) := by
  have c107202 := (publicBatchWrapper89_copies3350 a h).2.2.1
  have c107203 := (publicBatchWrapper89_copies3350 a h).2.2.2.1
  have c107204 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.1
  have c107205 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.1
  have c107206 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.1
  have c107207 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1753_10 := arithEq_of_rows h (row := 1753) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_10
  simp only [← c107202, ← c107203, ← c107204] at e_1753_10
  have e_1753_11 := arithEq_of_rows h (row := 1753) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_11
  simp only [← c107205, ← c107206, k2, ← c107207] at e_1753_11
  have hr := e_1753_11
  simp only [e_1753_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1753 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17621)) := by
  have c107208 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.1
  have c107209 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.1
  have c107210 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107211 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107212 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107213 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1753_12 := arithEq_of_rows h (row := 1753) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_12
  simp only [← c107208, ← c107209, ← c107210] at e_1753_12
  have e_1753_13 := arithEq_of_rows h (row := 1753) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_13
  simp only [← c107211, ← c107212, k2, ← c107213] at e_1753_13
  have hr := e_1753_13
  simp only [e_1753_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1753 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17622)) := by
  have c107214 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107215 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107216 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107217 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107218 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107219 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1753_14 := arithEq_of_rows h (row := 1753) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_14
  simp only [← c107214, ← c107215, ← c107216] at e_1753_14
  have e_1753_15 := arithEq_of_rows h (row := 1753) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_15
  simp only [← c107217, ← c107218, k2, ← c107219] at e_1753_15
  have hr := e_1753_15
  simp only [e_1753_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1753 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17623)) := by
  have c107220 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107221 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107222 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107223 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107224 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107225 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1753_16 := arithEq_of_rows h (row := 1753) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_16
  simp only [← c107220, ← c107221, ← c107222] at e_1753_16
  have e_1753_17 := arithEq_of_rows h (row := 1753) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_17
  simp only [← c107223, ← c107224, k2, ← c107225] at e_1753_17
  have hr := e_1753_17
  simp only [e_1753_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1753 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17624)) := by
  have c107226 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107227 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107228 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107229 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107230 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107231 := (publicBatchWrapper89_copies3350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1753_18 := arithEq_of_rows h (row := 1753) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_18
  simp only [← c107226, ← c107227, ← c107228] at e_1753_18
  have e_1753_19 := arithEq_of_rows h (row := 1753) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1753_19
  simp only [← c107229, ← c107230, k2, ← c107231] at e_1753_19
  have hr := e_1753_19
  simp only [e_1753_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1754 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17625)) := by
  have c107232 := (publicBatchWrapper89_copies3351 a h).1
  have c107233 := (publicBatchWrapper89_copies3351 a h).2.1
  have c107234 := (publicBatchWrapper89_copies3351 a h).2.2.1
  have c107235 := (publicBatchWrapper89_copies3351 a h).2.2.2.1
  have c107236 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.1
  have c107237 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1754_0 := arithEq_of_rows h (row := 1754) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_0
  simp only [← c107232, ← c107233, ← c107234] at e_1754_0
  have e_1754_1 := arithEq_of_rows h (row := 1754) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_1
  simp only [← c107235, ← c107236, k2, ← c107237] at e_1754_1
  have hr := e_1754_1
  simp only [e_1754_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1754 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17626)) := by
  have c107238 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.1
  have c107239 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.1
  have c107240 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.1
  have c107241 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.1
  have c107242 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107243 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1754_2 := arithEq_of_rows h (row := 1754) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_2
  simp only [← c107238, ← c107239, ← c107240] at e_1754_2
  have e_1754_3 := arithEq_of_rows h (row := 1754) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_3
  simp only [← c107241, ← c107242, k2, ← c107243] at e_1754_3
  have hr := e_1754_3
  simp only [e_1754_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1754 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17627)) := by
  have c107244 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107245 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107246 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107247 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107248 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107249 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1754_4 := arithEq_of_rows h (row := 1754) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_4
  simp only [← c107244, ← c107245, ← c107246] at e_1754_4
  have e_1754_5 := arithEq_of_rows h (row := 1754) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_5
  simp only [← c107247, ← c107248, k2, ← c107249] at e_1754_5
  have hr := e_1754_5
  simp only [e_1754_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1754 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17628)) := by
  have c107250 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107251 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107252 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107253 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107254 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107255 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1754_6 := arithEq_of_rows h (row := 1754) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_6
  simp only [← c107250, ← c107251, ← c107252] at e_1754_6
  have e_1754_7 := arithEq_of_rows h (row := 1754) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_7
  simp only [← c107253, ← c107254, k2, ← c107255] at e_1754_7
  have hr := e_1754_7
  simp only [e_1754_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1754 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17629)) := by
  have c107256 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107257 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107258 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107259 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107260 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107261 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1754_8 := arithEq_of_rows h (row := 1754) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_8
  simp only [← c107256, ← c107257, ← c107258] at e_1754_8
  have e_1754_9 := arithEq_of_rows h (row := 1754) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_9
  simp only [← c107259, ← c107260, k2, ← c107261] at e_1754_9
  have hr := e_1754_9
  simp only [e_1754_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1754 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17630)) := by
  have c107262 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107263 := (publicBatchWrapper89_copies3351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107264 := (publicBatchWrapper89_copies3352 a h).1
  have c107265 := (publicBatchWrapper89_copies3352 a h).2.1
  have c107266 := (publicBatchWrapper89_copies3352 a h).2.2.1
  have c107267 := (publicBatchWrapper89_copies3352 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1754_10 := arithEq_of_rows h (row := 1754) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_10
  simp only [← c107262, ← c107263, ← c107264] at e_1754_10
  have e_1754_11 := arithEq_of_rows h (row := 1754) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_11
  simp only [← c107265, ← c107266, k2, ← c107267] at e_1754_11
  have hr := e_1754_11
  simp only [e_1754_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1754 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17631)) := by
  have c107268 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.1
  have c107269 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.1
  have c107270 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.1
  have c107271 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.1
  have c107272 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.1
  have c107273 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1754_12 := arithEq_of_rows h (row := 1754) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_12
  simp only [← c107268, ← c107269, ← c107270] at e_1754_12
  have e_1754_13 := arithEq_of_rows h (row := 1754) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_13
  simp only [← c107271, ← c107272, k2, ← c107273] at e_1754_13
  have hr := e_1754_13
  simp only [e_1754_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1754 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17632)) := by
  have c107274 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107275 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107276 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107277 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107278 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107279 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1754_14 := arithEq_of_rows h (row := 1754) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_14
  simp only [← c107274, ← c107275, ← c107276] at e_1754_14
  have e_1754_15 := arithEq_of_rows h (row := 1754) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_15
  simp only [← c107277, ← c107278, k2, ← c107279] at e_1754_15
  have hr := e_1754_15
  simp only [e_1754_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1754 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17633)) := by
  have c107280 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107281 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107282 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107283 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107284 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107285 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1754_16 := arithEq_of_rows h (row := 1754) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_16
  simp only [← c107280, ← c107281, ← c107282] at e_1754_16
  have e_1754_17 := arithEq_of_rows h (row := 1754) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_17
  simp only [← c107283, ← c107284, k2, ← c107285] at e_1754_17
  have hr := e_1754_17
  simp only [e_1754_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1754 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17634)) := by
  have c107286 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107287 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107288 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107289 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107290 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107291 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1754_18 := arithEq_of_rows h (row := 1754) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_18
  simp only [← c107286, ← c107287, ← c107288] at e_1754_18
  have e_1754_19 := arithEq_of_rows h (row := 1754) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1754_19
  simp only [← c107289, ← c107290, k2, ← c107291] at e_1754_19
  have hr := e_1754_19
  simp only [e_1754_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1755 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17635)) := by
  have c107292 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107293 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107294 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107295 := (publicBatchWrapper89_copies3352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107296 := (publicBatchWrapper89_copies3353 a h).1
  have c107297 := (publicBatchWrapper89_copies3353 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1755_0 := arithEq_of_rows h (row := 1755) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_0
  simp only [← c107292, ← c107293, ← c107294] at e_1755_0
  have e_1755_1 := arithEq_of_rows h (row := 1755) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_1
  simp only [← c107295, ← c107296, k2, ← c107297] at e_1755_1
  have hr := e_1755_1
  simp only [e_1755_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1755 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17636)) := by
  have c107298 := (publicBatchWrapper89_copies3353 a h).2.2.1
  have c107299 := (publicBatchWrapper89_copies3353 a h).2.2.2.1
  have c107300 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.1
  have c107301 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.1
  have c107302 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.1
  have c107303 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1755_2 := arithEq_of_rows h (row := 1755) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_2
  simp only [← c107298, ← c107299, ← c107300] at e_1755_2
  have e_1755_3 := arithEq_of_rows h (row := 1755) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_3
  simp only [← c107301, ← c107302, k2, ← c107303] at e_1755_3
  have hr := e_1755_3
  simp only [e_1755_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1755 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17637)) := by
  have c107304 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.1
  have c107305 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.1
  have c107306 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107307 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107308 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107309 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1755_4 := arithEq_of_rows h (row := 1755) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_4
  simp only [← c107304, ← c107305, ← c107306] at e_1755_4
  have e_1755_5 := arithEq_of_rows h (row := 1755) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_5
  simp only [← c107307, ← c107308, k2, ← c107309] at e_1755_5
  have hr := e_1755_5
  simp only [e_1755_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1755 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17638)) := by
  have c107310 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107311 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107312 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107313 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107314 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107315 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1755_6 := arithEq_of_rows h (row := 1755) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_6
  simp only [← c107310, ← c107311, ← c107312] at e_1755_6
  have e_1755_7 := arithEq_of_rows h (row := 1755) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_7
  simp only [← c107313, ← c107314, k2, ← c107315] at e_1755_7
  have hr := e_1755_7
  simp only [e_1755_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1755 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17639)) := by
  have c107316 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107317 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107318 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107319 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107320 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107321 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1755_8 := arithEq_of_rows h (row := 1755) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_8
  simp only [← c107316, ← c107317, ← c107318] at e_1755_8
  have e_1755_9 := arithEq_of_rows h (row := 1755) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_9
  simp only [← c107319, ← c107320, k2, ← c107321] at e_1755_9
  have hr := e_1755_9
  simp only [e_1755_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1755 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17640)) := by
  have c107322 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107323 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107324 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107325 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107326 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107327 := (publicBatchWrapper89_copies3353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1755_10 := arithEq_of_rows h (row := 1755) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_10
  simp only [← c107322, ← c107323, ← c107324] at e_1755_10
  have e_1755_11 := arithEq_of_rows h (row := 1755) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_11
  simp only [← c107325, ← c107326, k2, ← c107327] at e_1755_11
  have hr := e_1755_11
  simp only [e_1755_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1755 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17641)) := by
  have c107328 := (publicBatchWrapper89_copies3354 a h).1
  have c107329 := (publicBatchWrapper89_copies3354 a h).2.1
  have c107330 := (publicBatchWrapper89_copies3354 a h).2.2.1
  have c107331 := (publicBatchWrapper89_copies3354 a h).2.2.2.1
  have c107332 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.1
  have c107333 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1755_12 := arithEq_of_rows h (row := 1755) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_12
  simp only [← c107328, ← c107329, ← c107330] at e_1755_12
  have e_1755_13 := arithEq_of_rows h (row := 1755) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_13
  simp only [← c107331, ← c107332, k2, ← c107333] at e_1755_13
  have hr := e_1755_13
  simp only [e_1755_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1755 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17642)) := by
  have c107334 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.1
  have c107335 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.1
  have c107336 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.1
  have c107337 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.1
  have c107338 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107339 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1755_14 := arithEq_of_rows h (row := 1755) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_14
  simp only [← c107334, ← c107335, ← c107336] at e_1755_14
  have e_1755_15 := arithEq_of_rows h (row := 1755) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_15
  simp only [← c107337, ← c107338, k2, ← c107339] at e_1755_15
  have hr := e_1755_15
  simp only [e_1755_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1755 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17643)) := by
  have c107340 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107341 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107342 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107343 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107344 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107345 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1755_16 := arithEq_of_rows h (row := 1755) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_16
  simp only [← c107340, ← c107341, ← c107342] at e_1755_16
  have e_1755_17 := arithEq_of_rows h (row := 1755) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_17
  simp only [← c107343, ← c107344, k2, ← c107345] at e_1755_17
  have hr := e_1755_17
  simp only [e_1755_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1755 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17644)) := by
  have c107346 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107347 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107348 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107349 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107350 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107351 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1755_18 := arithEq_of_rows h (row := 1755) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_18
  simp only [← c107346, ← c107347, ← c107348] at e_1755_18
  have e_1755_19 := arithEq_of_rows h (row := 1755) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1755_19
  simp only [← c107349, ← c107350, k2, ← c107351] at e_1755_19
  have hr := e_1755_19
  simp only [e_1755_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1756 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17645)) := by
  have c107352 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107353 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107354 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107355 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107356 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107357 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1756_0 := arithEq_of_rows h (row := 1756) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_0
  simp only [← c107352, ← c107353, ← c107354] at e_1756_0
  have e_1756_1 := arithEq_of_rows h (row := 1756) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_1
  simp only [← c107355, ← c107356, k2, ← c107357] at e_1756_1
  have hr := e_1756_1
  simp only [e_1756_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1756 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17646)) := by
  have c107358 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107359 := (publicBatchWrapper89_copies3354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107360 := (publicBatchWrapper89_copies3355 a h).1
  have c107361 := (publicBatchWrapper89_copies3355 a h).2.1
  have c107362 := (publicBatchWrapper89_copies3355 a h).2.2.1
  have c107363 := (publicBatchWrapper89_copies3355 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1756_2 := arithEq_of_rows h (row := 1756) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_2
  simp only [← c107358, ← c107359, ← c107360] at e_1756_2
  have e_1756_3 := arithEq_of_rows h (row := 1756) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_3
  simp only [← c107361, ← c107362, k2, ← c107363] at e_1756_3
  have hr := e_1756_3
  simp only [e_1756_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1756 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17647)) := by
  have c107364 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.1
  have c107365 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.1
  have c107366 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.1
  have c107367 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.1
  have c107368 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.1
  have c107369 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1756_4 := arithEq_of_rows h (row := 1756) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_4
  simp only [← c107364, ← c107365, ← c107366] at e_1756_4
  have e_1756_5 := arithEq_of_rows h (row := 1756) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_5
  simp only [← c107367, ← c107368, k2, ← c107369] at e_1756_5
  have hr := e_1756_5
  simp only [e_1756_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1756 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17648)) := by
  have c107370 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107371 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107372 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107373 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107374 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107375 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1756_6 := arithEq_of_rows h (row := 1756) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_6
  simp only [← c107370, ← c107371, ← c107372] at e_1756_6
  have e_1756_7 := arithEq_of_rows h (row := 1756) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_7
  simp only [← c107373, ← c107374, k2, ← c107375] at e_1756_7
  have hr := e_1756_7
  simp only [e_1756_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1756 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17649)) := by
  have c107376 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107377 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107378 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107379 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107380 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107381 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1756_8 := arithEq_of_rows h (row := 1756) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_8
  simp only [← c107376, ← c107377, ← c107378] at e_1756_8
  have e_1756_9 := arithEq_of_rows h (row := 1756) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_9
  simp only [← c107379, ← c107380, k2, ← c107381] at e_1756_9
  have hr := e_1756_9
  simp only [e_1756_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1756 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17650)) := by
  have c107382 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107383 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107384 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107385 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107386 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107387 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1756_10 := arithEq_of_rows h (row := 1756) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_10
  simp only [← c107382, ← c107383, ← c107384] at e_1756_10
  have e_1756_11 := arithEq_of_rows h (row := 1756) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_11
  simp only [← c107385, ← c107386, k2, ← c107387] at e_1756_11
  have hr := e_1756_11
  simp only [e_1756_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1756 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17651)) := by
  have c107388 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107389 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107390 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107391 := (publicBatchWrapper89_copies3355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107392 := (publicBatchWrapper89_copies3356 a h).1
  have c107393 := (publicBatchWrapper89_copies3356 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1756_12 := arithEq_of_rows h (row := 1756) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_12
  simp only [← c107388, ← c107389, ← c107390] at e_1756_12
  have e_1756_13 := arithEq_of_rows h (row := 1756) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_13
  simp only [← c107391, ← c107392, k2, ← c107393] at e_1756_13
  have hr := e_1756_13
  simp only [e_1756_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1756 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17652)) := by
  have c107394 := (publicBatchWrapper89_copies3356 a h).2.2.1
  have c107395 := (publicBatchWrapper89_copies3356 a h).2.2.2.1
  have c107396 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.1
  have c107397 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.1
  have c107398 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.1
  have c107399 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1756_14 := arithEq_of_rows h (row := 1756) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_14
  simp only [← c107394, ← c107395, ← c107396] at e_1756_14
  have e_1756_15 := arithEq_of_rows h (row := 1756) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_15
  simp only [← c107397, ← c107398, k2, ← c107399] at e_1756_15
  have hr := e_1756_15
  simp only [e_1756_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1756 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17653)) := by
  have c107400 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.1
  have c107401 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.1
  have c107402 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107403 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107404 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107405 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1756_16 := arithEq_of_rows h (row := 1756) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_16
  simp only [← c107400, ← c107401, ← c107402] at e_1756_16
  have e_1756_17 := arithEq_of_rows h (row := 1756) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_17
  simp only [← c107403, ← c107404, k2, ← c107405] at e_1756_17
  have hr := e_1756_17
  simp only [e_1756_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1756 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17654)) := by
  have c107406 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107407 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107408 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107409 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107410 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107411 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1756_18 := arithEq_of_rows h (row := 1756) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_18
  simp only [← c107406, ← c107407, ← c107408] at e_1756_18
  have e_1756_19 := arithEq_of_rows h (row := 1756) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1756_19
  simp only [← c107409, ← c107410, k2, ← c107411] at e_1756_19
  have hr := e_1756_19
  simp only [e_1756_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1757 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17655)) := by
  have c107412 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107413 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107414 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107415 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107416 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107417 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1757_0 := arithEq_of_rows h (row := 1757) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_0
  simp only [← c107412, ← c107413, ← c107414] at e_1757_0
  have e_1757_1 := arithEq_of_rows h (row := 1757) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_1
  simp only [← c107415, ← c107416, k2, ← c107417] at e_1757_1
  have hr := e_1757_1
  simp only [e_1757_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1757 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17656)) := by
  have c107418 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107419 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107420 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107421 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107422 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107423 := (publicBatchWrapper89_copies3356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1757_2 := arithEq_of_rows h (row := 1757) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_2
  simp only [← c107418, ← c107419, ← c107420] at e_1757_2
  have e_1757_3 := arithEq_of_rows h (row := 1757) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_3
  simp only [← c107421, ← c107422, k2, ← c107423] at e_1757_3
  have hr := e_1757_3
  simp only [e_1757_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1757 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17657)) := by
  have c107424 := (publicBatchWrapper89_copies3357 a h).1
  have c107425 := (publicBatchWrapper89_copies3357 a h).2.1
  have c107426 := (publicBatchWrapper89_copies3357 a h).2.2.1
  have c107427 := (publicBatchWrapper89_copies3357 a h).2.2.2.1
  have c107428 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.1
  have c107429 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1757_4 := arithEq_of_rows h (row := 1757) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_4
  simp only [← c107424, ← c107425, ← c107426] at e_1757_4
  have e_1757_5 := arithEq_of_rows h (row := 1757) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_5
  simp only [← c107427, ← c107428, k2, ← c107429] at e_1757_5
  have hr := e_1757_5
  simp only [e_1757_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1757 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17658)) := by
  have c107430 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.1
  have c107431 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.1
  have c107432 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.1
  have c107433 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.1
  have c107434 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107435 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1757_6 := arithEq_of_rows h (row := 1757) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_6
  simp only [← c107430, ← c107431, ← c107432] at e_1757_6
  have e_1757_7 := arithEq_of_rows h (row := 1757) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_7
  simp only [← c107433, ← c107434, k2, ← c107435] at e_1757_7
  have hr := e_1757_7
  simp only [e_1757_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1757 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17659)) := by
  have c107436 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107437 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107438 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107439 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107440 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107441 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1757_8 := arithEq_of_rows h (row := 1757) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_8
  simp only [← c107436, ← c107437, ← c107438] at e_1757_8
  have e_1757_9 := arithEq_of_rows h (row := 1757) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_9
  simp only [← c107439, ← c107440, k2, ← c107441] at e_1757_9
  have hr := e_1757_9
  simp only [e_1757_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1757 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17660)) := by
  have c107442 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107443 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107444 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107445 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107446 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107447 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1757_10 := arithEq_of_rows h (row := 1757) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_10
  simp only [← c107442, ← c107443, ← c107444] at e_1757_10
  have e_1757_11 := arithEq_of_rows h (row := 1757) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_11
  simp only [← c107445, ← c107446, k2, ← c107447] at e_1757_11
  have hr := e_1757_11
  simp only [e_1757_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1757 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17661)) := by
  have c107448 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107449 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107450 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107451 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107452 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107453 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1757_12 := arithEq_of_rows h (row := 1757) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_12
  simp only [← c107448, ← c107449, ← c107450] at e_1757_12
  have e_1757_13 := arithEq_of_rows h (row := 1757) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_13
  simp only [← c107451, ← c107452, k2, ← c107453] at e_1757_13
  have hr := e_1757_13
  simp only [e_1757_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1757 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17662)) := by
  have c107454 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107455 := (publicBatchWrapper89_copies3357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107456 := (publicBatchWrapper89_copies3358 a h).1
  have c107457 := (publicBatchWrapper89_copies3358 a h).2.1
  have c107458 := (publicBatchWrapper89_copies3358 a h).2.2.1
  have c107459 := (publicBatchWrapper89_copies3358 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1757_14 := arithEq_of_rows h (row := 1757) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_14
  simp only [← c107454, ← c107455, ← c107456] at e_1757_14
  have e_1757_15 := arithEq_of_rows h (row := 1757) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_15
  simp only [← c107457, ← c107458, k2, ← c107459] at e_1757_15
  have hr := e_1757_15
  simp only [e_1757_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1757 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17663)) := by
  have c107460 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.1
  have c107461 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.1
  have c107462 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.1
  have c107463 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.1
  have c107464 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.1
  have c107465 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1757_16 := arithEq_of_rows h (row := 1757) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_16
  simp only [← c107460, ← c107461, ← c107462] at e_1757_16
  have e_1757_17 := arithEq_of_rows h (row := 1757) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_17
  simp only [← c107463, ← c107464, k2, ← c107465] at e_1757_17
  have hr := e_1757_17
  simp only [e_1757_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1757 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17664)) := by
  have c107466 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107467 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107468 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107469 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107470 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107471 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1757_18 := arithEq_of_rows h (row := 1757) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_18
  simp only [← c107466, ← c107467, ← c107468] at e_1757_18
  have e_1757_19 := arithEq_of_rows h (row := 1757) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1757_19
  simp only [← c107469, ← c107470, k2, ← c107471] at e_1757_19
  have hr := e_1757_19
  simp only [e_1757_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1758 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17665)) := by
  have c107472 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107473 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107474 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107475 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107476 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107477 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1758_0 := arithEq_of_rows h (row := 1758) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_0
  simp only [← c107472, ← c107473, ← c107474] at e_1758_0
  have e_1758_1 := arithEq_of_rows h (row := 1758) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_1
  simp only [← c107475, ← c107476, k2, ← c107477] at e_1758_1
  have hr := e_1758_1
  simp only [e_1758_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1758 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17666)) := by
  have c107478 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107479 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107480 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107481 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107482 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107483 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1758_2 := arithEq_of_rows h (row := 1758) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_2
  simp only [← c107478, ← c107479, ← c107480] at e_1758_2
  have e_1758_3 := arithEq_of_rows h (row := 1758) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_3
  simp only [← c107481, ← c107482, k2, ← c107483] at e_1758_3
  have hr := e_1758_3
  simp only [e_1758_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1758 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17667)) := by
  have c107484 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107485 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107486 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107487 := (publicBatchWrapper89_copies3358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107488 := (publicBatchWrapper89_copies3359 a h).1
  have c107489 := (publicBatchWrapper89_copies3359 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1758_4 := arithEq_of_rows h (row := 1758) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_4
  simp only [← c107484, ← c107485, ← c107486] at e_1758_4
  have e_1758_5 := arithEq_of_rows h (row := 1758) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_5
  simp only [← c107487, ← c107488, k2, ← c107489] at e_1758_5
  have hr := e_1758_5
  simp only [e_1758_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1758 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17668)) := by
  have c107490 := (publicBatchWrapper89_copies3359 a h).2.2.1
  have c107491 := (publicBatchWrapper89_copies3359 a h).2.2.2.1
  have c107492 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.1
  have c107493 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.1
  have c107494 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.1
  have c107495 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1758_6 := arithEq_of_rows h (row := 1758) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_6
  simp only [← c107490, ← c107491, ← c107492] at e_1758_6
  have e_1758_7 := arithEq_of_rows h (row := 1758) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_7
  simp only [← c107493, ← c107494, k2, ← c107495] at e_1758_7
  have hr := e_1758_7
  simp only [e_1758_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1758 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17669)) := by
  have c107496 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.1
  have c107497 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.1
  have c107498 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107499 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107500 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107501 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1758_8 := arithEq_of_rows h (row := 1758) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_8
  simp only [← c107496, ← c107497, ← c107498] at e_1758_8
  have e_1758_9 := arithEq_of_rows h (row := 1758) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_9
  simp only [← c107499, ← c107500, k2, ← c107501] at e_1758_9
  have hr := e_1758_9
  simp only [e_1758_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1758 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17670)) := by
  have c107502 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107503 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107504 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107505 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107506 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107507 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1758_10 := arithEq_of_rows h (row := 1758) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_10
  simp only [← c107502, ← c107503, ← c107504] at e_1758_10
  have e_1758_11 := arithEq_of_rows h (row := 1758) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_11
  simp only [← c107505, ← c107506, k2, ← c107507] at e_1758_11
  have hr := e_1758_11
  simp only [e_1758_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1758 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17671)) := by
  have c107508 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107509 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107510 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107511 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107512 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107513 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1758_12 := arithEq_of_rows h (row := 1758) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_12
  simp only [← c107508, ← c107509, ← c107510] at e_1758_12
  have e_1758_13 := arithEq_of_rows h (row := 1758) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_13
  simp only [← c107511, ← c107512, k2, ← c107513] at e_1758_13
  have hr := e_1758_13
  simp only [e_1758_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1758 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17672)) := by
  have c107514 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107515 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107516 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107517 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107518 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107519 := (publicBatchWrapper89_copies3359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1758_14 := arithEq_of_rows h (row := 1758) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_14
  simp only [← c107514, ← c107515, ← c107516] at e_1758_14
  have e_1758_15 := arithEq_of_rows h (row := 1758) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_15
  simp only [← c107517, ← c107518, k2, ← c107519] at e_1758_15
  have hr := e_1758_15
  simp only [e_1758_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1758 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17673)) := by
  have c107520 := (publicBatchWrapper89_copies3360 a h).1
  have c107521 := (publicBatchWrapper89_copies3360 a h).2.1
  have c107522 := (publicBatchWrapper89_copies3360 a h).2.2.1
  have c107523 := (publicBatchWrapper89_copies3360 a h).2.2.2.1
  have c107524 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.1
  have c107525 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1758_16 := arithEq_of_rows h (row := 1758) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_16
  simp only [← c107520, ← c107521, ← c107522] at e_1758_16
  have e_1758_17 := arithEq_of_rows h (row := 1758) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_17
  simp only [← c107523, ← c107524, k2, ← c107525] at e_1758_17
  have hr := e_1758_17
  simp only [e_1758_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1758 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17674)) := by
  have c107526 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.1
  have c107527 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.1
  have c107528 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.1
  have c107529 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.1
  have c107530 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107531 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1758_18 := arithEq_of_rows h (row := 1758) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_18
  simp only [← c107526, ← c107527, ← c107528] at e_1758_18
  have e_1758_19 := arithEq_of_rows h (row := 1758) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1758_19
  simp only [← c107529, ← c107530, k2, ← c107531] at e_1758_19
  have hr := e_1758_19
  simp only [e_1758_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1759 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17675)) := by
  have c107532 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107533 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107534 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107535 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107536 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107537 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1759_0 := arithEq_of_rows h (row := 1759) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_0
  simp only [← c107532, ← c107533, ← c107534] at e_1759_0
  have e_1759_1 := arithEq_of_rows h (row := 1759) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_1
  simp only [← c107535, ← c107536, k2, ← c107537] at e_1759_1
  have hr := e_1759_1
  simp only [e_1759_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1759 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17676)) := by
  have c107538 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107539 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107540 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107541 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107542 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107543 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1759_2 := arithEq_of_rows h (row := 1759) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_2
  simp only [← c107538, ← c107539, ← c107540] at e_1759_2
  have e_1759_3 := arithEq_of_rows h (row := 1759) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_3
  simp only [← c107541, ← c107542, k2, ← c107543] at e_1759_3
  have hr := e_1759_3
  simp only [e_1759_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1759 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17677)) := by
  have c107544 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107545 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107546 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107547 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107548 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107549 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1759_4 := arithEq_of_rows h (row := 1759) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_4
  simp only [← c107544, ← c107545, ← c107546] at e_1759_4
  have e_1759_5 := arithEq_of_rows h (row := 1759) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_5
  simp only [← c107547, ← c107548, k2, ← c107549] at e_1759_5
  have hr := e_1759_5
  simp only [e_1759_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1759 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17678)) := by
  have c107550 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107551 := (publicBatchWrapper89_copies3360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107552 := (publicBatchWrapper89_copies3361 a h).1
  have c107553 := (publicBatchWrapper89_copies3361 a h).2.1
  have c107554 := (publicBatchWrapper89_copies3361 a h).2.2.1
  have c107555 := (publicBatchWrapper89_copies3361 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1759_6 := arithEq_of_rows h (row := 1759) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_6
  simp only [← c107550, ← c107551, ← c107552] at e_1759_6
  have e_1759_7 := arithEq_of_rows h (row := 1759) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_7
  simp only [← c107553, ← c107554, k2, ← c107555] at e_1759_7
  have hr := e_1759_7
  simp only [e_1759_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1759 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17679)) := by
  have c107556 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.1
  have c107557 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.1
  have c107558 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.1
  have c107559 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.1
  have c107560 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.1
  have c107561 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1759_8 := arithEq_of_rows h (row := 1759) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_8
  simp only [← c107556, ← c107557, ← c107558] at e_1759_8
  have e_1759_9 := arithEq_of_rows h (row := 1759) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_9
  simp only [← c107559, ← c107560, k2, ← c107561] at e_1759_9
  have hr := e_1759_9
  simp only [e_1759_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1759 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17680)) := by
  have c107562 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107563 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107564 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107565 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107566 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107567 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1759_10 := arithEq_of_rows h (row := 1759) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_10
  simp only [← c107562, ← c107563, ← c107564] at e_1759_10
  have e_1759_11 := arithEq_of_rows h (row := 1759) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_11
  simp only [← c107565, ← c107566, k2, ← c107567] at e_1759_11
  have hr := e_1759_11
  simp only [e_1759_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1759 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17681)) := by
  have c107568 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107569 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107570 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107571 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107572 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107573 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1759_12 := arithEq_of_rows h (row := 1759) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_12
  simp only [← c107568, ← c107569, ← c107570] at e_1759_12
  have e_1759_13 := arithEq_of_rows h (row := 1759) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_13
  simp only [← c107571, ← c107572, k2, ← c107573] at e_1759_13
  have hr := e_1759_13
  simp only [e_1759_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1759 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17682)) := by
  have c107574 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107575 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107576 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107577 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107578 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107579 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1759_14 := arithEq_of_rows h (row := 1759) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_14
  simp only [← c107574, ← c107575, ← c107576] at e_1759_14
  have e_1759_15 := arithEq_of_rows h (row := 1759) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_15
  simp only [← c107577, ← c107578, k2, ← c107579] at e_1759_15
  have hr := e_1759_15
  simp only [e_1759_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1759 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17683)) := by
  have c107580 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107581 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107582 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107583 := (publicBatchWrapper89_copies3361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107584 := (publicBatchWrapper89_copies3362 a h).1
  have c107585 := (publicBatchWrapper89_copies3362 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1759_16 := arithEq_of_rows h (row := 1759) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_16
  simp only [← c107580, ← c107581, ← c107582] at e_1759_16
  have e_1759_17 := arithEq_of_rows h (row := 1759) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_17
  simp only [← c107583, ← c107584, k2, ← c107585] at e_1759_17
  have hr := e_1759_17
  simp only [e_1759_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1759 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17684)) := by
  have c107586 := (publicBatchWrapper89_copies3362 a h).2.2.1
  have c107587 := (publicBatchWrapper89_copies3362 a h).2.2.2.1
  have c107588 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.1
  have c107589 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.1
  have c107590 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.1
  have c107591 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1759_18 := arithEq_of_rows h (row := 1759) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_18
  simp only [← c107586, ← c107587, ← c107588] at e_1759_18
  have e_1759_19 := arithEq_of_rows h (row := 1759) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1759_19
  simp only [← c107589, ← c107590, k2, ← c107591] at e_1759_19
  have hr := e_1759_19
  simp only [e_1759_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1760 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17685)) := by
  have c107592 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.1
  have c107593 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.1
  have c107594 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107595 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107596 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107597 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1760_0 := arithEq_of_rows h (row := 1760) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_0
  simp only [← c107592, ← c107593, ← c107594] at e_1760_0
  have e_1760_1 := arithEq_of_rows h (row := 1760) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_1
  simp only [← c107595, ← c107596, k2, ← c107597] at e_1760_1
  have hr := e_1760_1
  simp only [e_1760_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1760 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17686)) := by
  have c107598 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107599 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107600 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107601 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107602 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107603 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1760_2 := arithEq_of_rows h (row := 1760) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_2
  simp only [← c107598, ← c107599, ← c107600] at e_1760_2
  have e_1760_3 := arithEq_of_rows h (row := 1760) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_3
  simp only [← c107601, ← c107602, k2, ← c107603] at e_1760_3
  have hr := e_1760_3
  simp only [e_1760_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1760 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17687)) := by
  have c107604 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107605 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107606 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107607 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107608 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107609 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1760_4 := arithEq_of_rows h (row := 1760) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_4
  simp only [← c107604, ← c107605, ← c107606] at e_1760_4
  have e_1760_5 := arithEq_of_rows h (row := 1760) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_5
  simp only [← c107607, ← c107608, k2, ← c107609] at e_1760_5
  have hr := e_1760_5
  simp only [e_1760_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1760 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17688)) := by
  have c107610 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107611 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107612 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107613 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107614 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107615 := (publicBatchWrapper89_copies3362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1760_6 := arithEq_of_rows h (row := 1760) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_6
  simp only [← c107610, ← c107611, ← c107612] at e_1760_6
  have e_1760_7 := arithEq_of_rows h (row := 1760) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_7
  simp only [← c107613, ← c107614, k2, ← c107615] at e_1760_7
  have hr := e_1760_7
  simp only [e_1760_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1760 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17689)) := by
  have c107616 := (publicBatchWrapper89_copies3363 a h).1
  have c107617 := (publicBatchWrapper89_copies3363 a h).2.1
  have c107618 := (publicBatchWrapper89_copies3363 a h).2.2.1
  have c107619 := (publicBatchWrapper89_copies3363 a h).2.2.2.1
  have c107620 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.1
  have c107621 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1760_8 := arithEq_of_rows h (row := 1760) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_8
  simp only [← c107616, ← c107617, ← c107618] at e_1760_8
  have e_1760_9 := arithEq_of_rows h (row := 1760) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_9
  simp only [← c107619, ← c107620, k2, ← c107621] at e_1760_9
  have hr := e_1760_9
  simp only [e_1760_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1760 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17690)) := by
  have c107622 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.1
  have c107623 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.1
  have c107624 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.1
  have c107625 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.1
  have c107626 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107627 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1760_10 := arithEq_of_rows h (row := 1760) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_10
  simp only [← c107622, ← c107623, ← c107624] at e_1760_10
  have e_1760_11 := arithEq_of_rows h (row := 1760) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_11
  simp only [← c107625, ← c107626, k2, ← c107627] at e_1760_11
  have hr := e_1760_11
  simp only [e_1760_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1760 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17691)) := by
  have c107628 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107629 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107630 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107631 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107632 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107633 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1760_12 := arithEq_of_rows h (row := 1760) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_12
  simp only [← c107628, ← c107629, ← c107630] at e_1760_12
  have e_1760_13 := arithEq_of_rows h (row := 1760) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_13
  simp only [← c107631, ← c107632, k2, ← c107633] at e_1760_13
  have hr := e_1760_13
  simp only [e_1760_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1760 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17692)) := by
  have c107634 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107635 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107636 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107637 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107638 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107639 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1760_14 := arithEq_of_rows h (row := 1760) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_14
  simp only [← c107634, ← c107635, ← c107636] at e_1760_14
  have e_1760_15 := arithEq_of_rows h (row := 1760) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_15
  simp only [← c107637, ← c107638, k2, ← c107639] at e_1760_15
  have hr := e_1760_15
  simp only [e_1760_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1760 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17693)) := by
  have c107640 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107641 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107642 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107643 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107644 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107645 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1760_16 := arithEq_of_rows h (row := 1760) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_16
  simp only [← c107640, ← c107641, ← c107642] at e_1760_16
  have e_1760_17 := arithEq_of_rows h (row := 1760) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_17
  simp only [← c107643, ← c107644, k2, ← c107645] at e_1760_17
  have hr := e_1760_17
  simp only [e_1760_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1760 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17694)) := by
  have c107646 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107647 := (publicBatchWrapper89_copies3363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107648 := (publicBatchWrapper89_copies3364 a h).1
  have c107649 := (publicBatchWrapper89_copies3364 a h).2.1
  have c107650 := (publicBatchWrapper89_copies3364 a h).2.2.1
  have c107651 := (publicBatchWrapper89_copies3364 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1760_18 := arithEq_of_rows h (row := 1760) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_18
  simp only [← c107646, ← c107647, ← c107648] at e_1760_18
  have e_1760_19 := arithEq_of_rows h (row := 1760) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1760_19
  simp only [← c107649, ← c107650, k2, ← c107651] at e_1760_19
  have hr := e_1760_19
  simp only [e_1760_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1761 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17695)) := by
  have c107652 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.1
  have c107653 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.1
  have c107654 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.1
  have c107655 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.1
  have c107656 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.1
  have c107657 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1761_0 := arithEq_of_rows h (row := 1761) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_0
  simp only [← c107652, ← c107653, ← c107654] at e_1761_0
  have e_1761_1 := arithEq_of_rows h (row := 1761) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_1
  simp only [← c107655, ← c107656, k2, ← c107657] at e_1761_1
  have hr := e_1761_1
  simp only [e_1761_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1761 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17696)) := by
  have c107658 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107659 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107660 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107661 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107662 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107663 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1761_2 := arithEq_of_rows h (row := 1761) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_2
  simp only [← c107658, ← c107659, ← c107660] at e_1761_2
  have e_1761_3 := arithEq_of_rows h (row := 1761) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_3
  simp only [← c107661, ← c107662, k2, ← c107663] at e_1761_3
  have hr := e_1761_3
  simp only [e_1761_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1761 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17697)) := by
  have c107664 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107665 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107666 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107667 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107668 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107669 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1761_4 := arithEq_of_rows h (row := 1761) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_4
  simp only [← c107664, ← c107665, ← c107666] at e_1761_4
  have e_1761_5 := arithEq_of_rows h (row := 1761) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_5
  simp only [← c107667, ← c107668, k2, ← c107669] at e_1761_5
  have hr := e_1761_5
  simp only [e_1761_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1761 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17698)) := by
  have c107670 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107671 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107672 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107673 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107674 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107675 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1761_6 := arithEq_of_rows h (row := 1761) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_6
  simp only [← c107670, ← c107671, ← c107672] at e_1761_6
  have e_1761_7 := arithEq_of_rows h (row := 1761) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_7
  simp only [← c107673, ← c107674, k2, ← c107675] at e_1761_7
  have hr := e_1761_7
  simp only [e_1761_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1761 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17699)) := by
  have c107676 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107677 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107678 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107679 := (publicBatchWrapper89_copies3364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107680 := (publicBatchWrapper89_copies3365 a h).1
  have c107681 := (publicBatchWrapper89_copies3365 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1761_8 := arithEq_of_rows h (row := 1761) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_8
  simp only [← c107676, ← c107677, ← c107678] at e_1761_8
  have e_1761_9 := arithEq_of_rows h (row := 1761) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_9
  simp only [← c107679, ← c107680, k2, ← c107681] at e_1761_9
  have hr := e_1761_9
  simp only [e_1761_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1761 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17700)) := by
  have c107682 := (publicBatchWrapper89_copies3365 a h).2.2.1
  have c107683 := (publicBatchWrapper89_copies3365 a h).2.2.2.1
  have c107684 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.1
  have c107685 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.1
  have c107686 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.1
  have c107687 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1761_10 := arithEq_of_rows h (row := 1761) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_10
  simp only [← c107682, ← c107683, ← c107684] at e_1761_10
  have e_1761_11 := arithEq_of_rows h (row := 1761) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_11
  simp only [← c107685, ← c107686, k2, ← c107687] at e_1761_11
  have hr := e_1761_11
  simp only [e_1761_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1761 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17701)) := by
  have c107688 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.1
  have c107689 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.1
  have c107690 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107691 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107692 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107693 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1761_12 := arithEq_of_rows h (row := 1761) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_12
  simp only [← c107688, ← c107689, ← c107690] at e_1761_12
  have e_1761_13 := arithEq_of_rows h (row := 1761) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_13
  simp only [← c107691, ← c107692, k2, ← c107693] at e_1761_13
  have hr := e_1761_13
  simp only [e_1761_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1761 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17702)) := by
  have c107694 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107695 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107696 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107697 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107698 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107699 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1761_14 := arithEq_of_rows h (row := 1761) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_14
  simp only [← c107694, ← c107695, ← c107696] at e_1761_14
  have e_1761_15 := arithEq_of_rows h (row := 1761) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_15
  simp only [← c107697, ← c107698, k2, ← c107699] at e_1761_15
  have hr := e_1761_15
  simp only [e_1761_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1761 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17703)) := by
  have c107700 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107701 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107702 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107703 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107704 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107705 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1761_16 := arithEq_of_rows h (row := 1761) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_16
  simp only [← c107700, ← c107701, ← c107702] at e_1761_16
  have e_1761_17 := arithEq_of_rows h (row := 1761) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_17
  simp only [← c107703, ← c107704, k2, ← c107705] at e_1761_17
  have hr := e_1761_17
  simp only [e_1761_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1761 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17704)) := by
  have c107706 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107707 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107708 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107709 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107710 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107711 := (publicBatchWrapper89_copies3365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1761_18 := arithEq_of_rows h (row := 1761) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_18
  simp only [← c107706, ← c107707, ← c107708] at e_1761_18
  have e_1761_19 := arithEq_of_rows h (row := 1761) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1761_19
  simp only [← c107709, ← c107710, k2, ← c107711] at e_1761_19
  have hr := e_1761_19
  simp only [e_1761_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1762 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17705)) := by
  have c107712 := (publicBatchWrapper89_copies3366 a h).1
  have c107713 := (publicBatchWrapper89_copies3366 a h).2.1
  have c107714 := (publicBatchWrapper89_copies3366 a h).2.2.1
  have c107715 := (publicBatchWrapper89_copies3366 a h).2.2.2.1
  have c107716 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.1
  have c107717 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1762_0 := arithEq_of_rows h (row := 1762) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_0
  simp only [← c107712, ← c107713, ← c107714] at e_1762_0
  have e_1762_1 := arithEq_of_rows h (row := 1762) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_1
  simp only [← c107715, ← c107716, k2, ← c107717] at e_1762_1
  have hr := e_1762_1
  simp only [e_1762_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1762 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17706)) := by
  have c107718 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.1
  have c107719 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.1
  have c107720 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.1
  have c107721 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.1
  have c107722 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107723 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1762_2 := arithEq_of_rows h (row := 1762) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_2
  simp only [← c107718, ← c107719, ← c107720] at e_1762_2
  have e_1762_3 := arithEq_of_rows h (row := 1762) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_3
  simp only [← c107721, ← c107722, k2, ← c107723] at e_1762_3
  have hr := e_1762_3
  simp only [e_1762_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1762 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17707)) := by
  have c107724 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107725 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107726 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107727 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107728 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107729 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1762_4 := arithEq_of_rows h (row := 1762) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_4
  simp only [← c107724, ← c107725, ← c107726] at e_1762_4
  have e_1762_5 := arithEq_of_rows h (row := 1762) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_5
  simp only [← c107727, ← c107728, k2, ← c107729] at e_1762_5
  have hr := e_1762_5
  simp only [e_1762_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1762 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17708)) := by
  have c107730 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107731 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107732 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107733 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107734 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107735 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1762_6 := arithEq_of_rows h (row := 1762) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_6
  simp only [← c107730, ← c107731, ← c107732] at e_1762_6
  have e_1762_7 := arithEq_of_rows h (row := 1762) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_7
  simp only [← c107733, ← c107734, k2, ← c107735] at e_1762_7
  have hr := e_1762_7
  simp only [e_1762_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1762 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17709)) := by
  have c107736 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107737 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107738 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107739 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107740 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107741 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1762_8 := arithEq_of_rows h (row := 1762) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_8
  simp only [← c107736, ← c107737, ← c107738] at e_1762_8
  have e_1762_9 := arithEq_of_rows h (row := 1762) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_9
  simp only [← c107739, ← c107740, k2, ← c107741] at e_1762_9
  have hr := e_1762_9
  simp only [e_1762_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1762 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17710)) := by
  have c107742 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107743 := (publicBatchWrapper89_copies3366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107744 := (publicBatchWrapper89_copies3367 a h).1
  have c107745 := (publicBatchWrapper89_copies3367 a h).2.1
  have c107746 := (publicBatchWrapper89_copies3367 a h).2.2.1
  have c107747 := (publicBatchWrapper89_copies3367 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1762_10 := arithEq_of_rows h (row := 1762) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_10
  simp only [← c107742, ← c107743, ← c107744] at e_1762_10
  have e_1762_11 := arithEq_of_rows h (row := 1762) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_11
  simp only [← c107745, ← c107746, k2, ← c107747] at e_1762_11
  have hr := e_1762_11
  simp only [e_1762_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1762 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17711)) := by
  have c107748 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.1
  have c107749 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.1
  have c107750 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.1
  have c107751 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.1
  have c107752 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.1
  have c107753 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1762_12 := arithEq_of_rows h (row := 1762) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_12
  simp only [← c107748, ← c107749, ← c107750] at e_1762_12
  have e_1762_13 := arithEq_of_rows h (row := 1762) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_13
  simp only [← c107751, ← c107752, k2, ← c107753] at e_1762_13
  have hr := e_1762_13
  simp only [e_1762_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1762 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17712)) := by
  have c107754 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107755 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107756 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107757 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107758 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107759 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1762_14 := arithEq_of_rows h (row := 1762) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_14
  simp only [← c107754, ← c107755, ← c107756] at e_1762_14
  have e_1762_15 := arithEq_of_rows h (row := 1762) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_15
  simp only [← c107757, ← c107758, k2, ← c107759] at e_1762_15
  have hr := e_1762_15
  simp only [e_1762_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1762 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17713)) := by
  have c107760 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107761 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107762 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107763 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107764 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107765 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1762_16 := arithEq_of_rows h (row := 1762) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_16
  simp only [← c107760, ← c107761, ← c107762] at e_1762_16
  have e_1762_17 := arithEq_of_rows h (row := 1762) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_17
  simp only [← c107763, ← c107764, k2, ← c107765] at e_1762_17
  have hr := e_1762_17
  simp only [e_1762_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1762 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17714)) := by
  have c107766 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107767 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107768 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107769 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107770 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107771 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1762_18 := arithEq_of_rows h (row := 1762) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_18
  simp only [← c107766, ← c107767, ← c107768] at e_1762_18
  have e_1762_19 := arithEq_of_rows h (row := 1762) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1762_19
  simp only [← c107769, ← c107770, k2, ← c107771] at e_1762_19
  have hr := e_1762_19
  simp only [e_1762_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1763 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17715)) := by
  have c107772 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107773 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107774 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107775 := (publicBatchWrapper89_copies3367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107776 := (publicBatchWrapper89_copies3368 a h).1
  have c107777 := (publicBatchWrapper89_copies3368 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1763_0 := arithEq_of_rows h (row := 1763) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_0
  simp only [← c107772, ← c107773, ← c107774] at e_1763_0
  have e_1763_1 := arithEq_of_rows h (row := 1763) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_1
  simp only [← c107775, ← c107776, k2, ← c107777] at e_1763_1
  have hr := e_1763_1
  simp only [e_1763_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1763 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17716)) := by
  have c107778 := (publicBatchWrapper89_copies3368 a h).2.2.1
  have c107779 := (publicBatchWrapper89_copies3368 a h).2.2.2.1
  have c107780 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.1
  have c107781 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.1
  have c107782 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.1
  have c107783 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1763_2 := arithEq_of_rows h (row := 1763) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_2
  simp only [← c107778, ← c107779, ← c107780] at e_1763_2
  have e_1763_3 := arithEq_of_rows h (row := 1763) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_3
  simp only [← c107781, ← c107782, k2, ← c107783] at e_1763_3
  have hr := e_1763_3
  simp only [e_1763_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1763 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17717)) := by
  have c107784 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.1
  have c107785 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.1
  have c107786 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107787 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107788 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107789 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1763_4 := arithEq_of_rows h (row := 1763) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_4
  simp only [← c107784, ← c107785, ← c107786] at e_1763_4
  have e_1763_5 := arithEq_of_rows h (row := 1763) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_5
  simp only [← c107787, ← c107788, k2, ← c107789] at e_1763_5
  have hr := e_1763_5
  simp only [e_1763_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1763 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17718)) := by
  have c107790 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107791 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107792 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107793 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107794 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107795 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1763_6 := arithEq_of_rows h (row := 1763) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_6
  simp only [← c107790, ← c107791, ← c107792] at e_1763_6
  have e_1763_7 := arithEq_of_rows h (row := 1763) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_7
  simp only [← c107793, ← c107794, k2, ← c107795] at e_1763_7
  have hr := e_1763_7
  simp only [e_1763_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1763 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17719)) := by
  have c107796 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107797 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107798 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107799 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107800 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107801 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1763_8 := arithEq_of_rows h (row := 1763) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_8
  simp only [← c107796, ← c107797, ← c107798] at e_1763_8
  have e_1763_9 := arithEq_of_rows h (row := 1763) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_9
  simp only [← c107799, ← c107800, k2, ← c107801] at e_1763_9
  have hr := e_1763_9
  simp only [e_1763_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1763 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17720)) := by
  have c107802 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107803 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107804 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107805 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107806 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107807 := (publicBatchWrapper89_copies3368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1763_10 := arithEq_of_rows h (row := 1763) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_10
  simp only [← c107802, ← c107803, ← c107804] at e_1763_10
  have e_1763_11 := arithEq_of_rows h (row := 1763) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_11
  simp only [← c107805, ← c107806, k2, ← c107807] at e_1763_11
  have hr := e_1763_11
  simp only [e_1763_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1763 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17721)) := by
  have c107808 := (publicBatchWrapper89_copies3369 a h).1
  have c107809 := (publicBatchWrapper89_copies3369 a h).2.1
  have c107810 := (publicBatchWrapper89_copies3369 a h).2.2.1
  have c107811 := (publicBatchWrapper89_copies3369 a h).2.2.2.1
  have c107812 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.1
  have c107813 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1763_12 := arithEq_of_rows h (row := 1763) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_12
  simp only [← c107808, ← c107809, ← c107810] at e_1763_12
  have e_1763_13 := arithEq_of_rows h (row := 1763) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_13
  simp only [← c107811, ← c107812, k2, ← c107813] at e_1763_13
  have hr := e_1763_13
  simp only [e_1763_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1763 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17722)) := by
  have c107814 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.1
  have c107815 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.1
  have c107816 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.1
  have c107817 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.1
  have c107818 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107819 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1763_14 := arithEq_of_rows h (row := 1763) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_14
  simp only [← c107814, ← c107815, ← c107816] at e_1763_14
  have e_1763_15 := arithEq_of_rows h (row := 1763) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_15
  simp only [← c107817, ← c107818, k2, ← c107819] at e_1763_15
  have hr := e_1763_15
  simp only [e_1763_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1763 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17723)) := by
  have c107820 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107821 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107822 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107823 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107824 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107825 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1763_16 := arithEq_of_rows h (row := 1763) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_16
  simp only [← c107820, ← c107821, ← c107822] at e_1763_16
  have e_1763_17 := arithEq_of_rows h (row := 1763) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_17
  simp only [← c107823, ← c107824, k2, ← c107825] at e_1763_17
  have hr := e_1763_17
  simp only [e_1763_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1763 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17724)) := by
  have c107826 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107827 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107828 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107829 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107830 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107831 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1763_18 := arithEq_of_rows h (row := 1763) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_18
  simp only [← c107826, ← c107827, ← c107828] at e_1763_18
  have e_1763_19 := arithEq_of_rows h (row := 1763) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1763_19
  simp only [← c107829, ← c107830, k2, ← c107831] at e_1763_19
  have hr := e_1763_19
  simp only [e_1763_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1764 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17725)) := by
  have c107832 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107833 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107834 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107835 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107836 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107837 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1764_0 := arithEq_of_rows h (row := 1764) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_0
  simp only [← c107832, ← c107833, ← c107834] at e_1764_0
  have e_1764_1 := arithEq_of_rows h (row := 1764) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_1
  simp only [← c107835, ← c107836, k2, ← c107837] at e_1764_1
  have hr := e_1764_1
  simp only [e_1764_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1764 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17726)) := by
  have c107838 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107839 := (publicBatchWrapper89_copies3369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107840 := (publicBatchWrapper89_copies3370 a h).1
  have c107841 := (publicBatchWrapper89_copies3370 a h).2.1
  have c107842 := (publicBatchWrapper89_copies3370 a h).2.2.1
  have c107843 := (publicBatchWrapper89_copies3370 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1764_2 := arithEq_of_rows h (row := 1764) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_2
  simp only [← c107838, ← c107839, ← c107840] at e_1764_2
  have e_1764_3 := arithEq_of_rows h (row := 1764) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_3
  simp only [← c107841, ← c107842, k2, ← c107843] at e_1764_3
  have hr := e_1764_3
  simp only [e_1764_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1764 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17727)) := by
  have c107844 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.1
  have c107845 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.1
  have c107846 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.1
  have c107847 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.1
  have c107848 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.1
  have c107849 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1764_4 := arithEq_of_rows h (row := 1764) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_4
  simp only [← c107844, ← c107845, ← c107846] at e_1764_4
  have e_1764_5 := arithEq_of_rows h (row := 1764) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_5
  simp only [← c107847, ← c107848, k2, ← c107849] at e_1764_5
  have hr := e_1764_5
  simp only [e_1764_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1764 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17728)) := by
  have c107850 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107851 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107852 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107853 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107854 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107855 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1764_6 := arithEq_of_rows h (row := 1764) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_6
  simp only [← c107850, ← c107851, ← c107852] at e_1764_6
  have e_1764_7 := arithEq_of_rows h (row := 1764) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_7
  simp only [← c107853, ← c107854, k2, ← c107855] at e_1764_7
  have hr := e_1764_7
  simp only [e_1764_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1764 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17729)) := by
  have c107856 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107857 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107858 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107859 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107860 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107861 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1764_8 := arithEq_of_rows h (row := 1764) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_8
  simp only [← c107856, ← c107857, ← c107858] at e_1764_8
  have e_1764_9 := arithEq_of_rows h (row := 1764) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_9
  simp only [← c107859, ← c107860, k2, ← c107861] at e_1764_9
  have hr := e_1764_9
  simp only [e_1764_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1764 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17730)) := by
  have c107862 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107863 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107864 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107865 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107866 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107867 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1764_10 := arithEq_of_rows h (row := 1764) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_10
  simp only [← c107862, ← c107863, ← c107864] at e_1764_10
  have e_1764_11 := arithEq_of_rows h (row := 1764) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_11
  simp only [← c107865, ← c107866, k2, ← c107867] at e_1764_11
  have hr := e_1764_11
  simp only [e_1764_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1764 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17731)) := by
  have c107868 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107869 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107870 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107871 := (publicBatchWrapper89_copies3370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107872 := (publicBatchWrapper89_copies3371 a h).1
  have c107873 := (publicBatchWrapper89_copies3371 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1764_12 := arithEq_of_rows h (row := 1764) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_12
  simp only [← c107868, ← c107869, ← c107870] at e_1764_12
  have e_1764_13 := arithEq_of_rows h (row := 1764) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_13
  simp only [← c107871, ← c107872, k2, ← c107873] at e_1764_13
  have hr := e_1764_13
  simp only [e_1764_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1764 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17732)) := by
  have c107874 := (publicBatchWrapper89_copies3371 a h).2.2.1
  have c107875 := (publicBatchWrapper89_copies3371 a h).2.2.2.1
  have c107876 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.1
  have c107877 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.1
  have c107878 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.1
  have c107879 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1764_14 := arithEq_of_rows h (row := 1764) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_14
  simp only [← c107874, ← c107875, ← c107876] at e_1764_14
  have e_1764_15 := arithEq_of_rows h (row := 1764) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_15
  simp only [← c107877, ← c107878, k2, ← c107879] at e_1764_15
  have hr := e_1764_15
  simp only [e_1764_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1764 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17733)) := by
  have c107880 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.1
  have c107881 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.1
  have c107882 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107883 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107884 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107885 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1764_16 := arithEq_of_rows h (row := 1764) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_16
  simp only [← c107880, ← c107881, ← c107882] at e_1764_16
  have e_1764_17 := arithEq_of_rows h (row := 1764) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_17
  simp only [← c107883, ← c107884, k2, ← c107885] at e_1764_17
  have hr := e_1764_17
  simp only [e_1764_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1764 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17734)) := by
  have c107886 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107887 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107888 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107889 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107890 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107891 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1764_18 := arithEq_of_rows h (row := 1764) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_18
  simp only [← c107886, ← c107887, ← c107888] at e_1764_18
  have e_1764_19 := arithEq_of_rows h (row := 1764) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1764_19
  simp only [← c107889, ← c107890, k2, ← c107891] at e_1764_19
  have hr := e_1764_19
  simp only [e_1764_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1765 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17735)) := by
  have c107892 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107893 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107894 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107895 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107896 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107897 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1765_0 := arithEq_of_rows h (row := 1765) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_0
  simp only [← c107892, ← c107893, ← c107894] at e_1765_0
  have e_1765_1 := arithEq_of_rows h (row := 1765) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_1
  simp only [← c107895, ← c107896, k2, ← c107897] at e_1765_1
  have hr := e_1765_1
  simp only [e_1765_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1765 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17736)) := by
  have c107898 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107899 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107900 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107901 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107902 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107903 := (publicBatchWrapper89_copies3371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1765_2 := arithEq_of_rows h (row := 1765) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_2
  simp only [← c107898, ← c107899, ← c107900] at e_1765_2
  have e_1765_3 := arithEq_of_rows h (row := 1765) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_3
  simp only [← c107901, ← c107902, k2, ← c107903] at e_1765_3
  have hr := e_1765_3
  simp only [e_1765_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1765 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17737)) := by
  have c107904 := (publicBatchWrapper89_copies3372 a h).1
  have c107905 := (publicBatchWrapper89_copies3372 a h).2.1
  have c107906 := (publicBatchWrapper89_copies3372 a h).2.2.1
  have c107907 := (publicBatchWrapper89_copies3372 a h).2.2.2.1
  have c107908 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.1
  have c107909 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1765_4 := arithEq_of_rows h (row := 1765) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_4
  simp only [← c107904, ← c107905, ← c107906] at e_1765_4
  have e_1765_5 := arithEq_of_rows h (row := 1765) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_5
  simp only [← c107907, ← c107908, k2, ← c107909] at e_1765_5
  have hr := e_1765_5
  simp only [e_1765_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1765 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17738)) := by
  have c107910 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.1
  have c107911 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.1
  have c107912 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.1
  have c107913 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.1
  have c107914 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107915 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1765_6 := arithEq_of_rows h (row := 1765) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_6
  simp only [← c107910, ← c107911, ← c107912] at e_1765_6
  have e_1765_7 := arithEq_of_rows h (row := 1765) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_7
  simp only [← c107913, ← c107914, k2, ← c107915] at e_1765_7
  have hr := e_1765_7
  simp only [e_1765_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1765 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17739)) := by
  have c107916 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107917 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107918 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107919 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107920 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107921 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1765_8 := arithEq_of_rows h (row := 1765) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_8
  simp only [← c107916, ← c107917, ← c107918] at e_1765_8
  have e_1765_9 := arithEq_of_rows h (row := 1765) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_9
  simp only [← c107919, ← c107920, k2, ← c107921] at e_1765_9
  have hr := e_1765_9
  simp only [e_1765_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1765 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17740)) := by
  have c107922 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107923 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107924 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107925 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107926 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107927 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1765_10 := arithEq_of_rows h (row := 1765) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_10
  simp only [← c107922, ← c107923, ← c107924] at e_1765_10
  have e_1765_11 := arithEq_of_rows h (row := 1765) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_11
  simp only [← c107925, ← c107926, k2, ← c107927] at e_1765_11
  have hr := e_1765_11
  simp only [e_1765_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1765 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17741)) := by
  have c107928 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107929 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107930 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107931 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107932 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107933 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1765_12 := arithEq_of_rows h (row := 1765) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_12
  simp only [← c107928, ← c107929, ← c107930] at e_1765_12
  have e_1765_13 := arithEq_of_rows h (row := 1765) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_13
  simp only [← c107931, ← c107932, k2, ← c107933] at e_1765_13
  have hr := e_1765_13
  simp only [e_1765_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1765 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17742)) := by
  have c107934 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107935 := (publicBatchWrapper89_copies3372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107936 := (publicBatchWrapper89_copies3373 a h).1
  have c107937 := (publicBatchWrapper89_copies3373 a h).2.1
  have c107938 := (publicBatchWrapper89_copies3373 a h).2.2.1
  have c107939 := (publicBatchWrapper89_copies3373 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1765_14 := arithEq_of_rows h (row := 1765) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_14
  simp only [← c107934, ← c107935, ← c107936] at e_1765_14
  have e_1765_15 := arithEq_of_rows h (row := 1765) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_15
  simp only [← c107937, ← c107938, k2, ← c107939] at e_1765_15
  have hr := e_1765_15
  simp only [e_1765_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1765 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17743)) := by
  have c107940 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.1
  have c107941 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.1
  have c107942 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.1
  have c107943 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.1
  have c107944 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.1
  have c107945 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1765_16 := arithEq_of_rows h (row := 1765) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_16
  simp only [← c107940, ← c107941, ← c107942] at e_1765_16
  have e_1765_17 := arithEq_of_rows h (row := 1765) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_17
  simp only [← c107943, ← c107944, k2, ← c107945] at e_1765_17
  have hr := e_1765_17
  simp only [e_1765_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1765 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17744)) := by
  have c107946 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107947 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107948 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107949 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107950 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107951 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1765_18 := arithEq_of_rows h (row := 1765) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_18
  simp only [← c107946, ← c107947, ← c107948] at e_1765_18
  have e_1765_19 := arithEq_of_rows h (row := 1765) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1765_19
  simp only [← c107949, ← c107950, k2, ← c107951] at e_1765_19
  have hr := e_1765_19
  simp only [e_1765_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1766 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17745)) := by
  have c107952 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107953 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107954 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107955 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107956 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107957 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1766_0 := arithEq_of_rows h (row := 1766) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_0
  simp only [← c107952, ← c107953, ← c107954] at e_1766_0
  have e_1766_1 := arithEq_of_rows h (row := 1766) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_1
  simp only [← c107955, ← c107956, k2, ← c107957] at e_1766_1
  have hr := e_1766_1
  simp only [e_1766_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1766 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17746)) := by
  have c107958 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107959 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107960 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107961 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107962 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107963 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1766_2 := arithEq_of_rows h (row := 1766) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_2
  simp only [← c107958, ← c107959, ← c107960] at e_1766_2
  have e_1766_3 := arithEq_of_rows h (row := 1766) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_3
  simp only [← c107961, ← c107962, k2, ← c107963] at e_1766_3
  have hr := e_1766_3
  simp only [e_1766_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1766 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17747)) := by
  have c107964 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107965 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107966 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107967 := (publicBatchWrapper89_copies3373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c107968 := (publicBatchWrapper89_copies3374 a h).1
  have c107969 := (publicBatchWrapper89_copies3374 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1766_4 := arithEq_of_rows h (row := 1766) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_4
  simp only [← c107964, ← c107965, ← c107966] at e_1766_4
  have e_1766_5 := arithEq_of_rows h (row := 1766) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_5
  simp only [← c107967, ← c107968, k2, ← c107969] at e_1766_5
  have hr := e_1766_5
  simp only [e_1766_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1766 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17748)) := by
  have c107970 := (publicBatchWrapper89_copies3374 a h).2.2.1
  have c107971 := (publicBatchWrapper89_copies3374 a h).2.2.2.1
  have c107972 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.1
  have c107973 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.1
  have c107974 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.1
  have c107975 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1766_6 := arithEq_of_rows h (row := 1766) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_6
  simp only [← c107970, ← c107971, ← c107972] at e_1766_6
  have e_1766_7 := arithEq_of_rows h (row := 1766) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_7
  simp only [← c107973, ← c107974, k2, ← c107975] at e_1766_7
  have hr := e_1766_7
  simp only [e_1766_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1766 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17749)) := by
  have c107976 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.1
  have c107977 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.1
  have c107978 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.1
  have c107979 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c107980 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107981 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1766_8 := arithEq_of_rows h (row := 1766) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_8
  simp only [← c107976, ← c107977, ← c107978] at e_1766_8
  have e_1766_9 := arithEq_of_rows h (row := 1766) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_9
  simp only [← c107979, ← c107980, k2, ← c107981] at e_1766_9
  have hr := e_1766_9
  simp only [e_1766_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1766 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17750)) := by
  have c107982 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107983 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107984 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107985 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107986 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107987 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1766_10 := arithEq_of_rows h (row := 1766) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_10
  simp only [← c107982, ← c107983, ← c107984] at e_1766_10
  have e_1766_11 := arithEq_of_rows h (row := 1766) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_11
  simp only [← c107985, ← c107986, k2, ← c107987] at e_1766_11
  have hr := e_1766_11
  simp only [e_1766_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1766 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17751)) := by
  have c107988 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107989 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107990 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107991 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107992 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107993 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1766_12 := arithEq_of_rows h (row := 1766) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_12
  simp only [← c107988, ← c107989, ← c107990] at e_1766_12
  have e_1766_13 := arithEq_of_rows h (row := 1766) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_13
  simp only [← c107991, ← c107992, k2, ← c107993] at e_1766_13
  have hr := e_1766_13
  simp only [e_1766_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1766 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17752)) := by
  have c107994 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107995 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107996 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107997 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107998 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c107999 := (publicBatchWrapper89_copies3374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1766_14 := arithEq_of_rows h (row := 1766) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_14
  simp only [← c107994, ← c107995, ← c107996] at e_1766_14
  have e_1766_15 := arithEq_of_rows h (row := 1766) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_15
  simp only [← c107997, ← c107998, k2, ← c107999] at e_1766_15
  have hr := e_1766_15
  simp only [e_1766_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1766 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17753)) := by
  have c108000 := (publicBatchWrapper89_copies3375 a h).1
  have c108001 := (publicBatchWrapper89_copies3375 a h).2.1
  have c108002 := (publicBatchWrapper89_copies3375 a h).2.2.1
  have c108003 := (publicBatchWrapper89_copies3375 a h).2.2.2.1
  have c108004 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.1
  have c108005 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1766_16 := arithEq_of_rows h (row := 1766) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_16
  simp only [← c108000, ← c108001, ← c108002] at e_1766_16
  have e_1766_17 := arithEq_of_rows h (row := 1766) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_17
  simp only [← c108003, ← c108004, k2, ← c108005] at e_1766_17
  have hr := e_1766_17
  simp only [e_1766_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1766 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17754)) := by
  have c108006 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.1
  have c108007 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.1
  have c108008 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.1
  have c108009 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.1
  have c108010 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108011 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1766_18 := arithEq_of_rows h (row := 1766) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_18
  simp only [← c108006, ← c108007, ← c108008] at e_1766_18
  have e_1766_19 := arithEq_of_rows h (row := 1766) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1766_19
  simp only [← c108009, ← c108010, k2, ← c108011] at e_1766_19
  have hr := e_1766_19
  simp only [e_1766_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1767 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17755)) := by
  have c108012 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108013 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108014 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108015 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108016 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108017 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1767_0 := arithEq_of_rows h (row := 1767) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_0
  simp only [← c108012, ← c108013, ← c108014] at e_1767_0
  have e_1767_1 := arithEq_of_rows h (row := 1767) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_1
  simp only [← c108015, ← c108016, k2, ← c108017] at e_1767_1
  have hr := e_1767_1
  simp only [e_1767_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1767 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17756)) := by
  have c108018 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108019 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108020 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108021 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108022 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108023 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1767_2 := arithEq_of_rows h (row := 1767) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_2
  simp only [← c108018, ← c108019, ← c108020] at e_1767_2
  have e_1767_3 := arithEq_of_rows h (row := 1767) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_3
  simp only [← c108021, ← c108022, k2, ← c108023] at e_1767_3
  have hr := e_1767_3
  simp only [e_1767_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1767 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17757)) := by
  have c108024 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108025 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108026 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108027 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108028 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108029 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1767_4 := arithEq_of_rows h (row := 1767) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_4
  simp only [← c108024, ← c108025, ← c108026] at e_1767_4
  have e_1767_5 := arithEq_of_rows h (row := 1767) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_5
  simp only [← c108027, ← c108028, k2, ← c108029] at e_1767_5
  have hr := e_1767_5
  simp only [e_1767_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1767 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17758)) := by
  have c108030 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108031 := (publicBatchWrapper89_copies3375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108032 := (publicBatchWrapper89_copies3376 a h).1
  have c108033 := (publicBatchWrapper89_copies3376 a h).2.1
  have c108034 := (publicBatchWrapper89_copies3376 a h).2.2.1
  have c108035 := (publicBatchWrapper89_copies3376 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1767_6 := arithEq_of_rows h (row := 1767) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_6
  simp only [← c108030, ← c108031, ← c108032] at e_1767_6
  have e_1767_7 := arithEq_of_rows h (row := 1767) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_7
  simp only [← c108033, ← c108034, k2, ← c108035] at e_1767_7
  have hr := e_1767_7
  simp only [e_1767_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1767 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17759)) := by
  have c108036 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.1
  have c108037 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.1
  have c108038 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.1
  have c108039 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.1
  have c108040 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.1
  have c108041 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1767_8 := arithEq_of_rows h (row := 1767) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_8
  simp only [← c108036, ← c108037, ← c108038] at e_1767_8
  have e_1767_9 := arithEq_of_rows h (row := 1767) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_9
  simp only [← c108039, ← c108040, k2, ← c108041] at e_1767_9
  have hr := e_1767_9
  simp only [e_1767_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1767 47) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17760)) := by
  have c108042 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108043 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108044 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108045 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108046 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108047 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1767_10 := arithEq_of_rows h (row := 1767) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_10
  simp only [← c108042, ← c108043, ← c108044] at e_1767_10
  have e_1767_11 := arithEq_of_rows h (row := 1767) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_11
  simp only [← c108045, ← c108046, k2, ← c108047] at e_1767_11
  have hr := e_1767_11
  simp only [e_1767_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1767 55) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17761)) := by
  have c108048 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108049 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108050 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108051 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108052 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108053 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1767_12 := arithEq_of_rows h (row := 1767) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_12
  simp only [← c108048, ← c108049, ← c108050] at e_1767_12
  have e_1767_13 := arithEq_of_rows h (row := 1767) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_13
  simp only [← c108051, ← c108052, k2, ← c108053] at e_1767_13
  have hr := e_1767_13
  simp only [e_1767_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1767 63) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17762)) := by
  have c108054 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108055 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108056 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108057 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108058 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108059 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1767_14 := arithEq_of_rows h (row := 1767) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_14
  simp only [← c108054, ← c108055, ← c108056] at e_1767_14
  have e_1767_15 := arithEq_of_rows h (row := 1767) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_15
  simp only [← c108057, ← c108058, k2, ← c108059] at e_1767_15
  have hr := e_1767_15
  simp only [e_1767_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1767 71) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17763)) := by
  have c108060 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108061 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108062 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108063 := (publicBatchWrapper89_copies3376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108064 := (publicBatchWrapper89_copies3377 a h).1
  have c108065 := (publicBatchWrapper89_copies3377 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1767_16 := arithEq_of_rows h (row := 1767) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_16
  simp only [← c108060, ← c108061, ← c108062] at e_1767_16
  have e_1767_17 := arithEq_of_rows h (row := 1767) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_17
  simp only [← c108063, ← c108064, k2, ← c108065] at e_1767_17
  have hr := e_1767_17
  simp only [e_1767_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1767 79) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17764)) := by
  have c108066 := (publicBatchWrapper89_copies3377 a h).2.2.1
  have c108067 := (publicBatchWrapper89_copies3377 a h).2.2.2.1
  have c108068 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.1
  have c108069 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.1
  have c108070 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.1
  have c108071 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1767_18 := arithEq_of_rows h (row := 1767) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_18
  simp only [← c108066, ← c108067, ← c108068] at e_1767_18
  have e_1767_19 := arithEq_of_rows h (row := 1767) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1767_19
  simp only [← c108069, ← c108070, k2, ← c108071] at e_1767_19
  have hr := e_1767_19
  simp only [e_1767_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1768 7) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17765)) := by
  have c108072 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.1
  have c108073 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.1
  have c108074 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108075 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108076 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108077 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1768_0 := arithEq_of_rows h (row := 1768) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_0
  simp only [← c108072, ← c108073, ← c108074] at e_1768_0
  have e_1768_1 := arithEq_of_rows h (row := 1768) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_1
  simp only [← c108075, ← c108076, k2, ← c108077] at e_1768_1
  have hr := e_1768_1
  simp only [e_1768_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1768 15) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17766)) := by
  have c108078 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108079 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108080 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108081 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108082 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108083 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1768_2 := arithEq_of_rows h (row := 1768) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_2
  simp only [← c108078, ← c108079, ← c108080] at e_1768_2
  have e_1768_3 := arithEq_of_rows h (row := 1768) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_3
  simp only [← c108081, ← c108082, k2, ← c108083] at e_1768_3
  have hr := e_1768_3
  simp only [e_1768_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1768 23) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17767)) := by
  have c108084 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108085 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108086 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108087 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108088 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108089 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1768_4 := arithEq_of_rows h (row := 1768) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_4
  simp only [← c108084, ← c108085, ← c108086] at e_1768_4
  have e_1768_5 := arithEq_of_rows h (row := 1768) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_5
  simp only [← c108087, ← c108088, k2, ← c108089] at e_1768_5
  have hr := e_1768_5
  simp only [e_1768_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1768 31) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17768)) := by
  have c108090 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108091 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108092 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108093 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108094 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108095 := (publicBatchWrapper89_copies3377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1768_6 := arithEq_of_rows h (row := 1768) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_6
  simp only [← c108090, ← c108091, ← c108092] at e_1768_6
  have e_1768_7 := arithEq_of_rows h (row := 1768) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_7
  simp only [← c108093, ← c108094, k2, ← c108095] at e_1768_7
  have hr := e_1768_7
  simp only [e_1768_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1768 39) = bselect (a (.wire 86 11)) (a (.virt 200)) (a (.virt 17769)) := by
  have c108096 := (publicBatchWrapper89_copies3378 a h).1
  have c108097 := (publicBatchWrapper89_copies3378 a h).2.1
  have c108098 := (publicBatchWrapper89_copies3378 a h).2.2.1
  have c108099 := (publicBatchWrapper89_copies3378 a h).2.2.2.1
  have c108100 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.1
  have c108101 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1768_8 := arithEq_of_rows h (row := 1768) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_8
  simp only [← c108096, ← c108097, ← c108098] at e_1768_8
  have e_1768_9 := arithEq_of_rows h (row := 1768) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_9
  simp only [← c108099, ← c108100, k2, ← c108101] at e_1768_9
  have hr := e_1768_9
  simp only [e_1768_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1768 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 168)) := by
  have c108102 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.1
  have c108103 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.1
  have c108104 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.1
  have c108105 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.1
  have c108106 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108107 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1768_10 := arithEq_of_rows h (row := 1768) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_10
  simp only [← c108102, ← c108103, ← c108104] at e_1768_10
  have e_1768_11 := arithEq_of_rows h (row := 1768) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_11
  simp only [← c108105, ← c108106, k2, ← c108107] at e_1768_11
  have hr := e_1768_11
  simp only [e_1768_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1768 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 169)) := by
  have c108108 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108109 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108110 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108111 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108112 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108113 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1768_12 := arithEq_of_rows h (row := 1768) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_12
  simp only [← c108108, ← c108109, ← c108110] at e_1768_12
  have e_1768_13 := arithEq_of_rows h (row := 1768) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_13
  simp only [← c108111, ← c108112, k2, ← c108113] at e_1768_13
  have hr := e_1768_13
  simp only [e_1768_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1768 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 170)) := by
  have c108114 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108115 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108116 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108117 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108118 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108119 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1768_14 := arithEq_of_rows h (row := 1768) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_14
  simp only [← c108114, ← c108115, ← c108116] at e_1768_14
  have e_1768_15 := arithEq_of_rows h (row := 1768) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_15
  simp only [← c108117, ← c108118, k2, ← c108119] at e_1768_15
  have hr := e_1768_15
  simp only [e_1768_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1768 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 171)) := by
  have c108120 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108121 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108122 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108123 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108124 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108125 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1768_16 := arithEq_of_rows h (row := 1768) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_16
  simp only [← c108120, ← c108121, ← c108122] at e_1768_16
  have e_1768_17 := arithEq_of_rows h (row := 1768) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_17
  simp only [← c108123, ← c108124, k2, ← c108125] at e_1768_17
  have hr := e_1768_17
  simp only [e_1768_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1768 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 172)) := by
  have c108126 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108127 := (publicBatchWrapper89_copies3378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108128 := (publicBatchWrapper89_copies3379 a h).1
  have c108129 := (publicBatchWrapper89_copies3379 a h).2.1
  have c108130 := (publicBatchWrapper89_copies3379 a h).2.2.1
  have c108131 := (publicBatchWrapper89_copies3379 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1768_18 := arithEq_of_rows h (row := 1768) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_18
  simp only [← c108126, ← c108127, ← c108128] at e_1768_18
  have e_1768_19 := arithEq_of_rows h (row := 1768) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1768_19
  simp only [← c108129, ← c108130, k2, ← c108131] at e_1768_19
  have hr := e_1768_19
  simp only [e_1768_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1769 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 173)) := by
  have c108132 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.1
  have c108133 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.1
  have c108134 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.1
  have c108135 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.1
  have c108136 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.1
  have c108137 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1769_0 := arithEq_of_rows h (row := 1769) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_0
  simp only [← c108132, ← c108133, ← c108134] at e_1769_0
  have e_1769_1 := arithEq_of_rows h (row := 1769) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_1
  simp only [← c108135, ← c108136, k2, ← c108137] at e_1769_1
  have hr := e_1769_1
  simp only [e_1769_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1769 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 174)) := by
  have c108138 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108139 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108140 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108141 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108142 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108143 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1769_2 := arithEq_of_rows h (row := 1769) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_2
  simp only [← c108138, ← c108139, ← c108140] at e_1769_2
  have e_1769_3 := arithEq_of_rows h (row := 1769) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_3
  simp only [← c108141, ← c108142, k2, ← c108143] at e_1769_3
  have hr := e_1769_3
  simp only [e_1769_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1769 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 175)) := by
  have c108144 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108145 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108146 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108147 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108148 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108149 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1769_4 := arithEq_of_rows h (row := 1769) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_4
  simp only [← c108144, ← c108145, ← c108146] at e_1769_4
  have e_1769_5 := arithEq_of_rows h (row := 1769) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_5
  simp only [← c108147, ← c108148, k2, ← c108149] at e_1769_5
  have hr := e_1769_5
  simp only [e_1769_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1769 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 176)) := by
  have c108150 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108151 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108152 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108153 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108154 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108155 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1769_6 := arithEq_of_rows h (row := 1769) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_6
  simp only [← c108150, ← c108151, ← c108152] at e_1769_6
  have e_1769_7 := arithEq_of_rows h (row := 1769) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_7
  simp only [← c108153, ← c108154, k2, ← c108155] at e_1769_7
  have hr := e_1769_7
  simp only [e_1769_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1769 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 177)) := by
  have c108156 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108157 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108158 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108159 := (publicBatchWrapper89_copies3379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108160 := (publicBatchWrapper89_copies3380 a h).1
  have c108161 := (publicBatchWrapper89_copies3380 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1769_8 := arithEq_of_rows h (row := 1769) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_8
  simp only [← c108156, ← c108157, ← c108158] at e_1769_8
  have e_1769_9 := arithEq_of_rows h (row := 1769) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_9
  simp only [← c108159, ← c108160, k2, ← c108161] at e_1769_9
  have hr := e_1769_9
  simp only [e_1769_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1769 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 178)) := by
  have c108162 := (publicBatchWrapper89_copies3380 a h).2.2.1
  have c108163 := (publicBatchWrapper89_copies3380 a h).2.2.2.1
  have c108164 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.1
  have c108165 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.1
  have c108166 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.1
  have c108167 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1769_10 := arithEq_of_rows h (row := 1769) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_10
  simp only [← c108162, ← c108163, ← c108164] at e_1769_10
  have e_1769_11 := arithEq_of_rows h (row := 1769) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_11
  simp only [← c108165, ← c108166, k2, ← c108167] at e_1769_11
  have hr := e_1769_11
  simp only [e_1769_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1769 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 179)) := by
  have c108168 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.1
  have c108169 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.1
  have c108170 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108171 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108172 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108173 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1769_12 := arithEq_of_rows h (row := 1769) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_12
  simp only [← c108168, ← c108169, ← c108170] at e_1769_12
  have e_1769_13 := arithEq_of_rows h (row := 1769) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_13
  simp only [← c108171, ← c108172, k2, ← c108173] at e_1769_13
  have hr := e_1769_13
  simp only [e_1769_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1769 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 180)) := by
  have c108174 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108175 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108176 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108177 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108178 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108179 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1769_14 := arithEq_of_rows h (row := 1769) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_14
  simp only [← c108174, ← c108175, ← c108176] at e_1769_14
  have e_1769_15 := arithEq_of_rows h (row := 1769) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_15
  simp only [← c108177, ← c108178, k2, ← c108179] at e_1769_15
  have hr := e_1769_15
  simp only [e_1769_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1769 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 181)) := by
  have c108180 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108181 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108182 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108183 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108184 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108185 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1769_16 := arithEq_of_rows h (row := 1769) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_16
  simp only [← c108180, ← c108181, ← c108182] at e_1769_16
  have e_1769_17 := arithEq_of_rows h (row := 1769) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_17
  simp only [← c108183, ← c108184, k2, ← c108185] at e_1769_17
  have hr := e_1769_17
  simp only [e_1769_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1769 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 182)) := by
  have c108186 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108187 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108188 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108189 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108190 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108191 := (publicBatchWrapper89_copies3380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1769_18 := arithEq_of_rows h (row := 1769) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_18
  simp only [← c108186, ← c108187, ← c108188] at e_1769_18
  have e_1769_19 := arithEq_of_rows h (row := 1769) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1769_19
  simp only [← c108189, ← c108190, k2, ← c108191] at e_1769_19
  have hr := e_1769_19
  simp only [e_1769_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1770 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 183)) := by
  have c108192 := (publicBatchWrapper89_copies3381 a h).1
  have c108193 := (publicBatchWrapper89_copies3381 a h).2.1
  have c108194 := (publicBatchWrapper89_copies3381 a h).2.2.1
  have c108195 := (publicBatchWrapper89_copies3381 a h).2.2.2.1
  have c108196 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.1
  have c108197 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1770_0 := arithEq_of_rows h (row := 1770) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_0
  simp only [← c108192, ← c108193, ← c108194] at e_1770_0
  have e_1770_1 := arithEq_of_rows h (row := 1770) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_1
  simp only [← c108195, ← c108196, k2, ← c108197] at e_1770_1
  have hr := e_1770_1
  simp only [e_1770_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1770 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 184)) := by
  have c108198 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.1
  have c108199 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.1
  have c108200 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.1
  have c108201 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.1
  have c108202 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108203 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1770_2 := arithEq_of_rows h (row := 1770) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_2
  simp only [← c108198, ← c108199, ← c108200] at e_1770_2
  have e_1770_3 := arithEq_of_rows h (row := 1770) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_3
  simp only [← c108201, ← c108202, k2, ← c108203] at e_1770_3
  have hr := e_1770_3
  simp only [e_1770_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1770 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 185)) := by
  have c108204 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108205 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108206 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108207 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108208 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108209 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1770_4 := arithEq_of_rows h (row := 1770) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_4
  simp only [← c108204, ← c108205, ← c108206] at e_1770_4
  have e_1770_5 := arithEq_of_rows h (row := 1770) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_5
  simp only [← c108207, ← c108208, k2, ← c108209] at e_1770_5
  have hr := e_1770_5
  simp only [e_1770_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1770 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 186)) := by
  have c108210 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108211 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108212 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108213 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108214 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108215 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1770_6 := arithEq_of_rows h (row := 1770) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_6
  simp only [← c108210, ← c108211, ← c108212] at e_1770_6
  have e_1770_7 := arithEq_of_rows h (row := 1770) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_7
  simp only [← c108213, ← c108214, k2, ← c108215] at e_1770_7
  have hr := e_1770_7
  simp only [e_1770_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1770 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 187)) := by
  have c108216 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108217 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108218 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108219 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108220 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108221 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1770_8 := arithEq_of_rows h (row := 1770) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_8
  simp only [← c108216, ← c108217, ← c108218] at e_1770_8
  have e_1770_9 := arithEq_of_rows h (row := 1770) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_9
  simp only [← c108219, ← c108220, k2, ← c108221] at e_1770_9
  have hr := e_1770_9
  simp only [e_1770_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1770 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 188)) := by
  have c108222 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108223 := (publicBatchWrapper89_copies3381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108224 := (publicBatchWrapper89_copies3382 a h).1
  have c108225 := (publicBatchWrapper89_copies3382 a h).2.1
  have c108226 := (publicBatchWrapper89_copies3382 a h).2.2.1
  have c108227 := (publicBatchWrapper89_copies3382 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1770_10 := arithEq_of_rows h (row := 1770) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_10
  simp only [← c108222, ← c108223, ← c108224] at e_1770_10
  have e_1770_11 := arithEq_of_rows h (row := 1770) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_11
  simp only [← c108225, ← c108226, k2, ← c108227] at e_1770_11
  have hr := e_1770_11
  simp only [e_1770_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1770 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 189)) := by
  have c108228 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.1
  have c108229 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.1
  have c108230 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.1
  have c108231 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.1
  have c108232 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.1
  have c108233 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1770_12 := arithEq_of_rows h (row := 1770) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_12
  simp only [← c108228, ← c108229, ← c108230] at e_1770_12
  have e_1770_13 := arithEq_of_rows h (row := 1770) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_13
  simp only [← c108231, ← c108232, k2, ← c108233] at e_1770_13
  have hr := e_1770_13
  simp only [e_1770_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1770 63) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 190)) := by
  have c108234 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108235 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108236 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108237 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108238 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108239 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1770_14 := arithEq_of_rows h (row := 1770) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_14
  simp only [← c108234, ← c108235, ← c108236] at e_1770_14
  have e_1770_15 := arithEq_of_rows h (row := 1770) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_15
  simp only [← c108237, ← c108238, k2, ← c108239] at e_1770_15
  have hr := e_1770_15
  simp only [e_1770_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1770 71) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 191)) := by
  have c108240 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108241 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108242 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108243 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108244 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108245 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1770_16 := arithEq_of_rows h (row := 1770) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_16
  simp only [← c108240, ← c108241, ← c108242] at e_1770_16
  have e_1770_17 := arithEq_of_rows h (row := 1770) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_17
  simp only [← c108243, ← c108244, k2, ← c108245] at e_1770_17
  have hr := e_1770_17
  simp only [e_1770_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1770 79) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 192)) := by
  have c108246 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108247 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108248 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108249 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108250 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108251 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1770_18 := arithEq_of_rows h (row := 1770) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_18
  simp only [← c108246, ← c108247, ← c108248] at e_1770_18
  have e_1770_19 := arithEq_of_rows h (row := 1770) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1770_19
  simp only [← c108249, ← c108250, k2, ← c108251] at e_1770_19
  have hr := e_1770_19
  simp only [e_1770_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1771 7) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 193)) := by
  have c108252 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108253 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108254 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108255 := (publicBatchWrapper89_copies3382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108256 := (publicBatchWrapper89_copies3383 a h).1
  have c108257 := (publicBatchWrapper89_copies3383 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1771_0 := arithEq_of_rows h (row := 1771) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_0
  simp only [← c108252, ← c108253, ← c108254] at e_1771_0
  have e_1771_1 := arithEq_of_rows h (row := 1771) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_1
  simp only [← c108255, ← c108256, k2, ← c108257] at e_1771_1
  have hr := e_1771_1
  simp only [e_1771_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1771 15) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 194)) := by
  have c108258 := (publicBatchWrapper89_copies3383 a h).2.2.1
  have c108259 := (publicBatchWrapper89_copies3383 a h).2.2.2.1
  have c108260 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.1
  have c108261 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.1
  have c108262 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.1
  have c108263 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1771_2 := arithEq_of_rows h (row := 1771) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_2
  simp only [← c108258, ← c108259, ← c108260] at e_1771_2
  have e_1771_3 := arithEq_of_rows h (row := 1771) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_3
  simp only [← c108261, ← c108262, k2, ← c108263] at e_1771_3
  have hr := e_1771_3
  simp only [e_1771_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1771 23) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 195)) := by
  have c108264 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.1
  have c108265 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.1
  have c108266 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108267 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108268 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108269 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1771_4 := arithEq_of_rows h (row := 1771) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_4
  simp only [← c108264, ← c108265, ← c108266] at e_1771_4
  have e_1771_5 := arithEq_of_rows h (row := 1771) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_5
  simp only [← c108267, ← c108268, k2, ← c108269] at e_1771_5
  have hr := e_1771_5
  simp only [e_1771_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1771 31) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 196)) := by
  have c108270 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108271 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108272 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108273 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108274 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108275 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1771_6 := arithEq_of_rows h (row := 1771) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_6
  simp only [← c108270, ← c108271, ← c108272] at e_1771_6
  have e_1771_7 := arithEq_of_rows h (row := 1771) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_7
  simp only [← c108273, ← c108274, k2, ← c108275] at e_1771_7
  have hr := e_1771_7
  simp only [e_1771_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1771 39) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 197)) := by
  have c108276 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108277 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108278 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108279 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108280 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108281 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1771_8 := arithEq_of_rows h (row := 1771) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_8
  simp only [← c108276, ← c108277, ← c108278] at e_1771_8
  have e_1771_9 := arithEq_of_rows h (row := 1771) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_9
  simp only [← c108279, ← c108280, k2, ← c108281] at e_1771_9
  have hr := e_1771_9
  simp only [e_1771_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1771 47) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 198)) := by
  have c108282 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108283 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108284 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108285 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108286 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108287 := (publicBatchWrapper89_copies3383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1771_10 := arithEq_of_rows h (row := 1771) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_10
  simp only [← c108282, ← c108283, ← c108284] at e_1771_10
  have e_1771_11 := arithEq_of_rows h (row := 1771) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_11
  simp only [← c108285, ← c108286, k2, ← c108287] at e_1771_11
  have hr := e_1771_11
  simp only [e_1771_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1771 55) = bselect (a (.wire 1 59)) (a (.virt 200)) (a (.virt 199)) := by
  have c108288 := (publicBatchWrapper89_copies3384 a h).1
  have c108289 := (publicBatchWrapper89_copies3384 a h).2.1
  have c108290 := (publicBatchWrapper89_copies3384 a h).2.2.1
  have c108291 := (publicBatchWrapper89_copies3384 a h).2.2.2.1
  have c108292 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.1
  have c108293 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1771_12 := arithEq_of_rows h (row := 1771) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_12
  simp only [← c108288, ← c108289, ← c108290] at e_1771_12
  have e_1771_13 := arithEq_of_rows h (row := 1771) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_13
  simp only [← c108291, ← c108292, k2, ← c108293] at e_1771_13
  have hr := e_1771_13
  simp only [e_1771_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1771 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 370)) := by
  have c108294 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.1
  have c108295 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.1
  have c108296 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.1
  have c108297 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.1
  have c108298 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108299 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1771_14 := arithEq_of_rows h (row := 1771) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_14
  simp only [← c108294, ← c108295, ← c108296] at e_1771_14
  have e_1771_15 := arithEq_of_rows h (row := 1771) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_15
  simp only [← c108297, ← c108298, k2, ← c108299] at e_1771_15
  have hr := e_1771_15
  simp only [e_1771_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1771 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 371)) := by
  have c108300 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108301 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108302 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108303 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108304 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108305 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1771_16 := arithEq_of_rows h (row := 1771) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_16
  simp only [← c108300, ← c108301, ← c108302] at e_1771_16
  have e_1771_17 := arithEq_of_rows h (row := 1771) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_17
  simp only [← c108303, ← c108304, k2, ← c108305] at e_1771_17
  have hr := e_1771_17
  simp only [e_1771_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1771 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 372)) := by
  have c108306 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108307 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108308 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108309 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108310 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108311 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1771_18 := arithEq_of_rows h (row := 1771) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_18
  simp only [← c108306, ← c108307, ← c108308] at e_1771_18
  have e_1771_19 := arithEq_of_rows h (row := 1771) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1771_19
  simp only [← c108309, ← c108310, k2, ← c108311] at e_1771_19
  have hr := e_1771_19
  simp only [e_1771_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1772 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 373)) := by
  have c108312 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108313 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108314 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108315 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108316 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108317 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1772_0 := arithEq_of_rows h (row := 1772) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_0
  simp only [← c108312, ← c108313, ← c108314] at e_1772_0
  have e_1772_1 := arithEq_of_rows h (row := 1772) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_1
  simp only [← c108315, ← c108316, k2, ← c108317] at e_1772_1
  have hr := e_1772_1
  simp only [e_1772_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1772 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 374)) := by
  have c108318 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108319 := (publicBatchWrapper89_copies3384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108320 := (publicBatchWrapper89_copies3385 a h).1
  have c108321 := (publicBatchWrapper89_copies3385 a h).2.1
  have c108322 := (publicBatchWrapper89_copies3385 a h).2.2.1
  have c108323 := (publicBatchWrapper89_copies3385 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1772_2 := arithEq_of_rows h (row := 1772) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_2
  simp only [← c108318, ← c108319, ← c108320] at e_1772_2
  have e_1772_3 := arithEq_of_rows h (row := 1772) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_3
  simp only [← c108321, ← c108322, k2, ← c108323] at e_1772_3
  have hr := e_1772_3
  simp only [e_1772_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1772 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 375)) := by
  have c108324 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.1
  have c108325 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.1
  have c108326 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.1
  have c108327 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.1
  have c108328 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.1
  have c108329 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1772_4 := arithEq_of_rows h (row := 1772) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_4
  simp only [← c108324, ← c108325, ← c108326] at e_1772_4
  have e_1772_5 := arithEq_of_rows h (row := 1772) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_5
  simp only [← c108327, ← c108328, k2, ← c108329] at e_1772_5
  have hr := e_1772_5
  simp only [e_1772_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1772 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 376)) := by
  have c108330 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108331 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108332 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108333 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108334 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108335 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1772_6 := arithEq_of_rows h (row := 1772) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_6
  simp only [← c108330, ← c108331, ← c108332] at e_1772_6
  have e_1772_7 := arithEq_of_rows h (row := 1772) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_7
  simp only [← c108333, ← c108334, k2, ← c108335] at e_1772_7
  have hr := e_1772_7
  simp only [e_1772_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1772 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 377)) := by
  have c108336 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108337 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108338 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108339 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108340 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108341 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1772_8 := arithEq_of_rows h (row := 1772) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_8
  simp only [← c108336, ← c108337, ← c108338] at e_1772_8
  have e_1772_9 := arithEq_of_rows h (row := 1772) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_9
  simp only [← c108339, ← c108340, k2, ← c108341] at e_1772_9
  have hr := e_1772_9
  simp only [e_1772_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1772 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 378)) := by
  have c108342 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108343 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108344 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108345 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108346 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108347 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1772_10 := arithEq_of_rows h (row := 1772) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_10
  simp only [← c108342, ← c108343, ← c108344] at e_1772_10
  have e_1772_11 := arithEq_of_rows h (row := 1772) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_11
  simp only [← c108345, ← c108346, k2, ← c108347] at e_1772_11
  have hr := e_1772_11
  simp only [e_1772_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1772 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 379)) := by
  have c108348 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108349 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108350 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108351 := (publicBatchWrapper89_copies3385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108352 := (publicBatchWrapper89_copies3386 a h).1
  have c108353 := (publicBatchWrapper89_copies3386 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1772_12 := arithEq_of_rows h (row := 1772) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_12
  simp only [← c108348, ← c108349, ← c108350] at e_1772_12
  have e_1772_13 := arithEq_of_rows h (row := 1772) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_13
  simp only [← c108351, ← c108352, k2, ← c108353] at e_1772_13
  have hr := e_1772_13
  simp only [e_1772_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1772 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 380)) := by
  have c108354 := (publicBatchWrapper89_copies3386 a h).2.2.1
  have c108355 := (publicBatchWrapper89_copies3386 a h).2.2.2.1
  have c108356 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.1
  have c108357 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.1
  have c108358 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.1
  have c108359 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1772_14 := arithEq_of_rows h (row := 1772) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_14
  simp only [← c108354, ← c108355, ← c108356] at e_1772_14
  have e_1772_15 := arithEq_of_rows h (row := 1772) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_15
  simp only [← c108357, ← c108358, k2, ← c108359] at e_1772_15
  have hr := e_1772_15
  simp only [e_1772_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1772 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 381)) := by
  have c108360 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.1
  have c108361 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.1
  have c108362 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108363 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108364 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108365 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1772_16 := arithEq_of_rows h (row := 1772) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_16
  simp only [← c108360, ← c108361, ← c108362] at e_1772_16
  have e_1772_17 := arithEq_of_rows h (row := 1772) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_17
  simp only [← c108363, ← c108364, k2, ← c108365] at e_1772_17
  have hr := e_1772_17
  simp only [e_1772_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1772 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 382)) := by
  have c108366 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108367 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108368 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108369 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108370 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108371 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1772_18 := arithEq_of_rows h (row := 1772) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_18
  simp only [← c108366, ← c108367, ← c108368] at e_1772_18
  have e_1772_19 := arithEq_of_rows h (row := 1772) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1772_19
  simp only [← c108369, ← c108370, k2, ← c108371] at e_1772_19
  have hr := e_1772_19
  simp only [e_1772_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1773 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 383)) := by
  have c108372 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108373 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108374 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108375 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108376 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108377 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1773_0 := arithEq_of_rows h (row := 1773) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_0
  simp only [← c108372, ← c108373, ← c108374] at e_1773_0
  have e_1773_1 := arithEq_of_rows h (row := 1773) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_1
  simp only [← c108375, ← c108376, k2, ← c108377] at e_1773_1
  have hr := e_1773_1
  simp only [e_1773_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1773 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 384)) := by
  have c108378 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108379 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108380 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108381 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108382 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108383 := (publicBatchWrapper89_copies3386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1773_2 := arithEq_of_rows h (row := 1773) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_2
  simp only [← c108378, ← c108379, ← c108380] at e_1773_2
  have e_1773_3 := arithEq_of_rows h (row := 1773) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_3
  simp only [← c108381, ← c108382, k2, ← c108383] at e_1773_3
  have hr := e_1773_3
  simp only [e_1773_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1773 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 385)) := by
  have c108384 := (publicBatchWrapper89_copies3387 a h).1
  have c108385 := (publicBatchWrapper89_copies3387 a h).2.1
  have c108386 := (publicBatchWrapper89_copies3387 a h).2.2.1
  have c108387 := (publicBatchWrapper89_copies3387 a h).2.2.2.1
  have c108388 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.1
  have c108389 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1773_4 := arithEq_of_rows h (row := 1773) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_4
  simp only [← c108384, ← c108385, ← c108386] at e_1773_4
  have e_1773_5 := arithEq_of_rows h (row := 1773) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_5
  simp only [← c108387, ← c108388, k2, ← c108389] at e_1773_5
  have hr := e_1773_5
  simp only [e_1773_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1773 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 386)) := by
  have c108390 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.1
  have c108391 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.1
  have c108392 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.1
  have c108393 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.1
  have c108394 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108395 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1773_6 := arithEq_of_rows h (row := 1773) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_6
  simp only [← c108390, ← c108391, ← c108392] at e_1773_6
  have e_1773_7 := arithEq_of_rows h (row := 1773) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_7
  simp only [← c108393, ← c108394, k2, ← c108395] at e_1773_7
  have hr := e_1773_7
  simp only [e_1773_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1773 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 387)) := by
  have c108396 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108397 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108398 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108399 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108400 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108401 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1773_8 := arithEq_of_rows h (row := 1773) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_8
  simp only [← c108396, ← c108397, ← c108398] at e_1773_8
  have e_1773_9 := arithEq_of_rows h (row := 1773) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_9
  simp only [← c108399, ← c108400, k2, ← c108401] at e_1773_9
  have hr := e_1773_9
  simp only [e_1773_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1773 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 388)) := by
  have c108402 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108403 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108404 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108405 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108406 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108407 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1773_10 := arithEq_of_rows h (row := 1773) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_10
  simp only [← c108402, ← c108403, ← c108404] at e_1773_10
  have e_1773_11 := arithEq_of_rows h (row := 1773) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_11
  simp only [← c108405, ← c108406, k2, ← c108407] at e_1773_11
  have hr := e_1773_11
  simp only [e_1773_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1773 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 389)) := by
  have c108408 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108409 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108410 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108411 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108412 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108413 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1773_12 := arithEq_of_rows h (row := 1773) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_12
  simp only [← c108408, ← c108409, ← c108410] at e_1773_12
  have e_1773_13 := arithEq_of_rows h (row := 1773) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_13
  simp only [← c108411, ← c108412, k2, ← c108413] at e_1773_13
  have hr := e_1773_13
  simp only [e_1773_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1773 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 390)) := by
  have c108414 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108415 := (publicBatchWrapper89_copies3387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108416 := (publicBatchWrapper89_copies3388 a h).1
  have c108417 := (publicBatchWrapper89_copies3388 a h).2.1
  have c108418 := (publicBatchWrapper89_copies3388 a h).2.2.1
  have c108419 := (publicBatchWrapper89_copies3388 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1773_14 := arithEq_of_rows h (row := 1773) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_14
  simp only [← c108414, ← c108415, ← c108416] at e_1773_14
  have e_1773_15 := arithEq_of_rows h (row := 1773) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_15
  simp only [← c108417, ← c108418, k2, ← c108419] at e_1773_15
  have hr := e_1773_15
  simp only [e_1773_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1773 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 391)) := by
  have c108420 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.1
  have c108421 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.1
  have c108422 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.1
  have c108423 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.1
  have c108424 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.1
  have c108425 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1773_16 := arithEq_of_rows h (row := 1773) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_16
  simp only [← c108420, ← c108421, ← c108422] at e_1773_16
  have e_1773_17 := arithEq_of_rows h (row := 1773) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_17
  simp only [← c108423, ← c108424, k2, ← c108425] at e_1773_17
  have hr := e_1773_17
  simp only [e_1773_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1773 79) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 392)) := by
  have c108426 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108427 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108428 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108429 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108430 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108431 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1773_18 := arithEq_of_rows h (row := 1773) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_18
  simp only [← c108426, ← c108427, ← c108428] at e_1773_18
  have e_1773_19 := arithEq_of_rows h (row := 1773) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1773_19
  simp only [← c108429, ← c108430, k2, ← c108431] at e_1773_19
  have hr := e_1773_19
  simp only [e_1773_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1774 7) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 393)) := by
  have c108432 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108433 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108434 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108435 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108436 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108437 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1774_0 := arithEq_of_rows h (row := 1774) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_0
  simp only [← c108432, ← c108433, ← c108434] at e_1774_0
  have e_1774_1 := arithEq_of_rows h (row := 1774) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_1
  simp only [← c108435, ← c108436, k2, ← c108437] at e_1774_1
  have hr := e_1774_1
  simp only [e_1774_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1774 15) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 394)) := by
  have c108438 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108439 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108440 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108441 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108442 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108443 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1774_2 := arithEq_of_rows h (row := 1774) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_2
  simp only [← c108438, ← c108439, ← c108440] at e_1774_2
  have e_1774_3 := arithEq_of_rows h (row := 1774) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_3
  simp only [← c108441, ← c108442, k2, ← c108443] at e_1774_3
  have hr := e_1774_3
  simp only [e_1774_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1774 23) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 395)) := by
  have c108444 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108445 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108446 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108447 := (publicBatchWrapper89_copies3388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108448 := (publicBatchWrapper89_copies3389 a h).1
  have c108449 := (publicBatchWrapper89_copies3389 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1774_4 := arithEq_of_rows h (row := 1774) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_4
  simp only [← c108444, ← c108445, ← c108446] at e_1774_4
  have e_1774_5 := arithEq_of_rows h (row := 1774) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_5
  simp only [← c108447, ← c108448, k2, ← c108449] at e_1774_5
  have hr := e_1774_5
  simp only [e_1774_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1774 31) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 396)) := by
  have c108450 := (publicBatchWrapper89_copies3389 a h).2.2.1
  have c108451 := (publicBatchWrapper89_copies3389 a h).2.2.2.1
  have c108452 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.1
  have c108453 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.1
  have c108454 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.1
  have c108455 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1774_6 := arithEq_of_rows h (row := 1774) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_6
  simp only [← c108450, ← c108451, ← c108452] at e_1774_6
  have e_1774_7 := arithEq_of_rows h (row := 1774) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_7
  simp only [← c108453, ← c108454, k2, ← c108455] at e_1774_7
  have hr := e_1774_7
  simp only [e_1774_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1774 39) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 397)) := by
  have c108456 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.1
  have c108457 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.1
  have c108458 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108459 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108460 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108461 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1774_8 := arithEq_of_rows h (row := 1774) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_8
  simp only [← c108456, ← c108457, ← c108458] at e_1774_8
  have e_1774_9 := arithEq_of_rows h (row := 1774) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_9
  simp only [← c108459, ← c108460, k2, ← c108461] at e_1774_9
  have hr := e_1774_9
  simp only [e_1774_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1774 47) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 398)) := by
  have c108462 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108463 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108464 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108465 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108466 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108467 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1774_10 := arithEq_of_rows h (row := 1774) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_10
  simp only [← c108462, ← c108463, ← c108464] at e_1774_10
  have e_1774_11 := arithEq_of_rows h (row := 1774) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_11
  simp only [← c108465, ← c108466, k2, ← c108467] at e_1774_11
  have hr := e_1774_11
  simp only [e_1774_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1774 55) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 399)) := by
  have c108468 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108469 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108470 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108471 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108472 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108473 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1774_12 := arithEq_of_rows h (row := 1774) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_12
  simp only [← c108468, ← c108469, ← c108470] at e_1774_12
  have e_1774_13 := arithEq_of_rows h (row := 1774) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_13
  simp only [← c108471, ← c108472, k2, ← c108473] at e_1774_13
  have hr := e_1774_13
  simp only [e_1774_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1774 63) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 400)) := by
  have c108474 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108475 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108476 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108477 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108478 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108479 := (publicBatchWrapper89_copies3389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1774_14 := arithEq_of_rows h (row := 1774) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_14
  simp only [← c108474, ← c108475, ← c108476] at e_1774_14
  have e_1774_15 := arithEq_of_rows h (row := 1774) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_15
  simp only [← c108477, ← c108478, k2, ← c108479] at e_1774_15
  have hr := e_1774_15
  simp only [e_1774_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1774 71) = bselect (a (.wire 3 23)) (a (.virt 200)) (a (.virt 401)) := by
  have c108480 := (publicBatchWrapper89_copies3390 a h).1
  have c108481 := (publicBatchWrapper89_copies3390 a h).2.1
  have c108482 := (publicBatchWrapper89_copies3390 a h).2.2.1
  have c108483 := (publicBatchWrapper89_copies3390 a h).2.2.2.1
  have c108484 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.1
  have c108485 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1774_16 := arithEq_of_rows h (row := 1774) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_16
  simp only [← c108480, ← c108481, ← c108482] at e_1774_16
  have e_1774_17 := arithEq_of_rows h (row := 1774) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_17
  simp only [← c108483, ← c108484, k2, ← c108485] at e_1774_17
  have hr := e_1774_17
  simp only [e_1774_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1774 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 570)) := by
  have c108486 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.1
  have c108487 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.1
  have c108488 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.1
  have c108489 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.1
  have c108490 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108491 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1774_18 := arithEq_of_rows h (row := 1774) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_18
  simp only [← c108486, ← c108487, ← c108488] at e_1774_18
  have e_1774_19 := arithEq_of_rows h (row := 1774) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1774_19
  simp only [← c108489, ← c108490, k2, ← c108491] at e_1774_19
  have hr := e_1774_19
  simp only [e_1774_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1775 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 571)) := by
  have c108492 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108493 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108494 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108495 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108496 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108497 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1775_0 := arithEq_of_rows h (row := 1775) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_0
  simp only [← c108492, ← c108493, ← c108494] at e_1775_0
  have e_1775_1 := arithEq_of_rows h (row := 1775) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_1
  simp only [← c108495, ← c108496, k2, ← c108497] at e_1775_1
  have hr := e_1775_1
  simp only [e_1775_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1775 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 572)) := by
  have c108498 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108499 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108500 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108501 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108502 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108503 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1775_2 := arithEq_of_rows h (row := 1775) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_2
  simp only [← c108498, ← c108499, ← c108500] at e_1775_2
  have e_1775_3 := arithEq_of_rows h (row := 1775) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_3
  simp only [← c108501, ← c108502, k2, ← c108503] at e_1775_3
  have hr := e_1775_3
  simp only [e_1775_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1775 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 573)) := by
  have c108504 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108505 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108506 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108507 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108508 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108509 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1775_4 := arithEq_of_rows h (row := 1775) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_4
  simp only [← c108504, ← c108505, ← c108506] at e_1775_4
  have e_1775_5 := arithEq_of_rows h (row := 1775) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_5
  simp only [← c108507, ← c108508, k2, ← c108509] at e_1775_5
  have hr := e_1775_5
  simp only [e_1775_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1775 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 574)) := by
  have c108510 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108511 := (publicBatchWrapper89_copies3390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108512 := (publicBatchWrapper89_copies3391 a h).1
  have c108513 := (publicBatchWrapper89_copies3391 a h).2.1
  have c108514 := (publicBatchWrapper89_copies3391 a h).2.2.1
  have c108515 := (publicBatchWrapper89_copies3391 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1775_6 := arithEq_of_rows h (row := 1775) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_6
  simp only [← c108510, ← c108511, ← c108512] at e_1775_6
  have e_1775_7 := arithEq_of_rows h (row := 1775) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_7
  simp only [← c108513, ← c108514, k2, ← c108515] at e_1775_7
  have hr := e_1775_7
  simp only [e_1775_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1775 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 575)) := by
  have c108516 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.1
  have c108517 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.1
  have c108518 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.1
  have c108519 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.1
  have c108520 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.1
  have c108521 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1775_8 := arithEq_of_rows h (row := 1775) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_8
  simp only [← c108516, ← c108517, ← c108518] at e_1775_8
  have e_1775_9 := arithEq_of_rows h (row := 1775) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_9
  simp only [← c108519, ← c108520, k2, ← c108521] at e_1775_9
  have hr := e_1775_9
  simp only [e_1775_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1775 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 576)) := by
  have c108522 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108523 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108524 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108525 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108526 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108527 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1775_10 := arithEq_of_rows h (row := 1775) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_10
  simp only [← c108522, ← c108523, ← c108524] at e_1775_10
  have e_1775_11 := arithEq_of_rows h (row := 1775) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_11
  simp only [← c108525, ← c108526, k2, ← c108527] at e_1775_11
  have hr := e_1775_11
  simp only [e_1775_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1775 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 577)) := by
  have c108528 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108529 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108530 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108531 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108532 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108533 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1775_12 := arithEq_of_rows h (row := 1775) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_12
  simp only [← c108528, ← c108529, ← c108530] at e_1775_12
  have e_1775_13 := arithEq_of_rows h (row := 1775) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_13
  simp only [← c108531, ← c108532, k2, ← c108533] at e_1775_13
  have hr := e_1775_13
  simp only [e_1775_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1775 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 578)) := by
  have c108534 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108535 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108536 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108537 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108538 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108539 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1775_14 := arithEq_of_rows h (row := 1775) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_14
  simp only [← c108534, ← c108535, ← c108536] at e_1775_14
  have e_1775_15 := arithEq_of_rows h (row := 1775) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_15
  simp only [← c108537, ← c108538, k2, ← c108539] at e_1775_15
  have hr := e_1775_15
  simp only [e_1775_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1775 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 579)) := by
  have c108540 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108541 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108542 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108543 := (publicBatchWrapper89_copies3391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108544 := (publicBatchWrapper89_copies3392 a h).1
  have c108545 := (publicBatchWrapper89_copies3392 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1775_16 := arithEq_of_rows h (row := 1775) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_16
  simp only [← c108540, ← c108541, ← c108542] at e_1775_16
  have e_1775_17 := arithEq_of_rows h (row := 1775) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_17
  simp only [← c108543, ← c108544, k2, ← c108545] at e_1775_17
  have hr := e_1775_17
  simp only [e_1775_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1775 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 580)) := by
  have c108546 := (publicBatchWrapper89_copies3392 a h).2.2.1
  have c108547 := (publicBatchWrapper89_copies3392 a h).2.2.2.1
  have c108548 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.1
  have c108549 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.1
  have c108550 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.1
  have c108551 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1775_18 := arithEq_of_rows h (row := 1775) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_18
  simp only [← c108546, ← c108547, ← c108548] at e_1775_18
  have e_1775_19 := arithEq_of_rows h (row := 1775) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1775_19
  simp only [← c108549, ← c108550, k2, ← c108551] at e_1775_19
  have hr := e_1775_19
  simp only [e_1775_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1776 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 581)) := by
  have c108552 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.1
  have c108553 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.1
  have c108554 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108555 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108556 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108557 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1776_0 := arithEq_of_rows h (row := 1776) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_0
  simp only [← c108552, ← c108553, ← c108554] at e_1776_0
  have e_1776_1 := arithEq_of_rows h (row := 1776) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_1
  simp only [← c108555, ← c108556, k2, ← c108557] at e_1776_1
  have hr := e_1776_1
  simp only [e_1776_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1776 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 582)) := by
  have c108558 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108559 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108560 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108561 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108562 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108563 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1776_2 := arithEq_of_rows h (row := 1776) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_2
  simp only [← c108558, ← c108559, ← c108560] at e_1776_2
  have e_1776_3 := arithEq_of_rows h (row := 1776) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_3
  simp only [← c108561, ← c108562, k2, ← c108563] at e_1776_3
  have hr := e_1776_3
  simp only [e_1776_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1776 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 583)) := by
  have c108564 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108565 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108566 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108567 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108568 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108569 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1776_4 := arithEq_of_rows h (row := 1776) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_4
  simp only [← c108564, ← c108565, ← c108566] at e_1776_4
  have e_1776_5 := arithEq_of_rows h (row := 1776) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_5
  simp only [← c108567, ← c108568, k2, ← c108569] at e_1776_5
  have hr := e_1776_5
  simp only [e_1776_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1776 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 584)) := by
  have c108570 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108571 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108572 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108573 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108574 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108575 := (publicBatchWrapper89_copies3392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1776_6 := arithEq_of_rows h (row := 1776) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_6
  simp only [← c108570, ← c108571, ← c108572] at e_1776_6
  have e_1776_7 := arithEq_of_rows h (row := 1776) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_7
  simp only [← c108573, ← c108574, k2, ← c108575] at e_1776_7
  have hr := e_1776_7
  simp only [e_1776_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1776 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 585)) := by
  have c108576 := (publicBatchWrapper89_copies3393 a h).1
  have c108577 := (publicBatchWrapper89_copies3393 a h).2.1
  have c108578 := (publicBatchWrapper89_copies3393 a h).2.2.1
  have c108579 := (publicBatchWrapper89_copies3393 a h).2.2.2.1
  have c108580 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.1
  have c108581 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1776_8 := arithEq_of_rows h (row := 1776) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_8
  simp only [← c108576, ← c108577, ← c108578] at e_1776_8
  have e_1776_9 := arithEq_of_rows h (row := 1776) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_9
  simp only [← c108579, ← c108580, k2, ← c108581] at e_1776_9
  have hr := e_1776_9
  simp only [e_1776_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1776 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 586)) := by
  have c108582 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.1
  have c108583 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.1
  have c108584 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.1
  have c108585 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.1
  have c108586 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108587 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1776_10 := arithEq_of_rows h (row := 1776) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_10
  simp only [← c108582, ← c108583, ← c108584] at e_1776_10
  have e_1776_11 := arithEq_of_rows h (row := 1776) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_11
  simp only [← c108585, ← c108586, k2, ← c108587] at e_1776_11
  have hr := e_1776_11
  simp only [e_1776_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1776 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 587)) := by
  have c108588 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108589 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108590 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108591 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108592 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108593 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1776_12 := arithEq_of_rows h (row := 1776) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_12
  simp only [← c108588, ← c108589, ← c108590] at e_1776_12
  have e_1776_13 := arithEq_of_rows h (row := 1776) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_13
  simp only [← c108591, ← c108592, k2, ← c108593] at e_1776_13
  have hr := e_1776_13
  simp only [e_1776_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1776 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 588)) := by
  have c108594 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108595 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108596 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108597 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108598 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108599 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1776_14 := arithEq_of_rows h (row := 1776) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_14
  simp only [← c108594, ← c108595, ← c108596] at e_1776_14
  have e_1776_15 := arithEq_of_rows h (row := 1776) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_15
  simp only [← c108597, ← c108598, k2, ← c108599] at e_1776_15
  have hr := e_1776_15
  simp only [e_1776_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1776 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 589)) := by
  have c108600 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108601 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108602 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108603 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108604 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108605 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1776_16 := arithEq_of_rows h (row := 1776) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_16
  simp only [← c108600, ← c108601, ← c108602] at e_1776_16
  have e_1776_17 := arithEq_of_rows h (row := 1776) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_17
  simp only [← c108603, ← c108604, k2, ← c108605] at e_1776_17
  have hr := e_1776_17
  simp only [e_1776_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1776 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 590)) := by
  have c108606 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108607 := (publicBatchWrapper89_copies3393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108608 := (publicBatchWrapper89_copies3394 a h).1
  have c108609 := (publicBatchWrapper89_copies3394 a h).2.1
  have c108610 := (publicBatchWrapper89_copies3394 a h).2.2.1
  have c108611 := (publicBatchWrapper89_copies3394 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1776_18 := arithEq_of_rows h (row := 1776) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_18
  simp only [← c108606, ← c108607, ← c108608] at e_1776_18
  have e_1776_19 := arithEq_of_rows h (row := 1776) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1776_19
  simp only [← c108609, ← c108610, k2, ← c108611] at e_1776_19
  have hr := e_1776_19
  simp only [e_1776_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1777 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 591)) := by
  have c108612 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.1
  have c108613 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.1
  have c108614 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.1
  have c108615 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.1
  have c108616 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.1
  have c108617 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1777_0 := arithEq_of_rows h (row := 1777) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_0
  simp only [← c108612, ← c108613, ← c108614] at e_1777_0
  have e_1777_1 := arithEq_of_rows h (row := 1777) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_1
  simp only [← c108615, ← c108616, k2, ← c108617] at e_1777_1
  have hr := e_1777_1
  simp only [e_1777_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1777 15) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 592)) := by
  have c108618 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108619 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108620 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108621 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108622 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108623 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1777_2 := arithEq_of_rows h (row := 1777) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_2
  simp only [← c108618, ← c108619, ← c108620] at e_1777_2
  have e_1777_3 := arithEq_of_rows h (row := 1777) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_3
  simp only [← c108621, ← c108622, k2, ← c108623] at e_1777_3
  have hr := e_1777_3
  simp only [e_1777_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1777 23) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 593)) := by
  have c108624 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108625 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108626 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108627 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108628 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108629 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1777_4 := arithEq_of_rows h (row := 1777) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_4
  simp only [← c108624, ← c108625, ← c108626] at e_1777_4
  have e_1777_5 := arithEq_of_rows h (row := 1777) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_5
  simp only [← c108627, ← c108628, k2, ← c108629] at e_1777_5
  have hr := e_1777_5
  simp only [e_1777_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1777 31) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 594)) := by
  have c108630 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108631 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108632 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108633 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108634 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108635 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1777_6 := arithEq_of_rows h (row := 1777) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_6
  simp only [← c108630, ← c108631, ← c108632] at e_1777_6
  have e_1777_7 := arithEq_of_rows h (row := 1777) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_7
  simp only [← c108633, ← c108634, k2, ← c108635] at e_1777_7
  have hr := e_1777_7
  simp only [e_1777_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1777 39) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 595)) := by
  have c108636 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108637 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108638 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108639 := (publicBatchWrapper89_copies3394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108640 := (publicBatchWrapper89_copies3395 a h).1
  have c108641 := (publicBatchWrapper89_copies3395 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1777_8 := arithEq_of_rows h (row := 1777) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_8
  simp only [← c108636, ← c108637, ← c108638] at e_1777_8
  have e_1777_9 := arithEq_of_rows h (row := 1777) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_9
  simp only [← c108639, ← c108640, k2, ← c108641] at e_1777_9
  have hr := e_1777_9
  simp only [e_1777_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1777 47) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 596)) := by
  have c108642 := (publicBatchWrapper89_copies3395 a h).2.2.1
  have c108643 := (publicBatchWrapper89_copies3395 a h).2.2.2.1
  have c108644 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.1
  have c108645 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.1
  have c108646 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.1
  have c108647 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1777_10 := arithEq_of_rows h (row := 1777) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_10
  simp only [← c108642, ← c108643, ← c108644] at e_1777_10
  have e_1777_11 := arithEq_of_rows h (row := 1777) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_11
  simp only [← c108645, ← c108646, k2, ← c108647] at e_1777_11
  have hr := e_1777_11
  simp only [e_1777_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1777 55) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 597)) := by
  have c108648 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.1
  have c108649 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.1
  have c108650 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108651 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108652 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108653 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1777_12 := arithEq_of_rows h (row := 1777) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_12
  simp only [← c108648, ← c108649, ← c108650] at e_1777_12
  have e_1777_13 := arithEq_of_rows h (row := 1777) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_13
  simp only [← c108651, ← c108652, k2, ← c108653] at e_1777_13
  have hr := e_1777_13
  simp only [e_1777_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1777 63) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 598)) := by
  have c108654 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108655 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108656 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108657 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108658 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108659 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1777_14 := arithEq_of_rows h (row := 1777) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_14
  simp only [← c108654, ← c108655, ← c108656] at e_1777_14
  have e_1777_15 := arithEq_of_rows h (row := 1777) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_15
  simp only [← c108657, ← c108658, k2, ← c108659] at e_1777_15
  have hr := e_1777_15
  simp only [e_1777_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1777 71) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 599)) := by
  have c108660 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108661 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108662 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108663 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108664 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108665 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1777_16 := arithEq_of_rows h (row := 1777) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_16
  simp only [← c108660, ← c108661, ← c108662] at e_1777_16
  have e_1777_17 := arithEq_of_rows h (row := 1777) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_17
  simp only [← c108663, ← c108664, k2, ← c108665] at e_1777_17
  have hr := e_1777_17
  simp only [e_1777_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1777 79) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 600)) := by
  have c108666 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108667 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108668 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108669 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108670 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108671 := (publicBatchWrapper89_copies3395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1777_18 := arithEq_of_rows h (row := 1777) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_18
  simp only [← c108666, ← c108667, ← c108668] at e_1777_18
  have e_1777_19 := arithEq_of_rows h (row := 1777) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1777_19
  simp only [← c108669, ← c108670, k2, ← c108671] at e_1777_19
  have hr := e_1777_19
  simp only [e_1777_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1778 7) = bselect (a (.wire 3 67)) (a (.virt 200)) (a (.virt 601)) := by
  have c108672 := (publicBatchWrapper89_copies3396 a h).1
  have c108673 := (publicBatchWrapper89_copies3396 a h).2.1
  have c108674 := (publicBatchWrapper89_copies3396 a h).2.2.1
  have c108675 := (publicBatchWrapper89_copies3396 a h).2.2.2.1
  have c108676 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.1
  have c108677 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1778_0 := arithEq_of_rows h (row := 1778) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_0
  simp only [← c108672, ← c108673, ← c108674] at e_1778_0
  have e_1778_1 := arithEq_of_rows h (row := 1778) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_1
  simp only [← c108675, ← c108676, k2, ← c108677] at e_1778_1
  have hr := e_1778_1
  simp only [e_1778_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1778 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 770)) := by
  have c108678 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.1
  have c108679 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.1
  have c108680 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.1
  have c108681 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.1
  have c108682 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108683 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1778_2 := arithEq_of_rows h (row := 1778) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_2
  simp only [← c108678, ← c108679, ← c108680] at e_1778_2
  have e_1778_3 := arithEq_of_rows h (row := 1778) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_3
  simp only [← c108681, ← c108682, k2, ← c108683] at e_1778_3
  have hr := e_1778_3
  simp only [e_1778_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1778 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 771)) := by
  have c108684 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108685 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108686 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108687 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108688 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108689 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1778_4 := arithEq_of_rows h (row := 1778) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_4
  simp only [← c108684, ← c108685, ← c108686] at e_1778_4
  have e_1778_5 := arithEq_of_rows h (row := 1778) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_5
  simp only [← c108687, ← c108688, k2, ← c108689] at e_1778_5
  have hr := e_1778_5
  simp only [e_1778_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1778 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 772)) := by
  have c108690 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108691 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108692 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108693 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108694 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108695 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1778_6 := arithEq_of_rows h (row := 1778) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_6
  simp only [← c108690, ← c108691, ← c108692] at e_1778_6
  have e_1778_7 := arithEq_of_rows h (row := 1778) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_7
  simp only [← c108693, ← c108694, k2, ← c108695] at e_1778_7
  have hr := e_1778_7
  simp only [e_1778_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1778 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 773)) := by
  have c108696 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108697 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108698 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108699 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108700 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108701 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1778_8 := arithEq_of_rows h (row := 1778) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_8
  simp only [← c108696, ← c108697, ← c108698] at e_1778_8
  have e_1778_9 := arithEq_of_rows h (row := 1778) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_9
  simp only [← c108699, ← c108700, k2, ← c108701] at e_1778_9
  have hr := e_1778_9
  simp only [e_1778_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1778 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 774)) := by
  have c108702 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108703 := (publicBatchWrapper89_copies3396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108704 := (publicBatchWrapper89_copies3397 a h).1
  have c108705 := (publicBatchWrapper89_copies3397 a h).2.1
  have c108706 := (publicBatchWrapper89_copies3397 a h).2.2.1
  have c108707 := (publicBatchWrapper89_copies3397 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1778_10 := arithEq_of_rows h (row := 1778) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_10
  simp only [← c108702, ← c108703, ← c108704] at e_1778_10
  have e_1778_11 := arithEq_of_rows h (row := 1778) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_11
  simp only [← c108705, ← c108706, k2, ← c108707] at e_1778_11
  have hr := e_1778_11
  simp only [e_1778_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1778 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 775)) := by
  have c108708 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.1
  have c108709 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.1
  have c108710 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.1
  have c108711 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.1
  have c108712 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.1
  have c108713 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1778_12 := arithEq_of_rows h (row := 1778) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_12
  simp only [← c108708, ← c108709, ← c108710] at e_1778_12
  have e_1778_13 := arithEq_of_rows h (row := 1778) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_13
  simp only [← c108711, ← c108712, k2, ← c108713] at e_1778_13
  have hr := e_1778_13
  simp only [e_1778_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1778 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 776)) := by
  have c108714 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108715 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108716 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108717 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108718 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108719 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1778_14 := arithEq_of_rows h (row := 1778) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_14
  simp only [← c108714, ← c108715, ← c108716] at e_1778_14
  have e_1778_15 := arithEq_of_rows h (row := 1778) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_15
  simp only [← c108717, ← c108718, k2, ← c108719] at e_1778_15
  have hr := e_1778_15
  simp only [e_1778_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1778 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 777)) := by
  have c108720 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108721 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108722 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108723 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108724 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108725 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1778_16 := arithEq_of_rows h (row := 1778) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_16
  simp only [← c108720, ← c108721, ← c108722] at e_1778_16
  have e_1778_17 := arithEq_of_rows h (row := 1778) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_17
  simp only [← c108723, ← c108724, k2, ← c108725] at e_1778_17
  have hr := e_1778_17
  simp only [e_1778_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1778 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 778)) := by
  have c108726 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108727 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108728 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108729 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108730 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108731 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1778_18 := arithEq_of_rows h (row := 1778) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_18
  simp only [← c108726, ← c108727, ← c108728] at e_1778_18
  have e_1778_19 := arithEq_of_rows h (row := 1778) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1778_19
  simp only [← c108729, ← c108730, k2, ← c108731] at e_1778_19
  have hr := e_1778_19
  simp only [e_1778_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1779 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 779)) := by
  have c108732 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108733 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108734 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108735 := (publicBatchWrapper89_copies3397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108736 := (publicBatchWrapper89_copies3398 a h).1
  have c108737 := (publicBatchWrapper89_copies3398 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1779_0 := arithEq_of_rows h (row := 1779) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_0
  simp only [← c108732, ← c108733, ← c108734] at e_1779_0
  have e_1779_1 := arithEq_of_rows h (row := 1779) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_1
  simp only [← c108735, ← c108736, k2, ← c108737] at e_1779_1
  have hr := e_1779_1
  simp only [e_1779_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1779 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 780)) := by
  have c108738 := (publicBatchWrapper89_copies3398 a h).2.2.1
  have c108739 := (publicBatchWrapper89_copies3398 a h).2.2.2.1
  have c108740 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.1
  have c108741 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.1
  have c108742 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.1
  have c108743 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1779_2 := arithEq_of_rows h (row := 1779) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_2
  simp only [← c108738, ← c108739, ← c108740] at e_1779_2
  have e_1779_3 := arithEq_of_rows h (row := 1779) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_3
  simp only [← c108741, ← c108742, k2, ← c108743] at e_1779_3
  have hr := e_1779_3
  simp only [e_1779_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1779 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 781)) := by
  have c108744 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.1
  have c108745 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.1
  have c108746 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108747 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108748 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108749 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1779_4 := arithEq_of_rows h (row := 1779) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_4
  simp only [← c108744, ← c108745, ← c108746] at e_1779_4
  have e_1779_5 := arithEq_of_rows h (row := 1779) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_5
  simp only [← c108747, ← c108748, k2, ← c108749] at e_1779_5
  have hr := e_1779_5
  simp only [e_1779_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1779 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 782)) := by
  have c108750 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108751 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108752 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108753 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108754 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108755 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1779_6 := arithEq_of_rows h (row := 1779) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_6
  simp only [← c108750, ← c108751, ← c108752] at e_1779_6
  have e_1779_7 := arithEq_of_rows h (row := 1779) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_7
  simp only [← c108753, ← c108754, k2, ← c108755] at e_1779_7
  have hr := e_1779_7
  simp only [e_1779_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1779 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 783)) := by
  have c108756 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108757 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108758 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108759 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108760 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108761 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1779_8 := arithEq_of_rows h (row := 1779) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_8
  simp only [← c108756, ← c108757, ← c108758] at e_1779_8
  have e_1779_9 := arithEq_of_rows h (row := 1779) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_9
  simp only [← c108759, ← c108760, k2, ← c108761] at e_1779_9
  have hr := e_1779_9
  simp only [e_1779_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1779 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 784)) := by
  have c108762 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108763 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108764 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108765 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108766 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108767 := (publicBatchWrapper89_copies3398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1779_10 := arithEq_of_rows h (row := 1779) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_10
  simp only [← c108762, ← c108763, ← c108764] at e_1779_10
  have e_1779_11 := arithEq_of_rows h (row := 1779) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_11
  simp only [← c108765, ← c108766, k2, ← c108767] at e_1779_11
  have hr := e_1779_11
  simp only [e_1779_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1779 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 785)) := by
  have c108768 := (publicBatchWrapper89_copies3399 a h).1
  have c108769 := (publicBatchWrapper89_copies3399 a h).2.1
  have c108770 := (publicBatchWrapper89_copies3399 a h).2.2.1
  have c108771 := (publicBatchWrapper89_copies3399 a h).2.2.2.1
  have c108772 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.1
  have c108773 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1779_12 := arithEq_of_rows h (row := 1779) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_12
  simp only [← c108768, ← c108769, ← c108770] at e_1779_12
  have e_1779_13 := arithEq_of_rows h (row := 1779) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_13
  simp only [← c108771, ← c108772, k2, ← c108773] at e_1779_13
  have hr := e_1779_13
  simp only [e_1779_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1779 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 786)) := by
  have c108774 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.1
  have c108775 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.1
  have c108776 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.1
  have c108777 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.1
  have c108778 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108779 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1779_14 := arithEq_of_rows h (row := 1779) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_14
  simp only [← c108774, ← c108775, ← c108776] at e_1779_14
  have e_1779_15 := arithEq_of_rows h (row := 1779) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_15
  simp only [← c108777, ← c108778, k2, ← c108779] at e_1779_15
  have hr := e_1779_15
  simp only [e_1779_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1779 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 787)) := by
  have c108780 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108781 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108782 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108783 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108784 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108785 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1779_16 := arithEq_of_rows h (row := 1779) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_16
  simp only [← c108780, ← c108781, ← c108782] at e_1779_16
  have e_1779_17 := arithEq_of_rows h (row := 1779) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_17
  simp only [← c108783, ← c108784, k2, ← c108785] at e_1779_17
  have hr := e_1779_17
  simp only [e_1779_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1779 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 788)) := by
  have c108786 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108787 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108788 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108789 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108790 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108791 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1779_18 := arithEq_of_rows h (row := 1779) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_18
  simp only [← c108786, ← c108787, ← c108788] at e_1779_18
  have e_1779_19 := arithEq_of_rows h (row := 1779) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1779_19
  simp only [← c108789, ← c108790, k2, ← c108791] at e_1779_19
  have hr := e_1779_19
  simp only [e_1779_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1780 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 789)) := by
  have c108792 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108793 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108794 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108795 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108796 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108797 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1780_0 := arithEq_of_rows h (row := 1780) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_0
  simp only [← c108792, ← c108793, ← c108794] at e_1780_0
  have e_1780_1 := arithEq_of_rows h (row := 1780) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_1
  simp only [← c108795, ← c108796, k2, ← c108797] at e_1780_1
  have hr := e_1780_1
  simp only [e_1780_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1780 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 790)) := by
  have c108798 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108799 := (publicBatchWrapper89_copies3399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108800 := (publicBatchWrapper89_copies3400 a h).1
  have c108801 := (publicBatchWrapper89_copies3400 a h).2.1
  have c108802 := (publicBatchWrapper89_copies3400 a h).2.2.1
  have c108803 := (publicBatchWrapper89_copies3400 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1780_2 := arithEq_of_rows h (row := 1780) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_2
  simp only [← c108798, ← c108799, ← c108800] at e_1780_2
  have e_1780_3 := arithEq_of_rows h (row := 1780) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_3
  simp only [← c108801, ← c108802, k2, ← c108803] at e_1780_3
  have hr := e_1780_3
  simp only [e_1780_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1780 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 791)) := by
  have c108804 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.1
  have c108805 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.1
  have c108806 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.1
  have c108807 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.1
  have c108808 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.1
  have c108809 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1780_4 := arithEq_of_rows h (row := 1780) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_4
  simp only [← c108804, ← c108805, ← c108806] at e_1780_4
  have e_1780_5 := arithEq_of_rows h (row := 1780) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_5
  simp only [← c108807, ← c108808, k2, ← c108809] at e_1780_5
  have hr := e_1780_5
  simp only [e_1780_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1780 31) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 792)) := by
  have c108810 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108811 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108812 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108813 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108814 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108815 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1780_6 := arithEq_of_rows h (row := 1780) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_6
  simp only [← c108810, ← c108811, ← c108812] at e_1780_6
  have e_1780_7 := arithEq_of_rows h (row := 1780) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_7
  simp only [← c108813, ← c108814, k2, ← c108815] at e_1780_7
  have hr := e_1780_7
  simp only [e_1780_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1780 39) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 793)) := by
  have c108816 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108817 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108818 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108819 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108820 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108821 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1780_8 := arithEq_of_rows h (row := 1780) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_8
  simp only [← c108816, ← c108817, ← c108818] at e_1780_8
  have e_1780_9 := arithEq_of_rows h (row := 1780) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_9
  simp only [← c108819, ← c108820, k2, ← c108821] at e_1780_9
  have hr := e_1780_9
  simp only [e_1780_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1780 47) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 794)) := by
  have c108822 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108823 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108824 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108825 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108826 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108827 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1780_10 := arithEq_of_rows h (row := 1780) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_10
  simp only [← c108822, ← c108823, ← c108824] at e_1780_10
  have e_1780_11 := arithEq_of_rows h (row := 1780) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_11
  simp only [← c108825, ← c108826, k2, ← c108827] at e_1780_11
  have hr := e_1780_11
  simp only [e_1780_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1780 55) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 795)) := by
  have c108828 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108829 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108830 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108831 := (publicBatchWrapper89_copies3400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108832 := (publicBatchWrapper89_copies3401 a h).1
  have c108833 := (publicBatchWrapper89_copies3401 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1780_12 := arithEq_of_rows h (row := 1780) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_12
  simp only [← c108828, ← c108829, ← c108830] at e_1780_12
  have e_1780_13 := arithEq_of_rows h (row := 1780) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_13
  simp only [← c108831, ← c108832, k2, ← c108833] at e_1780_13
  have hr := e_1780_13
  simp only [e_1780_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1780 63) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 796)) := by
  have c108834 := (publicBatchWrapper89_copies3401 a h).2.2.1
  have c108835 := (publicBatchWrapper89_copies3401 a h).2.2.2.1
  have c108836 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.1
  have c108837 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.1
  have c108838 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.1
  have c108839 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1780_14 := arithEq_of_rows h (row := 1780) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_14
  simp only [← c108834, ← c108835, ← c108836] at e_1780_14
  have e_1780_15 := arithEq_of_rows h (row := 1780) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_15
  simp only [← c108837, ← c108838, k2, ← c108839] at e_1780_15
  have hr := e_1780_15
  simp only [e_1780_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1780 71) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 797)) := by
  have c108840 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.1
  have c108841 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.1
  have c108842 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108843 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108844 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108845 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1780_16 := arithEq_of_rows h (row := 1780) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_16
  simp only [← c108840, ← c108841, ← c108842] at e_1780_16
  have e_1780_17 := arithEq_of_rows h (row := 1780) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_17
  simp only [← c108843, ← c108844, k2, ← c108845] at e_1780_17
  have hr := e_1780_17
  simp only [e_1780_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1780 79) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 798)) := by
  have c108846 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108847 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108848 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108849 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108850 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108851 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1780_18 := arithEq_of_rows h (row := 1780) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_18
  simp only [← c108846, ← c108847, ← c108848] at e_1780_18
  have e_1780_19 := arithEq_of_rows h (row := 1780) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1780_19
  simp only [← c108849, ← c108850, k2, ← c108851] at e_1780_19
  have hr := e_1780_19
  simp only [e_1780_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1781 7) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 799)) := by
  have c108852 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108853 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108854 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108855 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108856 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108857 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1781_0 := arithEq_of_rows h (row := 1781) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_0
  simp only [← c108852, ← c108853, ← c108854] at e_1781_0
  have e_1781_1 := arithEq_of_rows h (row := 1781) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_1
  simp only [← c108855, ← c108856, k2, ← c108857] at e_1781_1
  have hr := e_1781_1
  simp only [e_1781_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1781 15) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 800)) := by
  have c108858 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108859 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108860 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108861 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108862 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108863 := (publicBatchWrapper89_copies3401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1781_2 := arithEq_of_rows h (row := 1781) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_2
  simp only [← c108858, ← c108859, ← c108860] at e_1781_2
  have e_1781_3 := arithEq_of_rows h (row := 1781) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_3
  simp only [← c108861, ← c108862, k2, ← c108863] at e_1781_3
  have hr := e_1781_3
  simp only [e_1781_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1781 23) = bselect (a (.wire 5 31)) (a (.virt 200)) (a (.virt 801)) := by
  have c108864 := (publicBatchWrapper89_copies3402 a h).1
  have c108865 := (publicBatchWrapper89_copies3402 a h).2.1
  have c108866 := (publicBatchWrapper89_copies3402 a h).2.2.1
  have c108867 := (publicBatchWrapper89_copies3402 a h).2.2.2.1
  have c108868 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.1
  have c108869 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1781_4 := arithEq_of_rows h (row := 1781) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_4
  simp only [← c108864, ← c108865, ← c108866] at e_1781_4
  have e_1781_5 := arithEq_of_rows h (row := 1781) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_5
  simp only [← c108867, ← c108868, k2, ← c108869] at e_1781_5
  have hr := e_1781_5
  simp only [e_1781_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1781 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 970)) := by
  have c108870 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.1
  have c108871 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.1
  have c108872 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.1
  have c108873 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.1
  have c108874 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108875 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1781_6 := arithEq_of_rows h (row := 1781) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_6
  simp only [← c108870, ← c108871, ← c108872] at e_1781_6
  have e_1781_7 := arithEq_of_rows h (row := 1781) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_7
  simp only [← c108873, ← c108874, k2, ← c108875] at e_1781_7
  have hr := e_1781_7
  simp only [e_1781_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1781 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 971)) := by
  have c108876 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108877 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108878 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108879 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108880 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108881 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1781_8 := arithEq_of_rows h (row := 1781) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_8
  simp only [← c108876, ← c108877, ← c108878] at e_1781_8
  have e_1781_9 := arithEq_of_rows h (row := 1781) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_9
  simp only [← c108879, ← c108880, k2, ← c108881] at e_1781_9
  have hr := e_1781_9
  simp only [e_1781_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1781 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 972)) := by
  have c108882 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108883 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108884 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108885 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108886 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108887 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1781_10 := arithEq_of_rows h (row := 1781) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_10
  simp only [← c108882, ← c108883, ← c108884] at e_1781_10
  have e_1781_11 := arithEq_of_rows h (row := 1781) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_11
  simp only [← c108885, ← c108886, k2, ← c108887] at e_1781_11
  have hr := e_1781_11
  simp only [e_1781_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1781 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 973)) := by
  have c108888 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108889 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108890 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108891 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108892 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108893 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1781_12 := arithEq_of_rows h (row := 1781) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_12
  simp only [← c108888, ← c108889, ← c108890] at e_1781_12
  have e_1781_13 := arithEq_of_rows h (row := 1781) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_13
  simp only [← c108891, ← c108892, k2, ← c108893] at e_1781_13
  have hr := e_1781_13
  simp only [e_1781_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1781 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 974)) := by
  have c108894 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108895 := (publicBatchWrapper89_copies3402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108896 := (publicBatchWrapper89_copies3403 a h).1
  have c108897 := (publicBatchWrapper89_copies3403 a h).2.1
  have c108898 := (publicBatchWrapper89_copies3403 a h).2.2.1
  have c108899 := (publicBatchWrapper89_copies3403 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1781_14 := arithEq_of_rows h (row := 1781) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_14
  simp only [← c108894, ← c108895, ← c108896] at e_1781_14
  have e_1781_15 := arithEq_of_rows h (row := 1781) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_15
  simp only [← c108897, ← c108898, k2, ← c108899] at e_1781_15
  have hr := e_1781_15
  simp only [e_1781_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1781 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 975)) := by
  have c108900 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.1
  have c108901 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.1
  have c108902 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.1
  have c108903 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.1
  have c108904 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.1
  have c108905 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1781_16 := arithEq_of_rows h (row := 1781) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_16
  simp only [← c108900, ← c108901, ← c108902] at e_1781_16
  have e_1781_17 := arithEq_of_rows h (row := 1781) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_17
  simp only [← c108903, ← c108904, k2, ← c108905] at e_1781_17
  have hr := e_1781_17
  simp only [e_1781_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1781 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 976)) := by
  have c108906 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108907 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108908 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108909 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108910 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108911 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1781_18 := arithEq_of_rows h (row := 1781) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_18
  simp only [← c108906, ← c108907, ← c108908] at e_1781_18
  have e_1781_19 := arithEq_of_rows h (row := 1781) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1781_19
  simp only [← c108909, ← c108910, k2, ← c108911] at e_1781_19
  have hr := e_1781_19
  simp only [e_1781_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1782 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 977)) := by
  have c108912 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108913 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108914 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108915 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108916 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108917 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1782_0 := arithEq_of_rows h (row := 1782) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_0
  simp only [← c108912, ← c108913, ← c108914] at e_1782_0
  have e_1782_1 := arithEq_of_rows h (row := 1782) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_1
  simp only [← c108915, ← c108916, k2, ← c108917] at e_1782_1
  have hr := e_1782_1
  simp only [e_1782_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1782 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 978)) := by
  have c108918 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108919 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108920 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108921 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108922 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108923 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1782_2 := arithEq_of_rows h (row := 1782) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_2
  simp only [← c108918, ← c108919, ← c108920] at e_1782_2
  have e_1782_3 := arithEq_of_rows h (row := 1782) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_3
  simp only [← c108921, ← c108922, k2, ← c108923] at e_1782_3
  have hr := e_1782_3
  simp only [e_1782_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1782 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 979)) := by
  have c108924 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108925 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108926 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108927 := (publicBatchWrapper89_copies3403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108928 := (publicBatchWrapper89_copies3404 a h).1
  have c108929 := (publicBatchWrapper89_copies3404 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1782_4 := arithEq_of_rows h (row := 1782) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_4
  simp only [← c108924, ← c108925, ← c108926] at e_1782_4
  have e_1782_5 := arithEq_of_rows h (row := 1782) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_5
  simp only [← c108927, ← c108928, k2, ← c108929] at e_1782_5
  have hr := e_1782_5
  simp only [e_1782_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1782 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 980)) := by
  have c108930 := (publicBatchWrapper89_copies3404 a h).2.2.1
  have c108931 := (publicBatchWrapper89_copies3404 a h).2.2.2.1
  have c108932 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.1
  have c108933 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.1
  have c108934 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.1
  have c108935 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1782_6 := arithEq_of_rows h (row := 1782) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_6
  simp only [← c108930, ← c108931, ← c108932] at e_1782_6
  have e_1782_7 := arithEq_of_rows h (row := 1782) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_7
  simp only [← c108933, ← c108934, k2, ← c108935] at e_1782_7
  have hr := e_1782_7
  simp only [e_1782_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1782 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 981)) := by
  have c108936 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.1
  have c108937 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.1
  have c108938 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108939 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c108940 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108941 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1782_8 := arithEq_of_rows h (row := 1782) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_8
  simp only [← c108936, ← c108937, ← c108938] at e_1782_8
  have e_1782_9 := arithEq_of_rows h (row := 1782) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_9
  simp only [← c108939, ← c108940, k2, ← c108941] at e_1782_9
  have hr := e_1782_9
  simp only [e_1782_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1782 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 982)) := by
  have c108942 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108943 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108944 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108945 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108946 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108947 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1782_10 := arithEq_of_rows h (row := 1782) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_10
  simp only [← c108942, ← c108943, ← c108944] at e_1782_10
  have e_1782_11 := arithEq_of_rows h (row := 1782) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_11
  simp only [← c108945, ← c108946, k2, ← c108947] at e_1782_11
  have hr := e_1782_11
  simp only [e_1782_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1782 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 983)) := by
  have c108948 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108949 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108950 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108951 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108952 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108953 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1782_12 := arithEq_of_rows h (row := 1782) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_12
  simp only [← c108948, ← c108949, ← c108950] at e_1782_12
  have e_1782_13 := arithEq_of_rows h (row := 1782) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_13
  simp only [← c108951, ← c108952, k2, ← c108953] at e_1782_13
  have hr := e_1782_13
  simp only [e_1782_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1782 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 984)) := by
  have c108954 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108955 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108956 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108957 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108958 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108959 := (publicBatchWrapper89_copies3404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1782_14 := arithEq_of_rows h (row := 1782) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_14
  simp only [← c108954, ← c108955, ← c108956] at e_1782_14
  have e_1782_15 := arithEq_of_rows h (row := 1782) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_15
  simp only [← c108957, ← c108958, k2, ← c108959] at e_1782_15
  have hr := e_1782_15
  simp only [e_1782_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1782 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 985)) := by
  have c108960 := (publicBatchWrapper89_copies3405 a h).1
  have c108961 := (publicBatchWrapper89_copies3405 a h).2.1
  have c108962 := (publicBatchWrapper89_copies3405 a h).2.2.1
  have c108963 := (publicBatchWrapper89_copies3405 a h).2.2.2.1
  have c108964 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.1
  have c108965 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1782_16 := arithEq_of_rows h (row := 1782) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_16
  simp only [← c108960, ← c108961, ← c108962] at e_1782_16
  have e_1782_17 := arithEq_of_rows h (row := 1782) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_17
  simp only [← c108963, ← c108964, k2, ← c108965] at e_1782_17
  have hr := e_1782_17
  simp only [e_1782_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1782 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 986)) := by
  have c108966 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.1
  have c108967 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.1
  have c108968 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.1
  have c108969 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.1
  have c108970 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.1
  have c108971 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1782_18 := arithEq_of_rows h (row := 1782) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_18
  simp only [← c108966, ← c108967, ← c108968] at e_1782_18
  have e_1782_19 := arithEq_of_rows h (row := 1782) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1782_19
  simp only [← c108969, ← c108970, k2, ← c108971] at e_1782_19
  have hr := e_1782_19
  simp only [e_1782_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1783 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 987)) := by
  have c108972 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108973 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108974 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108975 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108976 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108977 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1783_0 := arithEq_of_rows h (row := 1783) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_0
  simp only [← c108972, ← c108973, ← c108974] at e_1783_0
  have e_1783_1 := arithEq_of_rows h (row := 1783) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_1
  simp only [← c108975, ← c108976, k2, ← c108977] at e_1783_1
  have hr := e_1783_1
  simp only [e_1783_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1783 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 988)) := by
  have c108978 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108979 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108980 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108981 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108982 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108983 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1783_2 := arithEq_of_rows h (row := 1783) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_2
  simp only [← c108978, ← c108979, ← c108980] at e_1783_2
  have e_1783_3 := arithEq_of_rows h (row := 1783) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_3
  simp only [← c108981, ← c108982, k2, ← c108983] at e_1783_3
  have hr := e_1783_3
  simp only [e_1783_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1783 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 989)) := by
  have c108984 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108985 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108986 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108987 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108988 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108989 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1783_4 := arithEq_of_rows h (row := 1783) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_4
  simp only [← c108984, ← c108985, ← c108986] at e_1783_4
  have e_1783_5 := arithEq_of_rows h (row := 1783) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_5
  simp only [← c108987, ← c108988, k2, ← c108989] at e_1783_5
  have hr := e_1783_5
  simp only [e_1783_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1783 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 990)) := by
  have c108990 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c108991 := (publicBatchWrapper89_copies3405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c108992 := (publicBatchWrapper89_copies3406 a h).1
  have c108993 := (publicBatchWrapper89_copies3406 a h).2.1
  have c108994 := (publicBatchWrapper89_copies3406 a h).2.2.1
  have c108995 := (publicBatchWrapper89_copies3406 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1783_6 := arithEq_of_rows h (row := 1783) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_6
  simp only [← c108990, ← c108991, ← c108992] at e_1783_6
  have e_1783_7 := arithEq_of_rows h (row := 1783) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_7
  simp only [← c108993, ← c108994, k2, ← c108995] at e_1783_7
  have hr := e_1783_7
  simp only [e_1783_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1783 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 991)) := by
  have c108996 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.1
  have c108997 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.1
  have c108998 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.1
  have c108999 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.1
  have c109000 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.1
  have c109001 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1783_8 := arithEq_of_rows h (row := 1783) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_8
  simp only [← c108996, ← c108997, ← c108998] at e_1783_8
  have e_1783_9 := arithEq_of_rows h (row := 1783) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_9
  simp only [← c108999, ← c109000, k2, ← c109001] at e_1783_9
  have hr := e_1783_9
  simp only [e_1783_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1783 47) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 992)) := by
  have c109002 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109003 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109004 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109005 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109006 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109007 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1783_10 := arithEq_of_rows h (row := 1783) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_10
  simp only [← c109002, ← c109003, ← c109004] at e_1783_10
  have e_1783_11 := arithEq_of_rows h (row := 1783) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_11
  simp only [← c109005, ← c109006, k2, ← c109007] at e_1783_11
  have hr := e_1783_11
  simp only [e_1783_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1783 55) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 993)) := by
  have c109008 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109009 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109010 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109011 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109012 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109013 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1783_12 := arithEq_of_rows h (row := 1783) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_12
  simp only [← c109008, ← c109009, ← c109010] at e_1783_12
  have e_1783_13 := arithEq_of_rows h (row := 1783) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_13
  simp only [← c109011, ← c109012, k2, ← c109013] at e_1783_13
  have hr := e_1783_13
  simp only [e_1783_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1783 63) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 994)) := by
  have c109014 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109015 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109016 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109017 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109018 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109019 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1783_14 := arithEq_of_rows h (row := 1783) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_14
  simp only [← c109014, ← c109015, ← c109016] at e_1783_14
  have e_1783_15 := arithEq_of_rows h (row := 1783) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_15
  simp only [← c109017, ← c109018, k2, ← c109019] at e_1783_15
  have hr := e_1783_15
  simp only [e_1783_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1783 71) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 995)) := by
  have c109020 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109021 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109022 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109023 := (publicBatchWrapper89_copies3406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109024 := (publicBatchWrapper89_copies3407 a h).1
  have c109025 := (publicBatchWrapper89_copies3407 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1783_16 := arithEq_of_rows h (row := 1783) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_16
  simp only [← c109020, ← c109021, ← c109022] at e_1783_16
  have e_1783_17 := arithEq_of_rows h (row := 1783) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_17
  simp only [← c109023, ← c109024, k2, ← c109025] at e_1783_17
  have hr := e_1783_17
  simp only [e_1783_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1783 79) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 996)) := by
  have c109026 := (publicBatchWrapper89_copies3407 a h).2.2.1
  have c109027 := (publicBatchWrapper89_copies3407 a h).2.2.2.1
  have c109028 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.1
  have c109029 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.1
  have c109030 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.1
  have c109031 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1783_18 := arithEq_of_rows h (row := 1783) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_18
  simp only [← c109026, ← c109027, ← c109028] at e_1783_18
  have e_1783_19 := arithEq_of_rows h (row := 1783) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1783_19
  simp only [← c109029, ← c109030, k2, ← c109031] at e_1783_19
  have hr := e_1783_19
  simp only [e_1783_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1784 7) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 997)) := by
  have c109032 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.1
  have c109033 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.1
  have c109034 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109035 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109036 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109037 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1784_0 := arithEq_of_rows h (row := 1784) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_0
  simp only [← c109032, ← c109033, ← c109034] at e_1784_0
  have e_1784_1 := arithEq_of_rows h (row := 1784) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_1
  simp only [← c109035, ← c109036, k2, ← c109037] at e_1784_1
  have hr := e_1784_1
  simp only [e_1784_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1784 15) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 998)) := by
  have c109038 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109039 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109040 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109041 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109042 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109043 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1784_2 := arithEq_of_rows h (row := 1784) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_2
  simp only [← c109038, ← c109039, ← c109040] at e_1784_2
  have e_1784_3 := arithEq_of_rows h (row := 1784) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_3
  simp only [← c109041, ← c109042, k2, ← c109043] at e_1784_3
  have hr := e_1784_3
  simp only [e_1784_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1784 23) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 999)) := by
  have c109044 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109045 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109046 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109047 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109048 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109049 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1784_4 := arithEq_of_rows h (row := 1784) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_4
  simp only [← c109044, ← c109045, ← c109046] at e_1784_4
  have e_1784_5 := arithEq_of_rows h (row := 1784) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_5
  simp only [← c109047, ← c109048, k2, ← c109049] at e_1784_5
  have hr := e_1784_5
  simp only [e_1784_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1784 31) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 1000)) := by
  have c109050 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109051 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109052 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109053 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109054 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109055 := (publicBatchWrapper89_copies3407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1784_6 := arithEq_of_rows h (row := 1784) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_6
  simp only [← c109050, ← c109051, ← c109052] at e_1784_6
  have e_1784_7 := arithEq_of_rows h (row := 1784) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_7
  simp only [← c109053, ← c109054, k2, ← c109055] at e_1784_7
  have hr := e_1784_7
  simp only [e_1784_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1784 39) = bselect (a (.wire 5 75)) (a (.virt 200)) (a (.virt 1001)) := by
  have c109056 := (publicBatchWrapper89_copies3408 a h).1
  have c109057 := (publicBatchWrapper89_copies3408 a h).2.1
  have c109058 := (publicBatchWrapper89_copies3408 a h).2.2.1
  have c109059 := (publicBatchWrapper89_copies3408 a h).2.2.2.1
  have c109060 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.1
  have c109061 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1784_8 := arithEq_of_rows h (row := 1784) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_8
  simp only [← c109056, ← c109057, ← c109058] at e_1784_8
  have e_1784_9 := arithEq_of_rows h (row := 1784) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_9
  simp only [← c109059, ← c109060, k2, ← c109061] at e_1784_9
  have hr := e_1784_9
  simp only [e_1784_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1784 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1170)) := by
  have c109062 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.1
  have c109063 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.1
  have c109064 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.1
  have c109065 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.1
  have c109066 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109067 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1784_10 := arithEq_of_rows h (row := 1784) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_10
  simp only [← c109062, ← c109063, ← c109064] at e_1784_10
  have e_1784_11 := arithEq_of_rows h (row := 1784) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_11
  simp only [← c109065, ← c109066, k2, ← c109067] at e_1784_11
  have hr := e_1784_11
  simp only [e_1784_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1784 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1171)) := by
  have c109068 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109069 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109070 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109071 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109072 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109073 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1784_12 := arithEq_of_rows h (row := 1784) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_12
  simp only [← c109068, ← c109069, ← c109070] at e_1784_12
  have e_1784_13 := arithEq_of_rows h (row := 1784) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_13
  simp only [← c109071, ← c109072, k2, ← c109073] at e_1784_13
  have hr := e_1784_13
  simp only [e_1784_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1784 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1172)) := by
  have c109074 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109075 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109076 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109077 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109078 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109079 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1784_14 := arithEq_of_rows h (row := 1784) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_14
  simp only [← c109074, ← c109075, ← c109076] at e_1784_14
  have e_1784_15 := arithEq_of_rows h (row := 1784) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_15
  simp only [← c109077, ← c109078, k2, ← c109079] at e_1784_15
  have hr := e_1784_15
  simp only [e_1784_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1784 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1173)) := by
  have c109080 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109081 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109082 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109083 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109084 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109085 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1784_16 := arithEq_of_rows h (row := 1784) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_16
  simp only [← c109080, ← c109081, ← c109082] at e_1784_16
  have e_1784_17 := arithEq_of_rows h (row := 1784) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_17
  simp only [← c109083, ← c109084, k2, ← c109085] at e_1784_17
  have hr := e_1784_17
  simp only [e_1784_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1784 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1174)) := by
  have c109086 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109087 := (publicBatchWrapper89_copies3408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109088 := (publicBatchWrapper89_copies3409 a h).1
  have c109089 := (publicBatchWrapper89_copies3409 a h).2.1
  have c109090 := (publicBatchWrapper89_copies3409 a h).2.2.1
  have c109091 := (publicBatchWrapper89_copies3409 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1784_18 := arithEq_of_rows h (row := 1784) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_18
  simp only [← c109086, ← c109087, ← c109088] at e_1784_18
  have e_1784_19 := arithEq_of_rows h (row := 1784) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1784_19
  simp only [← c109089, ← c109090, k2, ← c109091] at e_1784_19
  have hr := e_1784_19
  simp only [e_1784_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1785 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1175)) := by
  have c109092 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.1
  have c109093 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.1
  have c109094 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.1
  have c109095 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.1
  have c109096 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.1
  have c109097 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1785_0 := arithEq_of_rows h (row := 1785) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_0
  simp only [← c109092, ← c109093, ← c109094] at e_1785_0
  have e_1785_1 := arithEq_of_rows h (row := 1785) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_1
  simp only [← c109095, ← c109096, k2, ← c109097] at e_1785_1
  have hr := e_1785_1
  simp only [e_1785_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1785 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1176)) := by
  have c109098 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109099 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109100 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109101 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109102 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109103 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1785_2 := arithEq_of_rows h (row := 1785) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_2
  simp only [← c109098, ← c109099, ← c109100] at e_1785_2
  have e_1785_3 := arithEq_of_rows h (row := 1785) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_3
  simp only [← c109101, ← c109102, k2, ← c109103] at e_1785_3
  have hr := e_1785_3
  simp only [e_1785_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1785 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1177)) := by
  have c109104 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109105 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109106 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109107 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109108 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109109 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1785_4 := arithEq_of_rows h (row := 1785) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_4
  simp only [← c109104, ← c109105, ← c109106] at e_1785_4
  have e_1785_5 := arithEq_of_rows h (row := 1785) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_5
  simp only [← c109107, ← c109108, k2, ← c109109] at e_1785_5
  have hr := e_1785_5
  simp only [e_1785_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1785 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1178)) := by
  have c109110 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109111 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109112 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109113 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109114 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109115 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1785_6 := arithEq_of_rows h (row := 1785) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_6
  simp only [← c109110, ← c109111, ← c109112] at e_1785_6
  have e_1785_7 := arithEq_of_rows h (row := 1785) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_7
  simp only [← c109113, ← c109114, k2, ← c109115] at e_1785_7
  have hr := e_1785_7
  simp only [e_1785_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1785 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1179)) := by
  have c109116 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109117 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109118 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109119 := (publicBatchWrapper89_copies3409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109120 := (publicBatchWrapper89_copies3410 a h).1
  have c109121 := (publicBatchWrapper89_copies3410 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1785_8 := arithEq_of_rows h (row := 1785) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_8
  simp only [← c109116, ← c109117, ← c109118] at e_1785_8
  have e_1785_9 := arithEq_of_rows h (row := 1785) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_9
  simp only [← c109119, ← c109120, k2, ← c109121] at e_1785_9
  have hr := e_1785_9
  simp only [e_1785_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1785 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1180)) := by
  have c109122 := (publicBatchWrapper89_copies3410 a h).2.2.1
  have c109123 := (publicBatchWrapper89_copies3410 a h).2.2.2.1
  have c109124 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.1
  have c109125 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.1
  have c109126 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.1
  have c109127 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1785_10 := arithEq_of_rows h (row := 1785) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_10
  simp only [← c109122, ← c109123, ← c109124] at e_1785_10
  have e_1785_11 := arithEq_of_rows h (row := 1785) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_11
  simp only [← c109125, ← c109126, k2, ← c109127] at e_1785_11
  have hr := e_1785_11
  simp only [e_1785_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1785 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1181)) := by
  have c109128 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.1
  have c109129 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.1
  have c109130 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109131 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109132 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109133 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1785_12 := arithEq_of_rows h (row := 1785) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_12
  simp only [← c109128, ← c109129, ← c109130] at e_1785_12
  have e_1785_13 := arithEq_of_rows h (row := 1785) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_13
  simp only [← c109131, ← c109132, k2, ← c109133] at e_1785_13
  have hr := e_1785_13
  simp only [e_1785_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1785 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1182)) := by
  have c109134 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109135 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109136 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109137 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109138 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109139 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1785_14 := arithEq_of_rows h (row := 1785) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_14
  simp only [← c109134, ← c109135, ← c109136] at e_1785_14
  have e_1785_15 := arithEq_of_rows h (row := 1785) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_15
  simp only [← c109137, ← c109138, k2, ← c109139] at e_1785_15
  have hr := e_1785_15
  simp only [e_1785_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1785 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1183)) := by
  have c109140 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109141 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109142 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109143 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109144 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109145 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1785_16 := arithEq_of_rows h (row := 1785) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_16
  simp only [← c109140, ← c109141, ← c109142] at e_1785_16
  have e_1785_17 := arithEq_of_rows h (row := 1785) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_17
  simp only [← c109143, ← c109144, k2, ← c109145] at e_1785_17
  have hr := e_1785_17
  simp only [e_1785_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1785 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1184)) := by
  have c109146 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109147 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109148 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109149 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109150 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109151 := (publicBatchWrapper89_copies3410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1785_18 := arithEq_of_rows h (row := 1785) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_18
  simp only [← c109146, ← c109147, ← c109148] at e_1785_18
  have e_1785_19 := arithEq_of_rows h (row := 1785) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1785_19
  simp only [← c109149, ← c109150, k2, ← c109151] at e_1785_19
  have hr := e_1785_19
  simp only [e_1785_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1786 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1185)) := by
  have c109152 := (publicBatchWrapper89_copies3411 a h).1
  have c109153 := (publicBatchWrapper89_copies3411 a h).2.1
  have c109154 := (publicBatchWrapper89_copies3411 a h).2.2.1
  have c109155 := (publicBatchWrapper89_copies3411 a h).2.2.2.1
  have c109156 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.1
  have c109157 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1786_0 := arithEq_of_rows h (row := 1786) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_0
  simp only [← c109152, ← c109153, ← c109154] at e_1786_0
  have e_1786_1 := arithEq_of_rows h (row := 1786) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_1
  simp only [← c109155, ← c109156, k2, ← c109157] at e_1786_1
  have hr := e_1786_1
  simp only [e_1786_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1786 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1186)) := by
  have c109158 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.1
  have c109159 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.1
  have c109160 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.1
  have c109161 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.1
  have c109162 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109163 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1786_2 := arithEq_of_rows h (row := 1786) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_2
  simp only [← c109158, ← c109159, ← c109160] at e_1786_2
  have e_1786_3 := arithEq_of_rows h (row := 1786) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_3
  simp only [← c109161, ← c109162, k2, ← c109163] at e_1786_3
  have hr := e_1786_3
  simp only [e_1786_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1786 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1187)) := by
  have c109164 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109165 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109166 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109167 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109168 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109169 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1786_4 := arithEq_of_rows h (row := 1786) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_4
  simp only [← c109164, ← c109165, ← c109166] at e_1786_4
  have e_1786_5 := arithEq_of_rows h (row := 1786) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_5
  simp only [← c109167, ← c109168, k2, ← c109169] at e_1786_5
  have hr := e_1786_5
  simp only [e_1786_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1786 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1188)) := by
  have c109170 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109171 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109172 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109173 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109174 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109175 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1786_6 := arithEq_of_rows h (row := 1786) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_6
  simp only [← c109170, ← c109171, ← c109172] at e_1786_6
  have e_1786_7 := arithEq_of_rows h (row := 1786) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_7
  simp only [← c109173, ← c109174, k2, ← c109175] at e_1786_7
  have hr := e_1786_7
  simp only [e_1786_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1786 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1189)) := by
  have c109176 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109177 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109178 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109179 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109180 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109181 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1786_8 := arithEq_of_rows h (row := 1786) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_8
  simp only [← c109176, ← c109177, ← c109178] at e_1786_8
  have e_1786_9 := arithEq_of_rows h (row := 1786) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_9
  simp only [← c109179, ← c109180, k2, ← c109181] at e_1786_9
  have hr := e_1786_9
  simp only [e_1786_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1786 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1190)) := by
  have c109182 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109183 := (publicBatchWrapper89_copies3411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109184 := (publicBatchWrapper89_copies3412 a h).1
  have c109185 := (publicBatchWrapper89_copies3412 a h).2.1
  have c109186 := (publicBatchWrapper89_copies3412 a h).2.2.1
  have c109187 := (publicBatchWrapper89_copies3412 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1786_10 := arithEq_of_rows h (row := 1786) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_10
  simp only [← c109182, ← c109183, ← c109184] at e_1786_10
  have e_1786_11 := arithEq_of_rows h (row := 1786) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_11
  simp only [← c109185, ← c109186, k2, ← c109187] at e_1786_11
  have hr := e_1786_11
  simp only [e_1786_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1786 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1191)) := by
  have c109188 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.1
  have c109189 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.1
  have c109190 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.1
  have c109191 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.1
  have c109192 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.1
  have c109193 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1786_12 := arithEq_of_rows h (row := 1786) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_12
  simp only [← c109188, ← c109189, ← c109190] at e_1786_12
  have e_1786_13 := arithEq_of_rows h (row := 1786) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_13
  simp only [← c109191, ← c109192, k2, ← c109193] at e_1786_13
  have hr := e_1786_13
  simp only [e_1786_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1786 63) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1192)) := by
  have c109194 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109195 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109196 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109197 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109198 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109199 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1786_14 := arithEq_of_rows h (row := 1786) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_14
  simp only [← c109194, ← c109195, ← c109196] at e_1786_14
  have e_1786_15 := arithEq_of_rows h (row := 1786) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_15
  simp only [← c109197, ← c109198, k2, ← c109199] at e_1786_15
  have hr := e_1786_15
  simp only [e_1786_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1786 71) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1193)) := by
  have c109200 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109201 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109202 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109203 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109204 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109205 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1786_16 := arithEq_of_rows h (row := 1786) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_16
  simp only [← c109200, ← c109201, ← c109202] at e_1786_16
  have e_1786_17 := arithEq_of_rows h (row := 1786) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_17
  simp only [← c109203, ← c109204, k2, ← c109205] at e_1786_17
  have hr := e_1786_17
  simp only [e_1786_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1786 79) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1194)) := by
  have c109206 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109207 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109208 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109209 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109210 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109211 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1786_18 := arithEq_of_rows h (row := 1786) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_18
  simp only [← c109206, ← c109207, ← c109208] at e_1786_18
  have e_1786_19 := arithEq_of_rows h (row := 1786) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1786_19
  simp only [← c109209, ← c109210, k2, ← c109211] at e_1786_19
  have hr := e_1786_19
  simp only [e_1786_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1787 7) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1195)) := by
  have c109212 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109213 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109214 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109215 := (publicBatchWrapper89_copies3412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109216 := (publicBatchWrapper89_copies3413 a h).1
  have c109217 := (publicBatchWrapper89_copies3413 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1787_0 := arithEq_of_rows h (row := 1787) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_0
  simp only [← c109212, ← c109213, ← c109214] at e_1787_0
  have e_1787_1 := arithEq_of_rows h (row := 1787) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_1
  simp only [← c109215, ← c109216, k2, ← c109217] at e_1787_1
  have hr := e_1787_1
  simp only [e_1787_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1787 15) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1196)) := by
  have c109218 := (publicBatchWrapper89_copies3413 a h).2.2.1
  have c109219 := (publicBatchWrapper89_copies3413 a h).2.2.2.1
  have c109220 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.1
  have c109221 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.1
  have c109222 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.1
  have c109223 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1787_2 := arithEq_of_rows h (row := 1787) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_2
  simp only [← c109218, ← c109219, ← c109220] at e_1787_2
  have e_1787_3 := arithEq_of_rows h (row := 1787) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_3
  simp only [← c109221, ← c109222, k2, ← c109223] at e_1787_3
  have hr := e_1787_3
  simp only [e_1787_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1787 23) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1197)) := by
  have c109224 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.1
  have c109225 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.1
  have c109226 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109227 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109228 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109229 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1787_4 := arithEq_of_rows h (row := 1787) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_4
  simp only [← c109224, ← c109225, ← c109226] at e_1787_4
  have e_1787_5 := arithEq_of_rows h (row := 1787) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_5
  simp only [← c109227, ← c109228, k2, ← c109229] at e_1787_5
  have hr := e_1787_5
  simp only [e_1787_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1787 31) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1198)) := by
  have c109230 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109231 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109232 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109233 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109234 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109235 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1787_6 := arithEq_of_rows h (row := 1787) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_6
  simp only [← c109230, ← c109231, ← c109232] at e_1787_6
  have e_1787_7 := arithEq_of_rows h (row := 1787) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_7
  simp only [← c109233, ← c109234, k2, ← c109235] at e_1787_7
  have hr := e_1787_7
  simp only [e_1787_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1787 39) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1199)) := by
  have c109236 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109237 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109238 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109239 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109240 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109241 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1787_8 := arithEq_of_rows h (row := 1787) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_8
  simp only [← c109236, ← c109237, ← c109238] at e_1787_8
  have e_1787_9 := arithEq_of_rows h (row := 1787) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_9
  simp only [← c109239, ← c109240, k2, ← c109241] at e_1787_9
  have hr := e_1787_9
  simp only [e_1787_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1787 47) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1200)) := by
  have c109242 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109243 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109244 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109245 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109246 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109247 := (publicBatchWrapper89_copies3413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1787_10 := arithEq_of_rows h (row := 1787) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_10
  simp only [← c109242, ← c109243, ← c109244] at e_1787_10
  have e_1787_11 := arithEq_of_rows h (row := 1787) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_11
  simp only [← c109245, ← c109246, k2, ← c109247] at e_1787_11
  have hr := e_1787_11
  simp only [e_1787_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1787 55) = bselect (a (.wire 7 39)) (a (.virt 200)) (a (.virt 1201)) := by
  have c109248 := (publicBatchWrapper89_copies3414 a h).1
  have c109249 := (publicBatchWrapper89_copies3414 a h).2.1
  have c109250 := (publicBatchWrapper89_copies3414 a h).2.2.1
  have c109251 := (publicBatchWrapper89_copies3414 a h).2.2.2.1
  have c109252 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.1
  have c109253 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1787_12 := arithEq_of_rows h (row := 1787) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_12
  simp only [← c109248, ← c109249, ← c109250] at e_1787_12
  have e_1787_13 := arithEq_of_rows h (row := 1787) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_13
  simp only [← c109251, ← c109252, k2, ← c109253] at e_1787_13
  have hr := e_1787_13
  simp only [e_1787_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1787 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1370)) := by
  have c109254 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.1
  have c109255 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.1
  have c109256 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.1
  have c109257 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.1
  have c109258 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109259 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1787_14 := arithEq_of_rows h (row := 1787) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_14
  simp only [← c109254, ← c109255, ← c109256] at e_1787_14
  have e_1787_15 := arithEq_of_rows h (row := 1787) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_15
  simp only [← c109257, ← c109258, k2, ← c109259] at e_1787_15
  have hr := e_1787_15
  simp only [e_1787_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1787 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1371)) := by
  have c109260 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109261 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109262 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109263 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109264 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109265 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1787_16 := arithEq_of_rows h (row := 1787) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_16
  simp only [← c109260, ← c109261, ← c109262] at e_1787_16
  have e_1787_17 := arithEq_of_rows h (row := 1787) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_17
  simp only [← c109263, ← c109264, k2, ← c109265] at e_1787_17
  have hr := e_1787_17
  simp only [e_1787_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1787 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1372)) := by
  have c109266 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109267 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109268 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109269 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109270 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109271 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1787_18 := arithEq_of_rows h (row := 1787) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_18
  simp only [← c109266, ← c109267, ← c109268] at e_1787_18
  have e_1787_19 := arithEq_of_rows h (row := 1787) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1787_19
  simp only [← c109269, ← c109270, k2, ← c109271] at e_1787_19
  have hr := e_1787_19
  simp only [e_1787_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1788 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1373)) := by
  have c109272 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109273 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109274 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109275 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109276 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109277 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1788_0 := arithEq_of_rows h (row := 1788) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_0
  simp only [← c109272, ← c109273, ← c109274] at e_1788_0
  have e_1788_1 := arithEq_of_rows h (row := 1788) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_1
  simp only [← c109275, ← c109276, k2, ← c109277] at e_1788_1
  have hr := e_1788_1
  simp only [e_1788_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1788 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1374)) := by
  have c109278 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109279 := (publicBatchWrapper89_copies3414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109280 := (publicBatchWrapper89_copies3415 a h).1
  have c109281 := (publicBatchWrapper89_copies3415 a h).2.1
  have c109282 := (publicBatchWrapper89_copies3415 a h).2.2.1
  have c109283 := (publicBatchWrapper89_copies3415 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1788_2 := arithEq_of_rows h (row := 1788) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_2
  simp only [← c109278, ← c109279, ← c109280] at e_1788_2
  have e_1788_3 := arithEq_of_rows h (row := 1788) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_3
  simp only [← c109281, ← c109282, k2, ← c109283] at e_1788_3
  have hr := e_1788_3
  simp only [e_1788_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1788 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1375)) := by
  have c109284 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.1
  have c109285 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.1
  have c109286 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.1
  have c109287 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.1
  have c109288 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.1
  have c109289 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1788_4 := arithEq_of_rows h (row := 1788) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_4
  simp only [← c109284, ← c109285, ← c109286] at e_1788_4
  have e_1788_5 := arithEq_of_rows h (row := 1788) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_5
  simp only [← c109287, ← c109288, k2, ← c109289] at e_1788_5
  have hr := e_1788_5
  simp only [e_1788_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1788 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1376)) := by
  have c109290 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109291 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109292 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109293 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109294 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109295 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1788_6 := arithEq_of_rows h (row := 1788) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_6
  simp only [← c109290, ← c109291, ← c109292] at e_1788_6
  have e_1788_7 := arithEq_of_rows h (row := 1788) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_7
  simp only [← c109293, ← c109294, k2, ← c109295] at e_1788_7
  have hr := e_1788_7
  simp only [e_1788_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1788 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1377)) := by
  have c109296 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109297 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109298 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109299 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109300 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109301 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1788_8 := arithEq_of_rows h (row := 1788) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_8
  simp only [← c109296, ← c109297, ← c109298] at e_1788_8
  have e_1788_9 := arithEq_of_rows h (row := 1788) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_9
  simp only [← c109299, ← c109300, k2, ← c109301] at e_1788_9
  have hr := e_1788_9
  simp only [e_1788_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1788 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1378)) := by
  have c109302 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109303 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109304 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109305 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109306 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109307 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1788_10 := arithEq_of_rows h (row := 1788) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_10
  simp only [← c109302, ← c109303, ← c109304] at e_1788_10
  have e_1788_11 := arithEq_of_rows h (row := 1788) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_11
  simp only [← c109305, ← c109306, k2, ← c109307] at e_1788_11
  have hr := e_1788_11
  simp only [e_1788_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1788 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1379)) := by
  have c109308 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109309 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109310 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109311 := (publicBatchWrapper89_copies3415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109312 := (publicBatchWrapper89_copies3416 a h).1
  have c109313 := (publicBatchWrapper89_copies3416 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1788_12 := arithEq_of_rows h (row := 1788) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_12
  simp only [← c109308, ← c109309, ← c109310] at e_1788_12
  have e_1788_13 := arithEq_of_rows h (row := 1788) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_13
  simp only [← c109311, ← c109312, k2, ← c109313] at e_1788_13
  have hr := e_1788_13
  simp only [e_1788_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1788 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1380)) := by
  have c109314 := (publicBatchWrapper89_copies3416 a h).2.2.1
  have c109315 := (publicBatchWrapper89_copies3416 a h).2.2.2.1
  have c109316 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.1
  have c109317 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.1
  have c109318 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.1
  have c109319 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1788_14 := arithEq_of_rows h (row := 1788) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_14
  simp only [← c109314, ← c109315, ← c109316] at e_1788_14
  have e_1788_15 := arithEq_of_rows h (row := 1788) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_15
  simp only [← c109317, ← c109318, k2, ← c109319] at e_1788_15
  have hr := e_1788_15
  simp only [e_1788_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1788 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1381)) := by
  have c109320 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.1
  have c109321 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.1
  have c109322 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109323 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109324 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109325 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1788_16 := arithEq_of_rows h (row := 1788) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_16
  simp only [← c109320, ← c109321, ← c109322] at e_1788_16
  have e_1788_17 := arithEq_of_rows h (row := 1788) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_17
  simp only [← c109323, ← c109324, k2, ← c109325] at e_1788_17
  have hr := e_1788_17
  simp only [e_1788_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1788 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1382)) := by
  have c109326 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109327 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109328 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109329 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109330 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109331 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1788_18 := arithEq_of_rows h (row := 1788) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_18
  simp only [← c109326, ← c109327, ← c109328] at e_1788_18
  have e_1788_19 := arithEq_of_rows h (row := 1788) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1788_19
  simp only [← c109329, ← c109330, k2, ← c109331] at e_1788_19
  have hr := e_1788_19
  simp only [e_1788_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1383)) := by
  have c109332 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109333 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109334 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109335 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109336 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109337 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_0 := arithEq_of_rows h (row := 1789) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_0
  simp only [← c109332, ← c109333, ← c109334] at e_1789_0
  have e_1789_1 := arithEq_of_rows h (row := 1789) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_1
  simp only [← c109335, ← c109336, k2, ← c109337] at e_1789_1
  have hr := e_1789_1
  simp only [e_1789_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1384)) := by
  have c109338 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109339 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109340 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109341 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109342 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109343 := (publicBatchWrapper89_copies3416 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_2 := arithEq_of_rows h (row := 1789) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_2
  simp only [← c109338, ← c109339, ← c109340] at e_1789_2
  have e_1789_3 := arithEq_of_rows h (row := 1789) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_3
  simp only [← c109341, ← c109342, k2, ← c109343] at e_1789_3
  have hr := e_1789_3
  simp only [e_1789_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1385)) := by
  have c109344 := (publicBatchWrapper89_copies3417 a h).1
  have c109345 := (publicBatchWrapper89_copies3417 a h).2.1
  have c109346 := (publicBatchWrapper89_copies3417 a h).2.2.1
  have c109347 := (publicBatchWrapper89_copies3417 a h).2.2.2.1
  have c109348 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.1
  have c109349 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_4 := arithEq_of_rows h (row := 1789) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_4
  simp only [← c109344, ← c109345, ← c109346] at e_1789_4
  have e_1789_5 := arithEq_of_rows h (row := 1789) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_5
  simp only [← c109347, ← c109348, k2, ← c109349] at e_1789_5
  have hr := e_1789_5
  simp only [e_1789_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1386)) := by
  have c109350 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.1
  have c109351 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.1
  have c109352 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.1
  have c109353 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.1
  have c109354 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109355 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_6 := arithEq_of_rows h (row := 1789) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_6
  simp only [← c109350, ← c109351, ← c109352] at e_1789_6
  have e_1789_7 := arithEq_of_rows h (row := 1789) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_7
  simp only [← c109353, ← c109354, k2, ← c109355] at e_1789_7
  have hr := e_1789_7
  simp only [e_1789_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1387)) := by
  have c109356 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109357 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109358 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109359 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109360 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109361 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_8 := arithEq_of_rows h (row := 1789) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_8
  simp only [← c109356, ← c109357, ← c109358] at e_1789_8
  have e_1789_9 := arithEq_of_rows h (row := 1789) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_9
  simp only [← c109359, ← c109360, k2, ← c109361] at e_1789_9
  have hr := e_1789_9
  simp only [e_1789_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1388)) := by
  have c109362 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109363 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109364 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109365 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109366 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109367 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_10 := arithEq_of_rows h (row := 1789) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_10
  simp only [← c109362, ← c109363, ← c109364] at e_1789_10
  have e_1789_11 := arithEq_of_rows h (row := 1789) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_11
  simp only [← c109365, ← c109366, k2, ← c109367] at e_1789_11
  have hr := e_1789_11
  simp only [e_1789_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1389)) := by
  have c109368 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109369 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109370 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109371 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109372 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109373 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_12 := arithEq_of_rows h (row := 1789) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_12
  simp only [← c109368, ← c109369, ← c109370] at e_1789_12
  have e_1789_13 := arithEq_of_rows h (row := 1789) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_13
  simp only [← c109371, ← c109372, k2, ← c109373] at e_1789_13
  have hr := e_1789_13
  simp only [e_1789_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1390)) := by
  have c109374 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109375 := (publicBatchWrapper89_copies3417 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109376 := (publicBatchWrapper89_copies3418 a h).1
  have c109377 := (publicBatchWrapper89_copies3418 a h).2.1
  have c109378 := (publicBatchWrapper89_copies3418 a h).2.2.1
  have c109379 := (publicBatchWrapper89_copies3418 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_14 := arithEq_of_rows h (row := 1789) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_14
  simp only [← c109374, ← c109375, ← c109376] at e_1789_14
  have e_1789_15 := arithEq_of_rows h (row := 1789) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_15
  simp only [← c109377, ← c109378, k2, ← c109379] at e_1789_15
  have hr := e_1789_15
  simp only [e_1789_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1391)) := by
  have c109380 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.1
  have c109381 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.1
  have c109382 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.1
  have c109383 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.1
  have c109384 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.1
  have c109385 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_16 := arithEq_of_rows h (row := 1789) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_16
  simp only [← c109380, ← c109381, ← c109382] at e_1789_16
  have e_1789_17 := arithEq_of_rows h (row := 1789) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_17
  simp only [← c109383, ← c109384, k2, ← c109385] at e_1789_17
  have hr := e_1789_17
  simp only [e_1789_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1789 79) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1392)) := by
  have c109386 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109387 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109388 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109389 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109390 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109391 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1789_18 := arithEq_of_rows h (row := 1789) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_18
  simp only [← c109386, ← c109387, ← c109388] at e_1789_18
  have e_1789_19 := arithEq_of_rows h (row := 1789) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1789_19
  simp only [← c109389, ← c109390, k2, ← c109391] at e_1789_19
  have hr := e_1789_19
  simp only [e_1789_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 7) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1393)) := by
  have c109392 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109393 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109394 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109395 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109396 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109397 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_0 := arithEq_of_rows h (row := 1790) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_0
  simp only [← c109392, ← c109393, ← c109394] at e_1790_0
  have e_1790_1 := arithEq_of_rows h (row := 1790) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_1
  simp only [← c109395, ← c109396, k2, ← c109397] at e_1790_1
  have hr := e_1790_1
  simp only [e_1790_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 15) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1394)) := by
  have c109398 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109399 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109400 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109401 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109402 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109403 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_2 := arithEq_of_rows h (row := 1790) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_2
  simp only [← c109398, ← c109399, ← c109400] at e_1790_2
  have e_1790_3 := arithEq_of_rows h (row := 1790) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_3
  simp only [← c109401, ← c109402, k2, ← c109403] at e_1790_3
  have hr := e_1790_3
  simp only [e_1790_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 23) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1395)) := by
  have c109404 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109405 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109406 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109407 := (publicBatchWrapper89_copies3418 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109408 := (publicBatchWrapper89_copies3419 a h).1
  have c109409 := (publicBatchWrapper89_copies3419 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_4 := arithEq_of_rows h (row := 1790) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_4
  simp only [← c109404, ← c109405, ← c109406] at e_1790_4
  have e_1790_5 := arithEq_of_rows h (row := 1790) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_5
  simp only [← c109407, ← c109408, k2, ← c109409] at e_1790_5
  have hr := e_1790_5
  simp only [e_1790_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 31) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1396)) := by
  have c109410 := (publicBatchWrapper89_copies3419 a h).2.2.1
  have c109411 := (publicBatchWrapper89_copies3419 a h).2.2.2.1
  have c109412 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.1
  have c109413 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.1
  have c109414 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.1
  have c109415 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_6 := arithEq_of_rows h (row := 1790) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_6
  simp only [← c109410, ← c109411, ← c109412] at e_1790_6
  have e_1790_7 := arithEq_of_rows h (row := 1790) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_7
  simp only [← c109413, ← c109414, k2, ← c109415] at e_1790_7
  have hr := e_1790_7
  simp only [e_1790_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 39) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1397)) := by
  have c109416 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.1
  have c109417 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.1
  have c109418 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109419 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109420 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109421 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_8 := arithEq_of_rows h (row := 1790) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_8
  simp only [← c109416, ← c109417, ← c109418] at e_1790_8
  have e_1790_9 := arithEq_of_rows h (row := 1790) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_9
  simp only [← c109419, ← c109420, k2, ← c109421] at e_1790_9
  have hr := e_1790_9
  simp only [e_1790_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 47) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1398)) := by
  have c109422 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109423 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109424 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109425 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109426 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109427 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_10 := arithEq_of_rows h (row := 1790) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_10
  simp only [← c109422, ← c109423, ← c109424] at e_1790_10
  have e_1790_11 := arithEq_of_rows h (row := 1790) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_11
  simp only [← c109425, ← c109426, k2, ← c109427] at e_1790_11
  have hr := e_1790_11
  simp only [e_1790_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 55) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1399)) := by
  have c109428 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109429 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109430 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109431 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109432 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109433 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_12 := arithEq_of_rows h (row := 1790) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_12
  simp only [← c109428, ← c109429, ← c109430] at e_1790_12
  have e_1790_13 := arithEq_of_rows h (row := 1790) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_13
  simp only [← c109431, ← c109432, k2, ← c109433] at e_1790_13
  have hr := e_1790_13
  simp only [e_1790_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 63) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1400)) := by
  have c109434 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109435 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109436 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109437 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109438 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109439 := (publicBatchWrapper89_copies3419 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_14 := arithEq_of_rows h (row := 1790) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_14
  simp only [← c109434, ← c109435, ← c109436] at e_1790_14
  have e_1790_15 := arithEq_of_rows h (row := 1790) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_15
  simp only [← c109437, ← c109438, k2, ← c109439] at e_1790_15
  have hr := e_1790_15
  simp only [e_1790_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 71) = bselect (a (.wire 8 3)) (a (.virt 200)) (a (.virt 1401)) := by
  have c109440 := (publicBatchWrapper89_copies3420 a h).1
  have c109441 := (publicBatchWrapper89_copies3420 a h).2.1
  have c109442 := (publicBatchWrapper89_copies3420 a h).2.2.1
  have c109443 := (publicBatchWrapper89_copies3420 a h).2.2.2.1
  have c109444 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.1
  have c109445 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_16 := arithEq_of_rows h (row := 1790) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_16
  simp only [← c109440, ← c109441, ← c109442] at e_1790_16
  have e_1790_17 := arithEq_of_rows h (row := 1790) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_17
  simp only [← c109443, ← c109444, k2, ← c109445] at e_1790_17
  have hr := e_1790_17
  simp only [e_1790_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1790 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1570)) := by
  have c109446 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.1
  have c109447 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.1
  have c109448 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.1
  have c109449 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.1
  have c109450 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109451 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1790_18 := arithEq_of_rows h (row := 1790) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_18
  simp only [← c109446, ← c109447, ← c109448] at e_1790_18
  have e_1790_19 := arithEq_of_rows h (row := 1790) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1790_19
  simp only [← c109449, ← c109450, k2, ← c109451] at e_1790_19
  have hr := e_1790_19
  simp only [e_1790_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1571)) := by
  have c109452 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109453 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109454 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109455 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109456 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109457 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_0 := arithEq_of_rows h (row := 1791) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_0
  simp only [← c109452, ← c109453, ← c109454] at e_1791_0
  have e_1791_1 := arithEq_of_rows h (row := 1791) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_1
  simp only [← c109455, ← c109456, k2, ← c109457] at e_1791_1
  have hr := e_1791_1
  simp only [e_1791_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1572)) := by
  have c109458 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109459 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109460 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109461 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109462 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109463 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_2 := arithEq_of_rows h (row := 1791) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_2
  simp only [← c109458, ← c109459, ← c109460] at e_1791_2
  have e_1791_3 := arithEq_of_rows h (row := 1791) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_3
  simp only [← c109461, ← c109462, k2, ← c109463] at e_1791_3
  have hr := e_1791_3
  simp only [e_1791_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1573)) := by
  have c109464 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109465 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109466 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109467 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109468 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109469 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_4 := arithEq_of_rows h (row := 1791) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_4
  simp only [← c109464, ← c109465, ← c109466] at e_1791_4
  have e_1791_5 := arithEq_of_rows h (row := 1791) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_5
  simp only [← c109467, ← c109468, k2, ← c109469] at e_1791_5
  have hr := e_1791_5
  simp only [e_1791_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1574)) := by
  have c109470 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109471 := (publicBatchWrapper89_copies3420 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109472 := (publicBatchWrapper89_copies3421 a h).1
  have c109473 := (publicBatchWrapper89_copies3421 a h).2.1
  have c109474 := (publicBatchWrapper89_copies3421 a h).2.2.1
  have c109475 := (publicBatchWrapper89_copies3421 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_6 := arithEq_of_rows h (row := 1791) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_6
  simp only [← c109470, ← c109471, ← c109472] at e_1791_6
  have e_1791_7 := arithEq_of_rows h (row := 1791) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_7
  simp only [← c109473, ← c109474, k2, ← c109475] at e_1791_7
  have hr := e_1791_7
  simp only [e_1791_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1575)) := by
  have c109476 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.1
  have c109477 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.1
  have c109478 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.1
  have c109479 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.1
  have c109480 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.1
  have c109481 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_8 := arithEq_of_rows h (row := 1791) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_8
  simp only [← c109476, ← c109477, ← c109478] at e_1791_8
  have e_1791_9 := arithEq_of_rows h (row := 1791) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_9
  simp only [← c109479, ← c109480, k2, ← c109481] at e_1791_9
  have hr := e_1791_9
  simp only [e_1791_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1576)) := by
  have c109482 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109483 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109484 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109485 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109486 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109487 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_10 := arithEq_of_rows h (row := 1791) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_10
  simp only [← c109482, ← c109483, ← c109484] at e_1791_10
  have e_1791_11 := arithEq_of_rows h (row := 1791) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_11
  simp only [← c109485, ← c109486, k2, ← c109487] at e_1791_11
  have hr := e_1791_11
  simp only [e_1791_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1577)) := by
  have c109488 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109489 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109490 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109491 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109492 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109493 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_12 := arithEq_of_rows h (row := 1791) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_12
  simp only [← c109488, ← c109489, ← c109490] at e_1791_12
  have e_1791_13 := arithEq_of_rows h (row := 1791) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_13
  simp only [← c109491, ← c109492, k2, ← c109493] at e_1791_13
  have hr := e_1791_13
  simp only [e_1791_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1578)) := by
  have c109494 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109495 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109496 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109497 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109498 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109499 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_14 := arithEq_of_rows h (row := 1791) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_14
  simp only [← c109494, ← c109495, ← c109496] at e_1791_14
  have e_1791_15 := arithEq_of_rows h (row := 1791) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_15
  simp only [← c109497, ← c109498, k2, ← c109499] at e_1791_15
  have hr := e_1791_15
  simp only [e_1791_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1579)) := by
  have c109500 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109501 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109502 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109503 := (publicBatchWrapper89_copies3421 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109504 := (publicBatchWrapper89_copies3422 a h).1
  have c109505 := (publicBatchWrapper89_copies3422 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_16 := arithEq_of_rows h (row := 1791) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_16
  simp only [← c109500, ← c109501, ← c109502] at e_1791_16
  have e_1791_17 := arithEq_of_rows h (row := 1791) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_17
  simp only [← c109503, ← c109504, k2, ← c109505] at e_1791_17
  have hr := e_1791_17
  simp only [e_1791_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1791 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1580)) := by
  have c109506 := (publicBatchWrapper89_copies3422 a h).2.2.1
  have c109507 := (publicBatchWrapper89_copies3422 a h).2.2.2.1
  have c109508 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.1
  have c109509 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.1
  have c109510 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.1
  have c109511 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1791_18 := arithEq_of_rows h (row := 1791) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_18
  simp only [← c109506, ← c109507, ← c109508] at e_1791_18
  have e_1791_19 := arithEq_of_rows h (row := 1791) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1791_19
  simp only [← c109509, ← c109510, k2, ← c109511] at e_1791_19
  have hr := e_1791_19
  simp only [e_1791_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1581)) := by
  have c109512 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.1
  have c109513 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.1
  have c109514 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109515 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109516 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109517 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_0 := arithEq_of_rows h (row := 1792) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_0
  simp only [← c109512, ← c109513, ← c109514] at e_1792_0
  have e_1792_1 := arithEq_of_rows h (row := 1792) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_1
  simp only [← c109515, ← c109516, k2, ← c109517] at e_1792_1
  have hr := e_1792_1
  simp only [e_1792_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1582)) := by
  have c109518 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109519 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109520 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109521 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109522 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109523 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_2 := arithEq_of_rows h (row := 1792) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_2
  simp only [← c109518, ← c109519, ← c109520] at e_1792_2
  have e_1792_3 := arithEq_of_rows h (row := 1792) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_3
  simp only [← c109521, ← c109522, k2, ← c109523] at e_1792_3
  have hr := e_1792_3
  simp only [e_1792_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1583)) := by
  have c109524 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109525 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109526 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109527 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109528 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109529 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_4 := arithEq_of_rows h (row := 1792) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_4
  simp only [← c109524, ← c109525, ← c109526] at e_1792_4
  have e_1792_5 := arithEq_of_rows h (row := 1792) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_5
  simp only [← c109527, ← c109528, k2, ← c109529] at e_1792_5
  have hr := e_1792_5
  simp only [e_1792_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1584)) := by
  have c109530 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109531 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109532 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109533 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109534 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109535 := (publicBatchWrapper89_copies3422 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_6 := arithEq_of_rows h (row := 1792) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_6
  simp only [← c109530, ← c109531, ← c109532] at e_1792_6
  have e_1792_7 := arithEq_of_rows h (row := 1792) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_7
  simp only [← c109533, ← c109534, k2, ← c109535] at e_1792_7
  have hr := e_1792_7
  simp only [e_1792_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1585)) := by
  have c109536 := (publicBatchWrapper89_copies3423 a h).1
  have c109537 := (publicBatchWrapper89_copies3423 a h).2.1
  have c109538 := (publicBatchWrapper89_copies3423 a h).2.2.1
  have c109539 := (publicBatchWrapper89_copies3423 a h).2.2.2.1
  have c109540 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.1
  have c109541 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_8 := arithEq_of_rows h (row := 1792) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_8
  simp only [← c109536, ← c109537, ← c109538] at e_1792_8
  have e_1792_9 := arithEq_of_rows h (row := 1792) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_9
  simp only [← c109539, ← c109540, k2, ← c109541] at e_1792_9
  have hr := e_1792_9
  simp only [e_1792_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1586)) := by
  have c109542 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.1
  have c109543 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.1
  have c109544 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.1
  have c109545 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.1
  have c109546 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109547 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_10 := arithEq_of_rows h (row := 1792) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_10
  simp only [← c109542, ← c109543, ← c109544] at e_1792_10
  have e_1792_11 := arithEq_of_rows h (row := 1792) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_11
  simp only [← c109545, ← c109546, k2, ← c109547] at e_1792_11
  have hr := e_1792_11
  simp only [e_1792_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1587)) := by
  have c109548 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109549 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109550 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109551 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109552 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109553 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_12 := arithEq_of_rows h (row := 1792) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_12
  simp only [← c109548, ← c109549, ← c109550] at e_1792_12
  have e_1792_13 := arithEq_of_rows h (row := 1792) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_13
  simp only [← c109551, ← c109552, k2, ← c109553] at e_1792_13
  have hr := e_1792_13
  simp only [e_1792_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1588)) := by
  have c109554 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109555 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109556 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109557 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109558 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109559 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_14 := arithEq_of_rows h (row := 1792) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_14
  simp only [← c109554, ← c109555, ← c109556] at e_1792_14
  have e_1792_15 := arithEq_of_rows h (row := 1792) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_15
  simp only [← c109557, ← c109558, k2, ← c109559] at e_1792_15
  have hr := e_1792_15
  simp only [e_1792_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1589)) := by
  have c109560 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109561 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109562 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109563 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109564 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109565 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_16 := arithEq_of_rows h (row := 1792) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_16
  simp only [← c109560, ← c109561, ← c109562] at e_1792_16
  have e_1792_17 := arithEq_of_rows h (row := 1792) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_17
  simp only [← c109563, ← c109564, k2, ← c109565] at e_1792_17
  have hr := e_1792_17
  simp only [e_1792_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1792 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1590)) := by
  have c109566 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109567 := (publicBatchWrapper89_copies3423 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109568 := (publicBatchWrapper89_copies3424 a h).1
  have c109569 := (publicBatchWrapper89_copies3424 a h).2.1
  have c109570 := (publicBatchWrapper89_copies3424 a h).2.2.1
  have c109571 := (publicBatchWrapper89_copies3424 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1792_18 := arithEq_of_rows h (row := 1792) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_18
  simp only [← c109566, ← c109567, ← c109568] at e_1792_18
  have e_1792_19 := arithEq_of_rows h (row := 1792) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1792_19
  simp only [← c109569, ← c109570, k2, ← c109571] at e_1792_19
  have hr := e_1792_19
  simp only [e_1792_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1591)) := by
  have c109572 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.1
  have c109573 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.1
  have c109574 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.1
  have c109575 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.1
  have c109576 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.1
  have c109577 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_0 := arithEq_of_rows h (row := 1793) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_0
  simp only [← c109572, ← c109573, ← c109574] at e_1793_0
  have e_1793_1 := arithEq_of_rows h (row := 1793) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_1
  simp only [← c109575, ← c109576, k2, ← c109577] at e_1793_1
  have hr := e_1793_1
  simp only [e_1793_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 15) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1592)) := by
  have c109578 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109579 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109580 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109581 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109582 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109583 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_2 := arithEq_of_rows h (row := 1793) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_2
  simp only [← c109578, ← c109579, ← c109580] at e_1793_2
  have e_1793_3 := arithEq_of_rows h (row := 1793) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_3
  simp only [← c109581, ← c109582, k2, ← c109583] at e_1793_3
  have hr := e_1793_3
  simp only [e_1793_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 23) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1593)) := by
  have c109584 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109585 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109586 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109587 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109588 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109589 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_4 := arithEq_of_rows h (row := 1793) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_4
  simp only [← c109584, ← c109585, ← c109586] at e_1793_4
  have e_1793_5 := arithEq_of_rows h (row := 1793) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_5
  simp only [← c109587, ← c109588, k2, ← c109589] at e_1793_5
  have hr := e_1793_5
  simp only [e_1793_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 31) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1594)) := by
  have c109590 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109591 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109592 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109593 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109594 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109595 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_6 := arithEq_of_rows h (row := 1793) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_6
  simp only [← c109590, ← c109591, ← c109592] at e_1793_6
  have e_1793_7 := arithEq_of_rows h (row := 1793) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_7
  simp only [← c109593, ← c109594, k2, ← c109595] at e_1793_7
  have hr := e_1793_7
  simp only [e_1793_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 39) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1595)) := by
  have c109596 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109597 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109598 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109599 := (publicBatchWrapper89_copies3424 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109600 := (publicBatchWrapper89_copies3425 a h).1
  have c109601 := (publicBatchWrapper89_copies3425 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_8 := arithEq_of_rows h (row := 1793) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_8
  simp only [← c109596, ← c109597, ← c109598] at e_1793_8
  have e_1793_9 := arithEq_of_rows h (row := 1793) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_9
  simp only [← c109599, ← c109600, k2, ← c109601] at e_1793_9
  have hr := e_1793_9
  simp only [e_1793_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 47) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1596)) := by
  have c109602 := (publicBatchWrapper89_copies3425 a h).2.2.1
  have c109603 := (publicBatchWrapper89_copies3425 a h).2.2.2.1
  have c109604 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.1
  have c109605 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.1
  have c109606 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.1
  have c109607 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_10 := arithEq_of_rows h (row := 1793) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_10
  simp only [← c109602, ← c109603, ← c109604] at e_1793_10
  have e_1793_11 := arithEq_of_rows h (row := 1793) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_11
  simp only [← c109605, ← c109606, k2, ← c109607] at e_1793_11
  have hr := e_1793_11
  simp only [e_1793_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 55) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1597)) := by
  have c109608 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.1
  have c109609 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.1
  have c109610 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109611 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109612 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109613 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_12 := arithEq_of_rows h (row := 1793) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_12
  simp only [← c109608, ← c109609, ← c109610] at e_1793_12
  have e_1793_13 := arithEq_of_rows h (row := 1793) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_13
  simp only [← c109611, ← c109612, k2, ← c109613] at e_1793_13
  have hr := e_1793_13
  simp only [e_1793_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 63) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1598)) := by
  have c109614 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109615 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109616 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109617 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109618 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109619 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_14 := arithEq_of_rows h (row := 1793) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_14
  simp only [← c109614, ← c109615, ← c109616] at e_1793_14
  have e_1793_15 := arithEq_of_rows h (row := 1793) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_15
  simp only [← c109617, ← c109618, k2, ← c109619] at e_1793_15
  have hr := e_1793_15
  simp only [e_1793_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 71) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1599)) := by
  have c109620 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109621 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109622 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109623 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109624 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109625 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_16 := arithEq_of_rows h (row := 1793) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_16
  simp only [← c109620, ← c109621, ← c109622] at e_1793_16
  have e_1793_17 := arithEq_of_rows h (row := 1793) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_17
  simp only [← c109623, ← c109624, k2, ← c109625] at e_1793_17
  have hr := e_1793_17
  simp only [e_1793_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1793 79) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1600)) := by
  have c109626 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109627 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109628 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109629 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109630 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109631 := (publicBatchWrapper89_copies3425 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1793_18 := arithEq_of_rows h (row := 1793) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_18
  simp only [← c109626, ← c109627, ← c109628] at e_1793_18
  have e_1793_19 := arithEq_of_rows h (row := 1793) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1793_19
  simp only [← c109629, ← c109630, k2, ← c109631] at e_1793_19
  have hr := e_1793_19
  simp only [e_1793_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 7) = bselect (a (.wire 8 47)) (a (.virt 200)) (a (.virt 1601)) := by
  have c109632 := (publicBatchWrapper89_copies3426 a h).1
  have c109633 := (publicBatchWrapper89_copies3426 a h).2.1
  have c109634 := (publicBatchWrapper89_copies3426 a h).2.2.1
  have c109635 := (publicBatchWrapper89_copies3426 a h).2.2.2.1
  have c109636 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.1
  have c109637 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_0 := arithEq_of_rows h (row := 1794) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_0
  simp only [← c109632, ← c109633, ← c109634] at e_1794_0
  have e_1794_1 := arithEq_of_rows h (row := 1794) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_1
  simp only [← c109635, ← c109636, k2, ← c109637] at e_1794_1
  have hr := e_1794_1
  simp only [e_1794_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 15) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1770)) := by
  have c109638 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.1
  have c109639 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.1
  have c109640 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.1
  have c109641 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.1
  have c109642 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109643 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_2 := arithEq_of_rows h (row := 1794) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_2
  simp only [← c109638, ← c109639, ← c109640] at e_1794_2
  have e_1794_3 := arithEq_of_rows h (row := 1794) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_3
  simp only [← c109641, ← c109642, k2, ← c109643] at e_1794_3
  have hr := e_1794_3
  simp only [e_1794_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 23) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1771)) := by
  have c109644 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109645 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109646 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109647 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109648 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109649 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_4 := arithEq_of_rows h (row := 1794) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_4
  simp only [← c109644, ← c109645, ← c109646] at e_1794_4
  have e_1794_5 := arithEq_of_rows h (row := 1794) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_5
  simp only [← c109647, ← c109648, k2, ← c109649] at e_1794_5
  have hr := e_1794_5
  simp only [e_1794_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 31) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1772)) := by
  have c109650 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109651 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109652 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109653 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109654 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109655 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_6 := arithEq_of_rows h (row := 1794) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_6
  simp only [← c109650, ← c109651, ← c109652] at e_1794_6
  have e_1794_7 := arithEq_of_rows h (row := 1794) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_7
  simp only [← c109653, ← c109654, k2, ← c109655] at e_1794_7
  have hr := e_1794_7
  simp only [e_1794_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 39) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1773)) := by
  have c109656 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109657 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109658 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109659 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109660 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109661 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_8 := arithEq_of_rows h (row := 1794) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_8
  simp only [← c109656, ← c109657, ← c109658] at e_1794_8
  have e_1794_9 := arithEq_of_rows h (row := 1794) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_9
  simp only [← c109659, ← c109660, k2, ← c109661] at e_1794_9
  have hr := e_1794_9
  simp only [e_1794_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 47) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1774)) := by
  have c109662 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109663 := (publicBatchWrapper89_copies3426 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109664 := (publicBatchWrapper89_copies3427 a h).1
  have c109665 := (publicBatchWrapper89_copies3427 a h).2.1
  have c109666 := (publicBatchWrapper89_copies3427 a h).2.2.1
  have c109667 := (publicBatchWrapper89_copies3427 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_10 := arithEq_of_rows h (row := 1794) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_10
  simp only [← c109662, ← c109663, ← c109664] at e_1794_10
  have e_1794_11 := arithEq_of_rows h (row := 1794) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_11
  simp only [← c109665, ← c109666, k2, ← c109667] at e_1794_11
  have hr := e_1794_11
  simp only [e_1794_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 55) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1775)) := by
  have c109668 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.1
  have c109669 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.1
  have c109670 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.1
  have c109671 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.1
  have c109672 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.1
  have c109673 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_12 := arithEq_of_rows h (row := 1794) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_12
  simp only [← c109668, ← c109669, ← c109670] at e_1794_12
  have e_1794_13 := arithEq_of_rows h (row := 1794) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_13
  simp only [← c109671, ← c109672, k2, ← c109673] at e_1794_13
  have hr := e_1794_13
  simp only [e_1794_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 63) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1776)) := by
  have c109674 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109675 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109676 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109677 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109678 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109679 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_14 := arithEq_of_rows h (row := 1794) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_14
  simp only [← c109674, ← c109675, ← c109676] at e_1794_14
  have e_1794_15 := arithEq_of_rows h (row := 1794) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_15
  simp only [← c109677, ← c109678, k2, ← c109679] at e_1794_15
  have hr := e_1794_15
  simp only [e_1794_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 71) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1777)) := by
  have c109680 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109681 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109682 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109683 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109684 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109685 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_16 := arithEq_of_rows h (row := 1794) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_16
  simp only [← c109680, ← c109681, ← c109682] at e_1794_16
  have e_1794_17 := arithEq_of_rows h (row := 1794) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_17
  simp only [← c109683, ← c109684, k2, ← c109685] at e_1794_17
  have hr := e_1794_17
  simp only [e_1794_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1794 79) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1778)) := by
  have c109686 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109687 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109688 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109689 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109690 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109691 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1794_18 := arithEq_of_rows h (row := 1794) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_18
  simp only [← c109686, ← c109687, ← c109688] at e_1794_18
  have e_1794_19 := arithEq_of_rows h (row := 1794) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1794_19
  simp only [← c109689, ← c109690, k2, ← c109691] at e_1794_19
  have hr := e_1794_19
  simp only [e_1794_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 7) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1779)) := by
  have c109692 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109693 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109694 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109695 := (publicBatchWrapper89_copies3427 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109696 := (publicBatchWrapper89_copies3428 a h).1
  have c109697 := (publicBatchWrapper89_copies3428 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_0 := arithEq_of_rows h (row := 1795) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_0
  simp only [← c109692, ← c109693, ← c109694] at e_1795_0
  have e_1795_1 := arithEq_of_rows h (row := 1795) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_1
  simp only [← c109695, ← c109696, k2, ← c109697] at e_1795_1
  have hr := e_1795_1
  simp only [e_1795_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 15) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1780)) := by
  have c109698 := (publicBatchWrapper89_copies3428 a h).2.2.1
  have c109699 := (publicBatchWrapper89_copies3428 a h).2.2.2.1
  have c109700 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.1
  have c109701 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.1
  have c109702 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.1
  have c109703 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_2 := arithEq_of_rows h (row := 1795) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_2
  simp only [← c109698, ← c109699, ← c109700] at e_1795_2
  have e_1795_3 := arithEq_of_rows h (row := 1795) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_3
  simp only [← c109701, ← c109702, k2, ← c109703] at e_1795_3
  have hr := e_1795_3
  simp only [e_1795_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 23) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1781)) := by
  have c109704 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.1
  have c109705 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.1
  have c109706 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109707 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109708 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109709 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_4 := arithEq_of_rows h (row := 1795) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_4
  simp only [← c109704, ← c109705, ← c109706] at e_1795_4
  have e_1795_5 := arithEq_of_rows h (row := 1795) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_5
  simp only [← c109707, ← c109708, k2, ← c109709] at e_1795_5
  have hr := e_1795_5
  simp only [e_1795_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 31) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1782)) := by
  have c109710 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109711 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109712 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109713 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109714 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109715 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_6 := arithEq_of_rows h (row := 1795) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_6
  simp only [← c109710, ← c109711, ← c109712] at e_1795_6
  have e_1795_7 := arithEq_of_rows h (row := 1795) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_7
  simp only [← c109713, ← c109714, k2, ← c109715] at e_1795_7
  have hr := e_1795_7
  simp only [e_1795_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 39) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1783)) := by
  have c109716 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109717 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109718 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109719 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109720 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109721 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_8 := arithEq_of_rows h (row := 1795) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_8
  simp only [← c109716, ← c109717, ← c109718] at e_1795_8
  have e_1795_9 := arithEq_of_rows h (row := 1795) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_9
  simp only [← c109719, ← c109720, k2, ← c109721] at e_1795_9
  have hr := e_1795_9
  simp only [e_1795_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 47) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1784)) := by
  have c109722 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109723 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109724 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109725 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109726 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109727 := (publicBatchWrapper89_copies3428 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_10 := arithEq_of_rows h (row := 1795) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_10
  simp only [← c109722, ← c109723, ← c109724] at e_1795_10
  have e_1795_11 := arithEq_of_rows h (row := 1795) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_11
  simp only [← c109725, ← c109726, k2, ← c109727] at e_1795_11
  have hr := e_1795_11
  simp only [e_1795_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 55) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1785)) := by
  have c109728 := (publicBatchWrapper89_copies3429 a h).1
  have c109729 := (publicBatchWrapper89_copies3429 a h).2.1
  have c109730 := (publicBatchWrapper89_copies3429 a h).2.2.1
  have c109731 := (publicBatchWrapper89_copies3429 a h).2.2.2.1
  have c109732 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.1
  have c109733 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_12 := arithEq_of_rows h (row := 1795) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_12
  simp only [← c109728, ← c109729, ← c109730] at e_1795_12
  have e_1795_13 := arithEq_of_rows h (row := 1795) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_13
  simp only [← c109731, ← c109732, k2, ← c109733] at e_1795_13
  have hr := e_1795_13
  simp only [e_1795_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 63) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1786)) := by
  have c109734 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.1
  have c109735 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.1
  have c109736 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.1
  have c109737 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.1
  have c109738 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109739 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_14 := arithEq_of_rows h (row := 1795) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_14
  simp only [← c109734, ← c109735, ← c109736] at e_1795_14
  have e_1795_15 := arithEq_of_rows h (row := 1795) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_15
  simp only [← c109737, ← c109738, k2, ← c109739] at e_1795_15
  have hr := e_1795_15
  simp only [e_1795_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 71) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1787)) := by
  have c109740 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109741 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109742 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109743 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109744 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109745 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_16 := arithEq_of_rows h (row := 1795) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_16
  simp only [← c109740, ← c109741, ← c109742] at e_1795_16
  have e_1795_17 := arithEq_of_rows h (row := 1795) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_17
  simp only [← c109743, ← c109744, k2, ← c109745] at e_1795_17
  have hr := e_1795_17
  simp only [e_1795_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1795 79) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1788)) := by
  have c109746 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109747 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109748 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109749 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109750 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109751 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1795_18 := arithEq_of_rows h (row := 1795) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_18
  simp only [← c109746, ← c109747, ← c109748] at e_1795_18
  have e_1795_19 := arithEq_of_rows h (row := 1795) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1795_19
  simp only [← c109749, ← c109750, k2, ← c109751] at e_1795_19
  have hr := e_1795_19
  simp only [e_1795_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 7) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1789)) := by
  have c109752 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109753 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109754 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109755 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109756 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109757 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_0 := arithEq_of_rows h (row := 1796) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_0
  simp only [← c109752, ← c109753, ← c109754] at e_1796_0
  have e_1796_1 := arithEq_of_rows h (row := 1796) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_1
  simp only [← c109755, ← c109756, k2, ← c109757] at e_1796_1
  have hr := e_1796_1
  simp only [e_1796_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 15) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1790)) := by
  have c109758 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109759 := (publicBatchWrapper89_copies3429 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109760 := (publicBatchWrapper89_copies3430 a h).1
  have c109761 := (publicBatchWrapper89_copies3430 a h).2.1
  have c109762 := (publicBatchWrapper89_copies3430 a h).2.2.1
  have c109763 := (publicBatchWrapper89_copies3430 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_2 := arithEq_of_rows h (row := 1796) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_2
  simp only [← c109758, ← c109759, ← c109760] at e_1796_2
  have e_1796_3 := arithEq_of_rows h (row := 1796) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_3
  simp only [← c109761, ← c109762, k2, ← c109763] at e_1796_3
  have hr := e_1796_3
  simp only [e_1796_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 23) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1791)) := by
  have c109764 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.1
  have c109765 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.1
  have c109766 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.1
  have c109767 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.1
  have c109768 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.1
  have c109769 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_4 := arithEq_of_rows h (row := 1796) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_4
  simp only [← c109764, ← c109765, ← c109766] at e_1796_4
  have e_1796_5 := arithEq_of_rows h (row := 1796) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_5
  simp only [← c109767, ← c109768, k2, ← c109769] at e_1796_5
  have hr := e_1796_5
  simp only [e_1796_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 31) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1792)) := by
  have c109770 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109771 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109772 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109773 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109774 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109775 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_6 := arithEq_of_rows h (row := 1796) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_6
  simp only [← c109770, ← c109771, ← c109772] at e_1796_6
  have e_1796_7 := arithEq_of_rows h (row := 1796) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_7
  simp only [← c109773, ← c109774, k2, ← c109775] at e_1796_7
  have hr := e_1796_7
  simp only [e_1796_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 39) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1793)) := by
  have c109776 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109777 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109778 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109779 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109780 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109781 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_8 := arithEq_of_rows h (row := 1796) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_8
  simp only [← c109776, ← c109777, ← c109778] at e_1796_8
  have e_1796_9 := arithEq_of_rows h (row := 1796) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_9
  simp only [← c109779, ← c109780, k2, ← c109781] at e_1796_9
  have hr := e_1796_9
  simp only [e_1796_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 47) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1794)) := by
  have c109782 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109783 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109784 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109785 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109786 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109787 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_10 := arithEq_of_rows h (row := 1796) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_10
  simp only [← c109782, ← c109783, ← c109784] at e_1796_10
  have e_1796_11 := arithEq_of_rows h (row := 1796) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_11
  simp only [← c109785, ← c109786, k2, ← c109787] at e_1796_11
  have hr := e_1796_11
  simp only [e_1796_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 55) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1795)) := by
  have c109788 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109789 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109790 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109791 := (publicBatchWrapper89_copies3430 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109792 := (publicBatchWrapper89_copies3431 a h).1
  have c109793 := (publicBatchWrapper89_copies3431 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_12 := arithEq_of_rows h (row := 1796) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_12
  simp only [← c109788, ← c109789, ← c109790] at e_1796_12
  have e_1796_13 := arithEq_of_rows h (row := 1796) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_13
  simp only [← c109791, ← c109792, k2, ← c109793] at e_1796_13
  have hr := e_1796_13
  simp only [e_1796_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 63) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1796)) := by
  have c109794 := (publicBatchWrapper89_copies3431 a h).2.2.1
  have c109795 := (publicBatchWrapper89_copies3431 a h).2.2.2.1
  have c109796 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.1
  have c109797 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.1
  have c109798 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.1
  have c109799 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_14 := arithEq_of_rows h (row := 1796) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_14
  simp only [← c109794, ← c109795, ← c109796] at e_1796_14
  have e_1796_15 := arithEq_of_rows h (row := 1796) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_15
  simp only [← c109797, ← c109798, k2, ← c109799] at e_1796_15
  have hr := e_1796_15
  simp only [e_1796_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 71) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1797)) := by
  have c109800 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.1
  have c109801 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.1
  have c109802 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109803 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109804 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109805 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_16 := arithEq_of_rows h (row := 1796) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_16
  simp only [← c109800, ← c109801, ← c109802] at e_1796_16
  have e_1796_17 := arithEq_of_rows h (row := 1796) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_17
  simp only [← c109803, ← c109804, k2, ← c109805] at e_1796_17
  have hr := e_1796_17
  simp only [e_1796_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1796 79) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1798)) := by
  have c109806 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109807 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109808 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109809 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109810 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109811 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1796_18 := arithEq_of_rows h (row := 1796) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_18
  simp only [← c109806, ← c109807, ← c109808] at e_1796_18
  have e_1796_19 := arithEq_of_rows h (row := 1796) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1796_19
  simp only [← c109809, ← c109810, k2, ← c109811] at e_1796_19
  have hr := e_1796_19
  simp only [e_1796_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 7) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1799)) := by
  have c109812 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109813 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109814 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109815 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109816 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109817 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_0 := arithEq_of_rows h (row := 1797) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_0
  simp only [← c109812, ← c109813, ← c109814] at e_1797_0
  have e_1797_1 := arithEq_of_rows h (row := 1797) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_1
  simp only [← c109815, ← c109816, k2, ← c109817] at e_1797_1
  have hr := e_1797_1
  simp only [e_1797_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 15) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1800)) := by
  have c109818 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109819 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109820 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109821 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109822 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109823 := (publicBatchWrapper89_copies3431 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_2 := arithEq_of_rows h (row := 1797) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_2
  simp only [← c109818, ← c109819, ← c109820] at e_1797_2
  have e_1797_3 := arithEq_of_rows h (row := 1797) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_3
  simp only [← c109821, ← c109822, k2, ← c109823] at e_1797_3
  have hr := e_1797_3
  simp only [e_1797_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 23) = bselect (a (.wire 10 11)) (a (.virt 200)) (a (.virt 1801)) := by
  have c109824 := (publicBatchWrapper89_copies3432 a h).1
  have c109825 := (publicBatchWrapper89_copies3432 a h).2.1
  have c109826 := (publicBatchWrapper89_copies3432 a h).2.2.1
  have c109827 := (publicBatchWrapper89_copies3432 a h).2.2.2.1
  have c109828 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.1
  have c109829 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_4 := arithEq_of_rows h (row := 1797) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_4
  simp only [← c109824, ← c109825, ← c109826] at e_1797_4
  have e_1797_5 := arithEq_of_rows h (row := 1797) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_5
  simp only [← c109827, ← c109828, k2, ← c109829] at e_1797_5
  have hr := e_1797_5
  simp only [e_1797_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 31) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1970)) := by
  have c109830 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.1
  have c109831 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.1
  have c109832 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.1
  have c109833 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.1
  have c109834 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109835 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_6 := arithEq_of_rows h (row := 1797) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_6
  simp only [← c109830, ← c109831, ← c109832] at e_1797_6
  have e_1797_7 := arithEq_of_rows h (row := 1797) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_7
  simp only [← c109833, ← c109834, k2, ← c109835] at e_1797_7
  have hr := e_1797_7
  simp only [e_1797_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 39) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1971)) := by
  have c109836 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109837 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109838 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109839 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109840 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109841 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_8 := arithEq_of_rows h (row := 1797) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_8
  simp only [← c109836, ← c109837, ← c109838] at e_1797_8
  have e_1797_9 := arithEq_of_rows h (row := 1797) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_9
  simp only [← c109839, ← c109840, k2, ← c109841] at e_1797_9
  have hr := e_1797_9
  simp only [e_1797_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 47) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1972)) := by
  have c109842 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109843 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109844 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109845 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109846 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109847 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_10 := arithEq_of_rows h (row := 1797) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_10
  simp only [← c109842, ← c109843, ← c109844] at e_1797_10
  have e_1797_11 := arithEq_of_rows h (row := 1797) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_11
  simp only [← c109845, ← c109846, k2, ← c109847] at e_1797_11
  have hr := e_1797_11
  simp only [e_1797_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 55) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1973)) := by
  have c109848 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109849 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109850 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109851 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109852 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109853 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_12 := arithEq_of_rows h (row := 1797) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_12
  simp only [← c109848, ← c109849, ← c109850] at e_1797_12
  have e_1797_13 := arithEq_of_rows h (row := 1797) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_13
  simp only [← c109851, ← c109852, k2, ← c109853] at e_1797_13
  have hr := e_1797_13
  simp only [e_1797_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 63) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1974)) := by
  have c109854 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109855 := (publicBatchWrapper89_copies3432 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109856 := (publicBatchWrapper89_copies3433 a h).1
  have c109857 := (publicBatchWrapper89_copies3433 a h).2.1
  have c109858 := (publicBatchWrapper89_copies3433 a h).2.2.1
  have c109859 := (publicBatchWrapper89_copies3433 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_14 := arithEq_of_rows h (row := 1797) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_14
  simp only [← c109854, ← c109855, ← c109856] at e_1797_14
  have e_1797_15 := arithEq_of_rows h (row := 1797) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_15
  simp only [← c109857, ← c109858, k2, ← c109859] at e_1797_15
  have hr := e_1797_15
  simp only [e_1797_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 71) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1975)) := by
  have c109860 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.1
  have c109861 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.1
  have c109862 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.1
  have c109863 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.1
  have c109864 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.1
  have c109865 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_16 := arithEq_of_rows h (row := 1797) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_16
  simp only [← c109860, ← c109861, ← c109862] at e_1797_16
  have e_1797_17 := arithEq_of_rows h (row := 1797) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_17
  simp only [← c109863, ← c109864, k2, ← c109865] at e_1797_17
  have hr := e_1797_17
  simp only [e_1797_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1797 79) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1976)) := by
  have c109866 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109867 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109868 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109869 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109870 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109871 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1797_18 := arithEq_of_rows h (row := 1797) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_18
  simp only [← c109866, ← c109867, ← c109868] at e_1797_18
  have e_1797_19 := arithEq_of_rows h (row := 1797) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1797_19
  simp only [← c109869, ← c109870, k2, ← c109871] at e_1797_19
  have hr := e_1797_19
  simp only [e_1797_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 7) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1977)) := by
  have c109872 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109873 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109874 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109875 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109876 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109877 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_0 := arithEq_of_rows h (row := 1798) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_0
  simp only [← c109872, ← c109873, ← c109874] at e_1798_0
  have e_1798_1 := arithEq_of_rows h (row := 1798) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_1
  simp only [← c109875, ← c109876, k2, ← c109877] at e_1798_1
  have hr := e_1798_1
  simp only [e_1798_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 15) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1978)) := by
  have c109878 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109879 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109880 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109881 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109882 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109883 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_2 := arithEq_of_rows h (row := 1798) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_2
  simp only [← c109878, ← c109879, ← c109880] at e_1798_2
  have e_1798_3 := arithEq_of_rows h (row := 1798) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_3
  simp only [← c109881, ← c109882, k2, ← c109883] at e_1798_3
  have hr := e_1798_3
  simp only [e_1798_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 23) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1979)) := by
  have c109884 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109885 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109886 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109887 := (publicBatchWrapper89_copies3433 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109888 := (publicBatchWrapper89_copies3434 a h).1
  have c109889 := (publicBatchWrapper89_copies3434 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_4 := arithEq_of_rows h (row := 1798) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_4
  simp only [← c109884, ← c109885, ← c109886] at e_1798_4
  have e_1798_5 := arithEq_of_rows h (row := 1798) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_5
  simp only [← c109887, ← c109888, k2, ← c109889] at e_1798_5
  have hr := e_1798_5
  simp only [e_1798_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 31) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1980)) := by
  have c109890 := (publicBatchWrapper89_copies3434 a h).2.2.1
  have c109891 := (publicBatchWrapper89_copies3434 a h).2.2.2.1
  have c109892 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.1
  have c109893 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.1
  have c109894 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.1
  have c109895 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_6 := arithEq_of_rows h (row := 1798) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_6
  simp only [← c109890, ← c109891, ← c109892] at e_1798_6
  have e_1798_7 := arithEq_of_rows h (row := 1798) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_7
  simp only [← c109893, ← c109894, k2, ← c109895] at e_1798_7
  have hr := e_1798_7
  simp only [e_1798_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 39) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1981)) := by
  have c109896 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.1
  have c109897 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.1
  have c109898 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109899 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109900 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109901 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_8 := arithEq_of_rows h (row := 1798) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_8
  simp only [← c109896, ← c109897, ← c109898] at e_1798_8
  have e_1798_9 := arithEq_of_rows h (row := 1798) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_9
  simp only [← c109899, ← c109900, k2, ← c109901] at e_1798_9
  have hr := e_1798_9
  simp only [e_1798_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 47) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1982)) := by
  have c109902 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109903 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109904 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109905 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109906 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109907 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_10 := arithEq_of_rows h (row := 1798) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_10
  simp only [← c109902, ← c109903, ← c109904] at e_1798_10
  have e_1798_11 := arithEq_of_rows h (row := 1798) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_11
  simp only [← c109905, ← c109906, k2, ← c109907] at e_1798_11
  have hr := e_1798_11
  simp only [e_1798_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 55) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1983)) := by
  have c109908 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109909 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109910 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109911 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109912 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109913 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_12 := arithEq_of_rows h (row := 1798) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_12
  simp only [← c109908, ← c109909, ← c109910] at e_1798_12
  have e_1798_13 := arithEq_of_rows h (row := 1798) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_13
  simp only [← c109911, ← c109912, k2, ← c109913] at e_1798_13
  have hr := e_1798_13
  simp only [e_1798_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 63) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1984)) := by
  have c109914 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109915 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109916 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109917 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109918 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109919 := (publicBatchWrapper89_copies3434 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_14 := arithEq_of_rows h (row := 1798) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_14
  simp only [← c109914, ← c109915, ← c109916] at e_1798_14
  have e_1798_15 := arithEq_of_rows h (row := 1798) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_15
  simp only [← c109917, ← c109918, k2, ← c109919] at e_1798_15
  have hr := e_1798_15
  simp only [e_1798_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 71) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1985)) := by
  have c109920 := (publicBatchWrapper89_copies3435 a h).1
  have c109921 := (publicBatchWrapper89_copies3435 a h).2.1
  have c109922 := (publicBatchWrapper89_copies3435 a h).2.2.1
  have c109923 := (publicBatchWrapper89_copies3435 a h).2.2.2.1
  have c109924 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.1
  have c109925 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_16 := arithEq_of_rows h (row := 1798) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_16
  simp only [← c109920, ← c109921, ← c109922] at e_1798_16
  have e_1798_17 := arithEq_of_rows h (row := 1798) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_17
  simp only [← c109923, ← c109924, k2, ← c109925] at e_1798_17
  have hr := e_1798_17
  simp only [e_1798_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1798 79) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1986)) := by
  have c109926 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.1
  have c109927 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.1
  have c109928 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.1
  have c109929 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.1
  have c109930 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109931 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1798_18 := arithEq_of_rows h (row := 1798) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_18
  simp only [← c109926, ← c109927, ← c109928] at e_1798_18
  have e_1798_19 := arithEq_of_rows h (row := 1798) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1798_19
  simp only [← c109929, ← c109930, k2, ← c109931] at e_1798_19
  have hr := e_1798_19
  simp only [e_1798_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 7) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1987)) := by
  have c109932 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109933 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109934 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109935 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109936 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109937 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_0 := arithEq_of_rows h (row := 1799) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_0
  simp only [← c109932, ← c109933, ← c109934] at e_1799_0
  have e_1799_1 := arithEq_of_rows h (row := 1799) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_1
  simp only [← c109935, ← c109936, k2, ← c109937] at e_1799_1
  have hr := e_1799_1
  simp only [e_1799_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 15) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1988)) := by
  have c109938 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109939 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109940 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109941 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109942 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109943 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_2 := arithEq_of_rows h (row := 1799) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_2
  simp only [← c109938, ← c109939, ← c109940] at e_1799_2
  have e_1799_3 := arithEq_of_rows h (row := 1799) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_3
  simp only [← c109941, ← c109942, k2, ← c109943] at e_1799_3
  have hr := e_1799_3
  simp only [e_1799_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 23) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1989)) := by
  have c109944 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109945 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109946 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109947 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109948 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109949 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_4 := arithEq_of_rows h (row := 1799) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_4
  simp only [← c109944, ← c109945, ← c109946] at e_1799_4
  have e_1799_5 := arithEq_of_rows h (row := 1799) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_5
  simp only [← c109947, ← c109948, k2, ← c109949] at e_1799_5
  have hr := e_1799_5
  simp only [e_1799_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 31) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1990)) := by
  have c109950 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109951 := (publicBatchWrapper89_copies3435 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109952 := (publicBatchWrapper89_copies3436 a h).1
  have c109953 := (publicBatchWrapper89_copies3436 a h).2.1
  have c109954 := (publicBatchWrapper89_copies3436 a h).2.2.1
  have c109955 := (publicBatchWrapper89_copies3436 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_6 := arithEq_of_rows h (row := 1799) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_6
  simp only [← c109950, ← c109951, ← c109952] at e_1799_6
  have e_1799_7 := arithEq_of_rows h (row := 1799) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_7
  simp only [← c109953, ← c109954, k2, ← c109955] at e_1799_7
  have hr := e_1799_7
  simp only [e_1799_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 39) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1991)) := by
  have c109956 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.1
  have c109957 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.1
  have c109958 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.1
  have c109959 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.1
  have c109960 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.1
  have c109961 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_8 := arithEq_of_rows h (row := 1799) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_8
  simp only [← c109956, ← c109957, ← c109958] at e_1799_8
  have e_1799_9 := arithEq_of_rows h (row := 1799) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_9
  simp only [← c109959, ← c109960, k2, ← c109961] at e_1799_9
  have hr := e_1799_9
  simp only [e_1799_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 47) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1992)) := by
  have c109962 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109963 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109964 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109965 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109966 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109967 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_10 := arithEq_of_rows h (row := 1799) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_10
  simp only [← c109962, ← c109963, ← c109964] at e_1799_10
  have e_1799_11 := arithEq_of_rows h (row := 1799) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_11
  simp only [← c109965, ← c109966, k2, ← c109967] at e_1799_11
  have hr := e_1799_11
  simp only [e_1799_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 55) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1993)) := by
  have c109968 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109969 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109970 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109971 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109972 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109973 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_12 := arithEq_of_rows h (row := 1799) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_12
  simp only [← c109968, ← c109969, ← c109970] at e_1799_12
  have e_1799_13 := arithEq_of_rows h (row := 1799) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_13
  simp only [← c109971, ← c109972, k2, ← c109973] at e_1799_13
  have hr := e_1799_13
  simp only [e_1799_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 63) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1994)) := by
  have c109974 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109975 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109976 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109977 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109978 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109979 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_14 := arithEq_of_rows h (row := 1799) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_14
  simp only [← c109974, ← c109975, ← c109976] at e_1799_14
  have e_1799_15 := arithEq_of_rows h (row := 1799) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_15
  simp only [← c109977, ← c109978, k2, ← c109979] at e_1799_15
  have hr := e_1799_15
  simp only [e_1799_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 71) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1995)) := by
  have c109980 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109981 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109982 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109983 := (publicBatchWrapper89_copies3436 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c109984 := (publicBatchWrapper89_copies3437 a h).1
  have c109985 := (publicBatchWrapper89_copies3437 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_16 := arithEq_of_rows h (row := 1799) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_16
  simp only [← c109980, ← c109981, ← c109982] at e_1799_16
  have e_1799_17 := arithEq_of_rows h (row := 1799) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_17
  simp only [← c109983, ← c109984, k2, ← c109985] at e_1799_17
  have hr := e_1799_17
  simp only [e_1799_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1799 79) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1996)) := by
  have c109986 := (publicBatchWrapper89_copies3437 a h).2.2.1
  have c109987 := (publicBatchWrapper89_copies3437 a h).2.2.2.1
  have c109988 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.1
  have c109989 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.1
  have c109990 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.1
  have c109991 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1799_18 := arithEq_of_rows h (row := 1799) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_18
  simp only [← c109986, ← c109987, ← c109988] at e_1799_18
  have e_1799_19 := arithEq_of_rows h (row := 1799) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1799_19
  simp only [← c109989, ← c109990, k2, ← c109991] at e_1799_19
  have hr := e_1799_19
  simp only [e_1799_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 7) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1997)) := by
  have c109992 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.1
  have c109993 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.1
  have c109994 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.1
  have c109995 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c109996 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109997 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_0 := arithEq_of_rows h (row := 1800) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_0
  simp only [← c109992, ← c109993, ← c109994] at e_1800_0
  have e_1800_1 := arithEq_of_rows h (row := 1800) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_1
  simp only [← c109995, ← c109996, k2, ← c109997] at e_1800_1
  have hr := e_1800_1
  simp only [e_1800_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 15) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1998)) := by
  have c109998 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c109999 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110000 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110001 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110002 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110003 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_2 := arithEq_of_rows h (row := 1800) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_2
  simp only [← c109998, ← c109999, ← c110000] at e_1800_2
  have e_1800_3 := arithEq_of_rows h (row := 1800) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_3
  simp only [← c110001, ← c110002, k2, ← c110003] at e_1800_3
  have hr := e_1800_3
  simp only [e_1800_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 23) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 1999)) := by
  have c110004 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110005 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110006 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110007 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110008 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110009 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_4 := arithEq_of_rows h (row := 1800) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_4
  simp only [← c110004, ← c110005, ← c110006] at e_1800_4
  have e_1800_5 := arithEq_of_rows h (row := 1800) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_5
  simp only [← c110007, ← c110008, k2, ← c110009] at e_1800_5
  have hr := e_1800_5
  simp only [e_1800_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 31) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 2000)) := by
  have c110010 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110011 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110012 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110013 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110014 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110015 := (publicBatchWrapper89_copies3437 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_6 := arithEq_of_rows h (row := 1800) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_6
  simp only [← c110010, ← c110011, ← c110012] at e_1800_6
  have e_1800_7 := arithEq_of_rows h (row := 1800) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_7
  simp only [← c110013, ← c110014, k2, ← c110015] at e_1800_7
  have hr := e_1800_7
  simp only [e_1800_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 39) = bselect (a (.wire 10 55)) (a (.virt 200)) (a (.virt 2001)) := by
  have c110016 := (publicBatchWrapper89_copies3438 a h).1
  have c110017 := (publicBatchWrapper89_copies3438 a h).2.1
  have c110018 := (publicBatchWrapper89_copies3438 a h).2.2.1
  have c110019 := (publicBatchWrapper89_copies3438 a h).2.2.2.1
  have c110020 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.1
  have c110021 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_8 := arithEq_of_rows h (row := 1800) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_8
  simp only [← c110016, ← c110017, ← c110018] at e_1800_8
  have e_1800_9 := arithEq_of_rows h (row := 1800) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_9
  simp only [← c110019, ← c110020, k2, ← c110021] at e_1800_9
  have hr := e_1800_9
  simp only [e_1800_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 47) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2170)) := by
  have c110022 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.1
  have c110023 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.1
  have c110024 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.1
  have c110025 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.1
  have c110026 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110027 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_10 := arithEq_of_rows h (row := 1800) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_10
  simp only [← c110022, ← c110023, ← c110024] at e_1800_10
  have e_1800_11 := arithEq_of_rows h (row := 1800) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_11
  simp only [← c110025, ← c110026, k2, ← c110027] at e_1800_11
  have hr := e_1800_11
  simp only [e_1800_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 55) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2171)) := by
  have c110028 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110029 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110030 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110031 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110032 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110033 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_12 := arithEq_of_rows h (row := 1800) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_12
  simp only [← c110028, ← c110029, ← c110030] at e_1800_12
  have e_1800_13 := arithEq_of_rows h (row := 1800) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_13
  simp only [← c110031, ← c110032, k2, ← c110033] at e_1800_13
  have hr := e_1800_13
  simp only [e_1800_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 63) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2172)) := by
  have c110034 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110035 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110036 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110037 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110038 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110039 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_14 := arithEq_of_rows h (row := 1800) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_14
  simp only [← c110034, ← c110035, ← c110036] at e_1800_14
  have e_1800_15 := arithEq_of_rows h (row := 1800) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_15
  simp only [← c110037, ← c110038, k2, ← c110039] at e_1800_15
  have hr := e_1800_15
  simp only [e_1800_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 71) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2173)) := by
  have c110040 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110041 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110042 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110043 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110044 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110045 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_16 := arithEq_of_rows h (row := 1800) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_16
  simp only [← c110040, ← c110041, ← c110042] at e_1800_16
  have e_1800_17 := arithEq_of_rows h (row := 1800) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_17
  simp only [← c110043, ← c110044, k2, ← c110045] at e_1800_17
  have hr := e_1800_17
  simp only [e_1800_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1800 79) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2174)) := by
  have c110046 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110047 := (publicBatchWrapper89_copies3438 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110048 := (publicBatchWrapper89_copies3439 a h).1
  have c110049 := (publicBatchWrapper89_copies3439 a h).2.1
  have c110050 := (publicBatchWrapper89_copies3439 a h).2.2.1
  have c110051 := (publicBatchWrapper89_copies3439 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1800_18 := arithEq_of_rows h (row := 1800) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_18
  simp only [← c110046, ← c110047, ← c110048] at e_1800_18
  have e_1800_19 := arithEq_of_rows h (row := 1800) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1800_19
  simp only [← c110049, ← c110050, k2, ← c110051] at e_1800_19
  have hr := e_1800_19
  simp only [e_1800_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 7) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2175)) := by
  have c110052 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.1
  have c110053 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.1
  have c110054 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.1
  have c110055 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.1
  have c110056 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.1
  have c110057 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_0 := arithEq_of_rows h (row := 1801) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_0
  simp only [← c110052, ← c110053, ← c110054] at e_1801_0
  have e_1801_1 := arithEq_of_rows h (row := 1801) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_1
  simp only [← c110055, ← c110056, k2, ← c110057] at e_1801_1
  have hr := e_1801_1
  simp only [e_1801_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 15) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2176)) := by
  have c110058 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110059 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110060 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110061 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110062 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110063 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_2 := arithEq_of_rows h (row := 1801) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_2
  simp only [← c110058, ← c110059, ← c110060] at e_1801_2
  have e_1801_3 := arithEq_of_rows h (row := 1801) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_3
  simp only [← c110061, ← c110062, k2, ← c110063] at e_1801_3
  have hr := e_1801_3
  simp only [e_1801_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 23) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2177)) := by
  have c110064 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110065 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110066 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110067 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110068 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110069 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_4 := arithEq_of_rows h (row := 1801) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_4
  simp only [← c110064, ← c110065, ← c110066] at e_1801_4
  have e_1801_5 := arithEq_of_rows h (row := 1801) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_5
  simp only [← c110067, ← c110068, k2, ← c110069] at e_1801_5
  have hr := e_1801_5
  simp only [e_1801_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 31) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2178)) := by
  have c110070 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110071 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110072 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110073 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110074 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110075 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_6 := arithEq_of_rows h (row := 1801) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_6
  simp only [← c110070, ← c110071, ← c110072] at e_1801_6
  have e_1801_7 := arithEq_of_rows h (row := 1801) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_7
  simp only [← c110073, ← c110074, k2, ← c110075] at e_1801_7
  have hr := e_1801_7
  simp only [e_1801_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 39) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2179)) := by
  have c110076 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110077 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110078 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110079 := (publicBatchWrapper89_copies3439 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110080 := (publicBatchWrapper89_copies3440 a h).1
  have c110081 := (publicBatchWrapper89_copies3440 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_8 := arithEq_of_rows h (row := 1801) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_8
  simp only [← c110076, ← c110077, ← c110078] at e_1801_8
  have e_1801_9 := arithEq_of_rows h (row := 1801) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_9
  simp only [← c110079, ← c110080, k2, ← c110081] at e_1801_9
  have hr := e_1801_9
  simp only [e_1801_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 47) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2180)) := by
  have c110082 := (publicBatchWrapper89_copies3440 a h).2.2.1
  have c110083 := (publicBatchWrapper89_copies3440 a h).2.2.2.1
  have c110084 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.1
  have c110085 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.1
  have c110086 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.1
  have c110087 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_10 := arithEq_of_rows h (row := 1801) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_10
  simp only [← c110082, ← c110083, ← c110084] at e_1801_10
  have e_1801_11 := arithEq_of_rows h (row := 1801) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_11
  simp only [← c110085, ← c110086, k2, ← c110087] at e_1801_11
  have hr := e_1801_11
  simp only [e_1801_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 55) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2181)) := by
  have c110088 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.1
  have c110089 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.1
  have c110090 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110091 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110092 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110093 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_12 := arithEq_of_rows h (row := 1801) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_12
  simp only [← c110088, ← c110089, ← c110090] at e_1801_12
  have e_1801_13 := arithEq_of_rows h (row := 1801) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_13
  simp only [← c110091, ← c110092, k2, ← c110093] at e_1801_13
  have hr := e_1801_13
  simp only [e_1801_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 63) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2182)) := by
  have c110094 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110095 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110096 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110097 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110098 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110099 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_14 := arithEq_of_rows h (row := 1801) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_14
  simp only [← c110094, ← c110095, ← c110096] at e_1801_14
  have e_1801_15 := arithEq_of_rows h (row := 1801) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_15
  simp only [← c110097, ← c110098, k2, ← c110099] at e_1801_15
  have hr := e_1801_15
  simp only [e_1801_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 71) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2183)) := by
  have c110100 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110101 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110102 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110103 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110104 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110105 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_16 := arithEq_of_rows h (row := 1801) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_16
  simp only [← c110100, ← c110101, ← c110102] at e_1801_16
  have e_1801_17 := arithEq_of_rows h (row := 1801) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_17
  simp only [← c110103, ← c110104, k2, ← c110105] at e_1801_17
  have hr := e_1801_17
  simp only [e_1801_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1801 79) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2184)) := by
  have c110106 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110107 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110108 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110109 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110110 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110111 := (publicBatchWrapper89_copies3440 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1801_18 := arithEq_of_rows h (row := 1801) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_18
  simp only [← c110106, ← c110107, ← c110108] at e_1801_18
  have e_1801_19 := arithEq_of_rows h (row := 1801) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1801_19
  simp only [← c110109, ← c110110, k2, ← c110111] at e_1801_19
  have hr := e_1801_19
  simp only [e_1801_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 7) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2185)) := by
  have c110112 := (publicBatchWrapper89_copies3441 a h).1
  have c110113 := (publicBatchWrapper89_copies3441 a h).2.1
  have c110114 := (publicBatchWrapper89_copies3441 a h).2.2.1
  have c110115 := (publicBatchWrapper89_copies3441 a h).2.2.2.1
  have c110116 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.1
  have c110117 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_0 := arithEq_of_rows h (row := 1802) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_0
  simp only [← c110112, ← c110113, ← c110114] at e_1802_0
  have e_1802_1 := arithEq_of_rows h (row := 1802) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_1
  simp only [← c110115, ← c110116, k2, ← c110117] at e_1802_1
  have hr := e_1802_1
  simp only [e_1802_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 15) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2186)) := by
  have c110118 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.1
  have c110119 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.1
  have c110120 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.1
  have c110121 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.1
  have c110122 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110123 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_2 := arithEq_of_rows h (row := 1802) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_2
  simp only [← c110118, ← c110119, ← c110120] at e_1802_2
  have e_1802_3 := arithEq_of_rows h (row := 1802) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_3
  simp only [← c110121, ← c110122, k2, ← c110123] at e_1802_3
  have hr := e_1802_3
  simp only [e_1802_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 23) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2187)) := by
  have c110124 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110125 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110126 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110127 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110128 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110129 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_4 := arithEq_of_rows h (row := 1802) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_4
  simp only [← c110124, ← c110125, ← c110126] at e_1802_4
  have e_1802_5 := arithEq_of_rows h (row := 1802) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_5
  simp only [← c110127, ← c110128, k2, ← c110129] at e_1802_5
  have hr := e_1802_5
  simp only [e_1802_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 31) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2188)) := by
  have c110130 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110131 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110132 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110133 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110134 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110135 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_6 := arithEq_of_rows h (row := 1802) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_6
  simp only [← c110130, ← c110131, ← c110132] at e_1802_6
  have e_1802_7 := arithEq_of_rows h (row := 1802) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_7
  simp only [← c110133, ← c110134, k2, ← c110135] at e_1802_7
  have hr := e_1802_7
  simp only [e_1802_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 39) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2189)) := by
  have c110136 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110137 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110138 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110139 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110140 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110141 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_8 := arithEq_of_rows h (row := 1802) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_8
  simp only [← c110136, ← c110137, ← c110138] at e_1802_8
  have e_1802_9 := arithEq_of_rows h (row := 1802) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_9
  simp only [← c110139, ← c110140, k2, ← c110141] at e_1802_9
  have hr := e_1802_9
  simp only [e_1802_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 47) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2190)) := by
  have c110142 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110143 := (publicBatchWrapper89_copies3441 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110144 := (publicBatchWrapper89_copies3442 a h).1
  have c110145 := (publicBatchWrapper89_copies3442 a h).2.1
  have c110146 := (publicBatchWrapper89_copies3442 a h).2.2.1
  have c110147 := (publicBatchWrapper89_copies3442 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_10 := arithEq_of_rows h (row := 1802) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_10
  simp only [← c110142, ← c110143, ← c110144] at e_1802_10
  have e_1802_11 := arithEq_of_rows h (row := 1802) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_11
  simp only [← c110145, ← c110146, k2, ← c110147] at e_1802_11
  have hr := e_1802_11
  simp only [e_1802_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 55) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2191)) := by
  have c110148 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.1
  have c110149 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.1
  have c110150 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.1
  have c110151 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.1
  have c110152 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.1
  have c110153 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_12 := arithEq_of_rows h (row := 1802) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_12
  simp only [← c110148, ← c110149, ← c110150] at e_1802_12
  have e_1802_13 := arithEq_of_rows h (row := 1802) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_13
  simp only [← c110151, ← c110152, k2, ← c110153] at e_1802_13
  have hr := e_1802_13
  simp only [e_1802_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 63) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2192)) := by
  have c110154 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110155 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110156 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110157 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110158 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110159 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_14 := arithEq_of_rows h (row := 1802) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_14
  simp only [← c110154, ← c110155, ← c110156] at e_1802_14
  have e_1802_15 := arithEq_of_rows h (row := 1802) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_15
  simp only [← c110157, ← c110158, k2, ← c110159] at e_1802_15
  have hr := e_1802_15
  simp only [e_1802_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 71) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2193)) := by
  have c110160 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110161 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110162 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110163 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110164 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110165 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_16 := arithEq_of_rows h (row := 1802) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_16
  simp only [← c110160, ← c110161, ← c110162] at e_1802_16
  have e_1802_17 := arithEq_of_rows h (row := 1802) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_17
  simp only [← c110163, ← c110164, k2, ← c110165] at e_1802_17
  have hr := e_1802_17
  simp only [e_1802_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1802 79) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2194)) := by
  have c110166 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110167 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110168 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110169 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110170 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110171 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1802_18 := arithEq_of_rows h (row := 1802) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_18
  simp only [← c110166, ← c110167, ← c110168] at e_1802_18
  have e_1802_19 := arithEq_of_rows h (row := 1802) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1802_19
  simp only [← c110169, ← c110170, k2, ← c110171] at e_1802_19
  have hr := e_1802_19
  simp only [e_1802_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 7) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2195)) := by
  have c110172 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110173 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110174 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110175 := (publicBatchWrapper89_copies3442 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110176 := (publicBatchWrapper89_copies3443 a h).1
  have c110177 := (publicBatchWrapper89_copies3443 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_0 := arithEq_of_rows h (row := 1803) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_0
  simp only [← c110172, ← c110173, ← c110174] at e_1803_0
  have e_1803_1 := arithEq_of_rows h (row := 1803) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_1
  simp only [← c110175, ← c110176, k2, ← c110177] at e_1803_1
  have hr := e_1803_1
  simp only [e_1803_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 15) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2196)) := by
  have c110178 := (publicBatchWrapper89_copies3443 a h).2.2.1
  have c110179 := (publicBatchWrapper89_copies3443 a h).2.2.2.1
  have c110180 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.1
  have c110181 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.1
  have c110182 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.1
  have c110183 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_2 := arithEq_of_rows h (row := 1803) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_2
  simp only [← c110178, ← c110179, ← c110180] at e_1803_2
  have e_1803_3 := arithEq_of_rows h (row := 1803) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_3
  simp only [← c110181, ← c110182, k2, ← c110183] at e_1803_3
  have hr := e_1803_3
  simp only [e_1803_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 23) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2197)) := by
  have c110184 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.1
  have c110185 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.1
  have c110186 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110187 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110188 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110189 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_4 := arithEq_of_rows h (row := 1803) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_4
  simp only [← c110184, ← c110185, ← c110186] at e_1803_4
  have e_1803_5 := arithEq_of_rows h (row := 1803) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_5
  simp only [← c110187, ← c110188, k2, ← c110189] at e_1803_5
  have hr := e_1803_5
  simp only [e_1803_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 31) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2198)) := by
  have c110190 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110191 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110192 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110193 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110194 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110195 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_6 := arithEq_of_rows h (row := 1803) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_6
  simp only [← c110190, ← c110191, ← c110192] at e_1803_6
  have e_1803_7 := arithEq_of_rows h (row := 1803) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_7
  simp only [← c110193, ← c110194, k2, ← c110195] at e_1803_7
  have hr := e_1803_7
  simp only [e_1803_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 39) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2199)) := by
  have c110196 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110197 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110198 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110199 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110200 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110201 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_8 := arithEq_of_rows h (row := 1803) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_8
  simp only [← c110196, ← c110197, ← c110198] at e_1803_8
  have e_1803_9 := arithEq_of_rows h (row := 1803) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_9
  simp only [← c110199, ← c110200, k2, ← c110201] at e_1803_9
  have hr := e_1803_9
  simp only [e_1803_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 47) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2200)) := by
  have c110202 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110203 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110204 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110205 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110206 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110207 := (publicBatchWrapper89_copies3443 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_10 := arithEq_of_rows h (row := 1803) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_10
  simp only [← c110202, ← c110203, ← c110204] at e_1803_10
  have e_1803_11 := arithEq_of_rows h (row := 1803) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_11
  simp only [← c110205, ← c110206, k2, ← c110207] at e_1803_11
  have hr := e_1803_11
  simp only [e_1803_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 55) = bselect (a (.wire 12 19)) (a (.virt 200)) (a (.virt 2201)) := by
  have c110208 := (publicBatchWrapper89_copies3444 a h).1
  have c110209 := (publicBatchWrapper89_copies3444 a h).2.1
  have c110210 := (publicBatchWrapper89_copies3444 a h).2.2.1
  have c110211 := (publicBatchWrapper89_copies3444 a h).2.2.2.1
  have c110212 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.1
  have c110213 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_12 := arithEq_of_rows h (row := 1803) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_12
  simp only [← c110208, ← c110209, ← c110210] at e_1803_12
  have e_1803_13 := arithEq_of_rows h (row := 1803) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_13
  simp only [← c110211, ← c110212, k2, ← c110213] at e_1803_13
  have hr := e_1803_13
  simp only [e_1803_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 63) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2370)) := by
  have c110214 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.1
  have c110215 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.1
  have c110216 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.1
  have c110217 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.1
  have c110218 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110219 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_14 := arithEq_of_rows h (row := 1803) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_14
  simp only [← c110214, ← c110215, ← c110216] at e_1803_14
  have e_1803_15 := arithEq_of_rows h (row := 1803) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_15
  simp only [← c110217, ← c110218, k2, ← c110219] at e_1803_15
  have hr := e_1803_15
  simp only [e_1803_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 71) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2371)) := by
  have c110220 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110221 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110222 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110223 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110224 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110225 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_16 := arithEq_of_rows h (row := 1803) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_16
  simp only [← c110220, ← c110221, ← c110222] at e_1803_16
  have e_1803_17 := arithEq_of_rows h (row := 1803) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_17
  simp only [← c110223, ← c110224, k2, ← c110225] at e_1803_17
  have hr := e_1803_17
  simp only [e_1803_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1803 79) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2372)) := by
  have c110226 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110227 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110228 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110229 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110230 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110231 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1803_18 := arithEq_of_rows h (row := 1803) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_18
  simp only [← c110226, ← c110227, ← c110228] at e_1803_18
  have e_1803_19 := arithEq_of_rows h (row := 1803) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1803_19
  simp only [← c110229, ← c110230, k2, ← c110231] at e_1803_19
  have hr := e_1803_19
  simp only [e_1803_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 7) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2373)) := by
  have c110232 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110233 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110234 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110235 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110236 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110237 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_0 := arithEq_of_rows h (row := 1804) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_0
  simp only [← c110232, ← c110233, ← c110234] at e_1804_0
  have e_1804_1 := arithEq_of_rows h (row := 1804) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_1
  simp only [← c110235, ← c110236, k2, ← c110237] at e_1804_1
  have hr := e_1804_1
  simp only [e_1804_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 15) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2374)) := by
  have c110238 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110239 := (publicBatchWrapper89_copies3444 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110240 := (publicBatchWrapper89_copies3445 a h).1
  have c110241 := (publicBatchWrapper89_copies3445 a h).2.1
  have c110242 := (publicBatchWrapper89_copies3445 a h).2.2.1
  have c110243 := (publicBatchWrapper89_copies3445 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_2 := arithEq_of_rows h (row := 1804) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_2
  simp only [← c110238, ← c110239, ← c110240] at e_1804_2
  have e_1804_3 := arithEq_of_rows h (row := 1804) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_3
  simp only [← c110241, ← c110242, k2, ← c110243] at e_1804_3
  have hr := e_1804_3
  simp only [e_1804_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 23) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2375)) := by
  have c110244 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.1
  have c110245 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.1
  have c110246 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.1
  have c110247 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.1
  have c110248 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.1
  have c110249 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_4 := arithEq_of_rows h (row := 1804) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_4
  simp only [← c110244, ← c110245, ← c110246] at e_1804_4
  have e_1804_5 := arithEq_of_rows h (row := 1804) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_5
  simp only [← c110247, ← c110248, k2, ← c110249] at e_1804_5
  have hr := e_1804_5
  simp only [e_1804_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 31) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2376)) := by
  have c110250 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110251 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110252 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110253 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110254 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110255 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_6 := arithEq_of_rows h (row := 1804) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_6
  simp only [← c110250, ← c110251, ← c110252] at e_1804_6
  have e_1804_7 := arithEq_of_rows h (row := 1804) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_7
  simp only [← c110253, ← c110254, k2, ← c110255] at e_1804_7
  have hr := e_1804_7
  simp only [e_1804_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 39) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2377)) := by
  have c110256 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110257 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110258 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110259 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110260 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110261 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_8 := arithEq_of_rows h (row := 1804) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_8
  simp only [← c110256, ← c110257, ← c110258] at e_1804_8
  have e_1804_9 := arithEq_of_rows h (row := 1804) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_9
  simp only [← c110259, ← c110260, k2, ← c110261] at e_1804_9
  have hr := e_1804_9
  simp only [e_1804_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 47) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2378)) := by
  have c110262 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110263 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110264 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110265 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110266 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110267 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_10 := arithEq_of_rows h (row := 1804) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_10
  simp only [← c110262, ← c110263, ← c110264] at e_1804_10
  have e_1804_11 := arithEq_of_rows h (row := 1804) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_11
  simp only [← c110265, ← c110266, k2, ← c110267] at e_1804_11
  have hr := e_1804_11
  simp only [e_1804_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 55) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2379)) := by
  have c110268 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110269 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110270 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110271 := (publicBatchWrapper89_copies3445 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110272 := (publicBatchWrapper89_copies3446 a h).1
  have c110273 := (publicBatchWrapper89_copies3446 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_12 := arithEq_of_rows h (row := 1804) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_12
  simp only [← c110268, ← c110269, ← c110270] at e_1804_12
  have e_1804_13 := arithEq_of_rows h (row := 1804) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_13
  simp only [← c110271, ← c110272, k2, ← c110273] at e_1804_13
  have hr := e_1804_13
  simp only [e_1804_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 63) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2380)) := by
  have c110274 := (publicBatchWrapper89_copies3446 a h).2.2.1
  have c110275 := (publicBatchWrapper89_copies3446 a h).2.2.2.1
  have c110276 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.1
  have c110277 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.1
  have c110278 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.1
  have c110279 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_14 := arithEq_of_rows h (row := 1804) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_14
  simp only [← c110274, ← c110275, ← c110276] at e_1804_14
  have e_1804_15 := arithEq_of_rows h (row := 1804) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_15
  simp only [← c110277, ← c110278, k2, ← c110279] at e_1804_15
  have hr := e_1804_15
  simp only [e_1804_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 71) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2381)) := by
  have c110280 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.1
  have c110281 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.1
  have c110282 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110283 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110284 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110285 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_16 := arithEq_of_rows h (row := 1804) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_16
  simp only [← c110280, ← c110281, ← c110282] at e_1804_16
  have e_1804_17 := arithEq_of_rows h (row := 1804) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_17
  simp only [← c110283, ← c110284, k2, ← c110285] at e_1804_17
  have hr := e_1804_17
  simp only [e_1804_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1804 79) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2382)) := by
  have c110286 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110287 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110288 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110289 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110290 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110291 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1804_18 := arithEq_of_rows h (row := 1804) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_18
  simp only [← c110286, ← c110287, ← c110288] at e_1804_18
  have e_1804_19 := arithEq_of_rows h (row := 1804) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1804_19
  simp only [← c110289, ← c110290, k2, ← c110291] at e_1804_19
  have hr := e_1804_19
  simp only [e_1804_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 7) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2383)) := by
  have c110292 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110293 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110294 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110295 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110296 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110297 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_0 := arithEq_of_rows h (row := 1805) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_0
  simp only [← c110292, ← c110293, ← c110294] at e_1805_0
  have e_1805_1 := arithEq_of_rows h (row := 1805) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_1
  simp only [← c110295, ← c110296, k2, ← c110297] at e_1805_1
  have hr := e_1805_1
  simp only [e_1805_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 15) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2384)) := by
  have c110298 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110299 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110300 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110301 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110302 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110303 := (publicBatchWrapper89_copies3446 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_2 := arithEq_of_rows h (row := 1805) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_2
  simp only [← c110298, ← c110299, ← c110300] at e_1805_2
  have e_1805_3 := arithEq_of_rows h (row := 1805) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_3
  simp only [← c110301, ← c110302, k2, ← c110303] at e_1805_3
  have hr := e_1805_3
  simp only [e_1805_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 23) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2385)) := by
  have c110304 := (publicBatchWrapper89_copies3447 a h).1
  have c110305 := (publicBatchWrapper89_copies3447 a h).2.1
  have c110306 := (publicBatchWrapper89_copies3447 a h).2.2.1
  have c110307 := (publicBatchWrapper89_copies3447 a h).2.2.2.1
  have c110308 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.1
  have c110309 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_4 := arithEq_of_rows h (row := 1805) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_4
  simp only [← c110304, ← c110305, ← c110306] at e_1805_4
  have e_1805_5 := arithEq_of_rows h (row := 1805) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_5
  simp only [← c110307, ← c110308, k2, ← c110309] at e_1805_5
  have hr := e_1805_5
  simp only [e_1805_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 31) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2386)) := by
  have c110310 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.1
  have c110311 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.1
  have c110312 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.1
  have c110313 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.1
  have c110314 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110315 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_6 := arithEq_of_rows h (row := 1805) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_6
  simp only [← c110310, ← c110311, ← c110312] at e_1805_6
  have e_1805_7 := arithEq_of_rows h (row := 1805) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_7
  simp only [← c110313, ← c110314, k2, ← c110315] at e_1805_7
  have hr := e_1805_7
  simp only [e_1805_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 39) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2387)) := by
  have c110316 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110317 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110318 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110319 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110320 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110321 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_8 := arithEq_of_rows h (row := 1805) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_8
  simp only [← c110316, ← c110317, ← c110318] at e_1805_8
  have e_1805_9 := arithEq_of_rows h (row := 1805) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_9
  simp only [← c110319, ← c110320, k2, ← c110321] at e_1805_9
  have hr := e_1805_9
  simp only [e_1805_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 47) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2388)) := by
  have c110322 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110323 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110324 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110325 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110326 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110327 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_10 := arithEq_of_rows h (row := 1805) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_10
  simp only [← c110322, ← c110323, ← c110324] at e_1805_10
  have e_1805_11 := arithEq_of_rows h (row := 1805) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_11
  simp only [← c110325, ← c110326, k2, ← c110327] at e_1805_11
  have hr := e_1805_11
  simp only [e_1805_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 55) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2389)) := by
  have c110328 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110329 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110330 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110331 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110332 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110333 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_12 := arithEq_of_rows h (row := 1805) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_12
  simp only [← c110328, ← c110329, ← c110330] at e_1805_12
  have e_1805_13 := arithEq_of_rows h (row := 1805) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_13
  simp only [← c110331, ← c110332, k2, ← c110333] at e_1805_13
  have hr := e_1805_13
  simp only [e_1805_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 63) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2390)) := by
  have c110334 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110335 := (publicBatchWrapper89_copies3447 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110336 := (publicBatchWrapper89_copies3448 a h).1
  have c110337 := (publicBatchWrapper89_copies3448 a h).2.1
  have c110338 := (publicBatchWrapper89_copies3448 a h).2.2.1
  have c110339 := (publicBatchWrapper89_copies3448 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_14 := arithEq_of_rows h (row := 1805) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_14
  simp only [← c110334, ← c110335, ← c110336] at e_1805_14
  have e_1805_15 := arithEq_of_rows h (row := 1805) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_15
  simp only [← c110337, ← c110338, k2, ← c110339] at e_1805_15
  have hr := e_1805_15
  simp only [e_1805_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 71) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2391)) := by
  have c110340 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.1
  have c110341 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.1
  have c110342 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.1
  have c110343 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.1
  have c110344 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.1
  have c110345 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_16 := arithEq_of_rows h (row := 1805) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_16
  simp only [← c110340, ← c110341, ← c110342] at e_1805_16
  have e_1805_17 := arithEq_of_rows h (row := 1805) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_17
  simp only [← c110343, ← c110344, k2, ← c110345] at e_1805_17
  have hr := e_1805_17
  simp only [e_1805_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1805 79) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2392)) := by
  have c110346 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110347 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110348 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110349 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110350 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110351 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1805_18 := arithEq_of_rows h (row := 1805) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_18
  simp only [← c110346, ← c110347, ← c110348] at e_1805_18
  have e_1805_19 := arithEq_of_rows h (row := 1805) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1805_19
  simp only [← c110349, ← c110350, k2, ← c110351] at e_1805_19
  have hr := e_1805_19
  simp only [e_1805_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 7) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2393)) := by
  have c110352 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110353 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110354 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110355 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110356 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110357 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_0 := arithEq_of_rows h (row := 1806) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_0
  simp only [← c110352, ← c110353, ← c110354] at e_1806_0
  have e_1806_1 := arithEq_of_rows h (row := 1806) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_1
  simp only [← c110355, ← c110356, k2, ← c110357] at e_1806_1
  have hr := e_1806_1
  simp only [e_1806_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 15) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2394)) := by
  have c110358 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110359 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110360 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110361 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110362 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110363 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_2 := arithEq_of_rows h (row := 1806) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_2
  simp only [← c110358, ← c110359, ← c110360] at e_1806_2
  have e_1806_3 := arithEq_of_rows h (row := 1806) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_3
  simp only [← c110361, ← c110362, k2, ← c110363] at e_1806_3
  have hr := e_1806_3
  simp only [e_1806_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 23) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2395)) := by
  have c110364 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110365 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110366 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110367 := (publicBatchWrapper89_copies3448 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110368 := (publicBatchWrapper89_copies3449 a h).1
  have c110369 := (publicBatchWrapper89_copies3449 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_4 := arithEq_of_rows h (row := 1806) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_4
  simp only [← c110364, ← c110365, ← c110366] at e_1806_4
  have e_1806_5 := arithEq_of_rows h (row := 1806) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_5
  simp only [← c110367, ← c110368, k2, ← c110369] at e_1806_5
  have hr := e_1806_5
  simp only [e_1806_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 31) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2396)) := by
  have c110370 := (publicBatchWrapper89_copies3449 a h).2.2.1
  have c110371 := (publicBatchWrapper89_copies3449 a h).2.2.2.1
  have c110372 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.1
  have c110373 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.1
  have c110374 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.1
  have c110375 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_6 := arithEq_of_rows h (row := 1806) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_6
  simp only [← c110370, ← c110371, ← c110372] at e_1806_6
  have e_1806_7 := arithEq_of_rows h (row := 1806) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_7
  simp only [← c110373, ← c110374, k2, ← c110375] at e_1806_7
  have hr := e_1806_7
  simp only [e_1806_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 39) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2397)) := by
  have c110376 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.1
  have c110377 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.1
  have c110378 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110379 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110380 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110381 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_8 := arithEq_of_rows h (row := 1806) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_8
  simp only [← c110376, ← c110377, ← c110378] at e_1806_8
  have e_1806_9 := arithEq_of_rows h (row := 1806) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_9
  simp only [← c110379, ← c110380, k2, ← c110381] at e_1806_9
  have hr := e_1806_9
  simp only [e_1806_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 47) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2398)) := by
  have c110382 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110383 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110384 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110385 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110386 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110387 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_10 := arithEq_of_rows h (row := 1806) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_10
  simp only [← c110382, ← c110383, ← c110384] at e_1806_10
  have e_1806_11 := arithEq_of_rows h (row := 1806) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_11
  simp only [← c110385, ← c110386, k2, ← c110387] at e_1806_11
  have hr := e_1806_11
  simp only [e_1806_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 55) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2399)) := by
  have c110388 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110389 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110390 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110391 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110392 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110393 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_12 := arithEq_of_rows h (row := 1806) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_12
  simp only [← c110388, ← c110389, ← c110390] at e_1806_12
  have e_1806_13 := arithEq_of_rows h (row := 1806) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_13
  simp only [← c110391, ← c110392, k2, ← c110393] at e_1806_13
  have hr := e_1806_13
  simp only [e_1806_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 63) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2400)) := by
  have c110394 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110395 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110396 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110397 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110398 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110399 := (publicBatchWrapper89_copies3449 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_14 := arithEq_of_rows h (row := 1806) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_14
  simp only [← c110394, ← c110395, ← c110396] at e_1806_14
  have e_1806_15 := arithEq_of_rows h (row := 1806) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_15
  simp only [← c110397, ← c110398, k2, ← c110399] at e_1806_15
  have hr := e_1806_15
  simp only [e_1806_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 71) = bselect (a (.wire 12 63)) (a (.virt 200)) (a (.virt 2401)) := by
  have c110400 := (publicBatchWrapper89_copies3450 a h).1
  have c110401 := (publicBatchWrapper89_copies3450 a h).2.1
  have c110402 := (publicBatchWrapper89_copies3450 a h).2.2.1
  have c110403 := (publicBatchWrapper89_copies3450 a h).2.2.2.1
  have c110404 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.1
  have c110405 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_16 := arithEq_of_rows h (row := 1806) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_16
  simp only [← c110400, ← c110401, ← c110402] at e_1806_16
  have e_1806_17 := arithEq_of_rows h (row := 1806) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_17
  simp only [← c110403, ← c110404, k2, ← c110405] at e_1806_17
  have hr := e_1806_17
  simp only [e_1806_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1806 79) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2570)) := by
  have c110406 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.1
  have c110407 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.1
  have c110408 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.1
  have c110409 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.1
  have c110410 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110411 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1806_18 := arithEq_of_rows h (row := 1806) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_18
  simp only [← c110406, ← c110407, ← c110408] at e_1806_18
  have e_1806_19 := arithEq_of_rows h (row := 1806) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1806_19
  simp only [← c110409, ← c110410, k2, ← c110411] at e_1806_19
  have hr := e_1806_19
  simp only [e_1806_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 7) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2571)) := by
  have c110412 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110413 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110414 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110415 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110416 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110417 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_0 := arithEq_of_rows h (row := 1807) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_0
  simp only [← c110412, ← c110413, ← c110414] at e_1807_0
  have e_1807_1 := arithEq_of_rows h (row := 1807) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_1
  simp only [← c110415, ← c110416, k2, ← c110417] at e_1807_1
  have hr := e_1807_1
  simp only [e_1807_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 15) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2572)) := by
  have c110418 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110419 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110420 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110421 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110422 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110423 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_2 := arithEq_of_rows h (row := 1807) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_2
  simp only [← c110418, ← c110419, ← c110420] at e_1807_2
  have e_1807_3 := arithEq_of_rows h (row := 1807) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_3
  simp only [← c110421, ← c110422, k2, ← c110423] at e_1807_3
  have hr := e_1807_3
  simp only [e_1807_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 23) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2573)) := by
  have c110424 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110425 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110426 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110427 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110428 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110429 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_4 := arithEq_of_rows h (row := 1807) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_4
  simp only [← c110424, ← c110425, ← c110426] at e_1807_4
  have e_1807_5 := arithEq_of_rows h (row := 1807) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_5
  simp only [← c110427, ← c110428, k2, ← c110429] at e_1807_5
  have hr := e_1807_5
  simp only [e_1807_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 31) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2574)) := by
  have c110430 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110431 := (publicBatchWrapper89_copies3450 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110432 := (publicBatchWrapper89_copies3451 a h).1
  have c110433 := (publicBatchWrapper89_copies3451 a h).2.1
  have c110434 := (publicBatchWrapper89_copies3451 a h).2.2.1
  have c110435 := (publicBatchWrapper89_copies3451 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_6 := arithEq_of_rows h (row := 1807) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_6
  simp only [← c110430, ← c110431, ← c110432] at e_1807_6
  have e_1807_7 := arithEq_of_rows h (row := 1807) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_7
  simp only [← c110433, ← c110434, k2, ← c110435] at e_1807_7
  have hr := e_1807_7
  simp only [e_1807_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 39) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2575)) := by
  have c110436 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.1
  have c110437 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.1
  have c110438 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.1
  have c110439 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.1
  have c110440 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.1
  have c110441 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_8 := arithEq_of_rows h (row := 1807) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_8
  simp only [← c110436, ← c110437, ← c110438] at e_1807_8
  have e_1807_9 := arithEq_of_rows h (row := 1807) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_9
  simp only [← c110439, ← c110440, k2, ← c110441] at e_1807_9
  have hr := e_1807_9
  simp only [e_1807_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 47) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2576)) := by
  have c110442 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110443 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110444 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110445 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110446 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110447 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_10 := arithEq_of_rows h (row := 1807) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_10
  simp only [← c110442, ← c110443, ← c110444] at e_1807_10
  have e_1807_11 := arithEq_of_rows h (row := 1807) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_11
  simp only [← c110445, ← c110446, k2, ← c110447] at e_1807_11
  have hr := e_1807_11
  simp only [e_1807_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 55) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2577)) := by
  have c110448 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110449 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110450 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110451 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110452 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110453 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_12 := arithEq_of_rows h (row := 1807) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_12
  simp only [← c110448, ← c110449, ← c110450] at e_1807_12
  have e_1807_13 := arithEq_of_rows h (row := 1807) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_13
  simp only [← c110451, ← c110452, k2, ← c110453] at e_1807_13
  have hr := e_1807_13
  simp only [e_1807_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 63) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2578)) := by
  have c110454 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110455 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110456 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110457 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110458 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110459 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_14 := arithEq_of_rows h (row := 1807) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_14
  simp only [← c110454, ← c110455, ← c110456] at e_1807_14
  have e_1807_15 := arithEq_of_rows h (row := 1807) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_15
  simp only [← c110457, ← c110458, k2, ← c110459] at e_1807_15
  have hr := e_1807_15
  simp only [e_1807_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 71) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2579)) := by
  have c110460 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110461 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110462 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110463 := (publicBatchWrapper89_copies3451 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110464 := (publicBatchWrapper89_copies3452 a h).1
  have c110465 := (publicBatchWrapper89_copies3452 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_16 := arithEq_of_rows h (row := 1807) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_16
  simp only [← c110460, ← c110461, ← c110462] at e_1807_16
  have e_1807_17 := arithEq_of_rows h (row := 1807) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_17
  simp only [← c110463, ← c110464, k2, ← c110465] at e_1807_17
  have hr := e_1807_17
  simp only [e_1807_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1807 79) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2580)) := by
  have c110466 := (publicBatchWrapper89_copies3452 a h).2.2.1
  have c110467 := (publicBatchWrapper89_copies3452 a h).2.2.2.1
  have c110468 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.1
  have c110469 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.1
  have c110470 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.1
  have c110471 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1807_18 := arithEq_of_rows h (row := 1807) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_18
  simp only [← c110466, ← c110467, ← c110468] at e_1807_18
  have e_1807_19 := arithEq_of_rows h (row := 1807) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1807_19
  simp only [← c110469, ← c110470, k2, ← c110471] at e_1807_19
  have hr := e_1807_19
  simp only [e_1807_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 7) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2581)) := by
  have c110472 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.1
  have c110473 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.1
  have c110474 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110475 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110476 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110477 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_0 := arithEq_of_rows h (row := 1808) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_0
  simp only [← c110472, ← c110473, ← c110474] at e_1808_0
  have e_1808_1 := arithEq_of_rows h (row := 1808) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_1
  simp only [← c110475, ← c110476, k2, ← c110477] at e_1808_1
  have hr := e_1808_1
  simp only [e_1808_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 15) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2582)) := by
  have c110478 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110479 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110480 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110481 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110482 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110483 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_2 := arithEq_of_rows h (row := 1808) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_2
  simp only [← c110478, ← c110479, ← c110480] at e_1808_2
  have e_1808_3 := arithEq_of_rows h (row := 1808) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_3
  simp only [← c110481, ← c110482, k2, ← c110483] at e_1808_3
  have hr := e_1808_3
  simp only [e_1808_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 23) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2583)) := by
  have c110484 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110485 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110486 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110487 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110488 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110489 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_4 := arithEq_of_rows h (row := 1808) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_4
  simp only [← c110484, ← c110485, ← c110486] at e_1808_4
  have e_1808_5 := arithEq_of_rows h (row := 1808) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_5
  simp only [← c110487, ← c110488, k2, ← c110489] at e_1808_5
  have hr := e_1808_5
  simp only [e_1808_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 31) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2584)) := by
  have c110490 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110491 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110492 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110493 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110494 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110495 := (publicBatchWrapper89_copies3452 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_6 := arithEq_of_rows h (row := 1808) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_6
  simp only [← c110490, ← c110491, ← c110492] at e_1808_6
  have e_1808_7 := arithEq_of_rows h (row := 1808) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_7
  simp only [← c110493, ← c110494, k2, ← c110495] at e_1808_7
  have hr := e_1808_7
  simp only [e_1808_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 39) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2585)) := by
  have c110496 := (publicBatchWrapper89_copies3453 a h).1
  have c110497 := (publicBatchWrapper89_copies3453 a h).2.1
  have c110498 := (publicBatchWrapper89_copies3453 a h).2.2.1
  have c110499 := (publicBatchWrapper89_copies3453 a h).2.2.2.1
  have c110500 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.1
  have c110501 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_8 := arithEq_of_rows h (row := 1808) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_8
  simp only [← c110496, ← c110497, ← c110498] at e_1808_8
  have e_1808_9 := arithEq_of_rows h (row := 1808) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_9
  simp only [← c110499, ← c110500, k2, ← c110501] at e_1808_9
  have hr := e_1808_9
  simp only [e_1808_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 47) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2586)) := by
  have c110502 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.1
  have c110503 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.1
  have c110504 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.1
  have c110505 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.1
  have c110506 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110507 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_10 := arithEq_of_rows h (row := 1808) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_10
  simp only [← c110502, ← c110503, ← c110504] at e_1808_10
  have e_1808_11 := arithEq_of_rows h (row := 1808) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_11
  simp only [← c110505, ← c110506, k2, ← c110507] at e_1808_11
  have hr := e_1808_11
  simp only [e_1808_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 55) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2587)) := by
  have c110508 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110509 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110510 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110511 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110512 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110513 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_12 := arithEq_of_rows h (row := 1808) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_12
  simp only [← c110508, ← c110509, ← c110510] at e_1808_12
  have e_1808_13 := arithEq_of_rows h (row := 1808) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_13
  simp only [← c110511, ← c110512, k2, ← c110513] at e_1808_13
  have hr := e_1808_13
  simp only [e_1808_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 63) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2588)) := by
  have c110514 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110515 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110516 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110517 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110518 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110519 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_14 := arithEq_of_rows h (row := 1808) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_14
  simp only [← c110514, ← c110515, ← c110516] at e_1808_14
  have e_1808_15 := arithEq_of_rows h (row := 1808) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_15
  simp only [← c110517, ← c110518, k2, ← c110519] at e_1808_15
  have hr := e_1808_15
  simp only [e_1808_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 71) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2589)) := by
  have c110520 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110521 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110522 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110523 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110524 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110525 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_16 := arithEq_of_rows h (row := 1808) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_16
  simp only [← c110520, ← c110521, ← c110522] at e_1808_16
  have e_1808_17 := arithEq_of_rows h (row := 1808) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_17
  simp only [← c110523, ← c110524, k2, ← c110525] at e_1808_17
  have hr := e_1808_17
  simp only [e_1808_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1808 79) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2590)) := by
  have c110526 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110527 := (publicBatchWrapper89_copies3453 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110528 := (publicBatchWrapper89_copies3454 a h).1
  have c110529 := (publicBatchWrapper89_copies3454 a h).2.1
  have c110530 := (publicBatchWrapper89_copies3454 a h).2.2.1
  have c110531 := (publicBatchWrapper89_copies3454 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1808_18 := arithEq_of_rows h (row := 1808) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_18
  simp only [← c110526, ← c110527, ← c110528] at e_1808_18
  have e_1808_19 := arithEq_of_rows h (row := 1808) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1808_19
  simp only [← c110529, ← c110530, k2, ← c110531] at e_1808_19
  have hr := e_1808_19
  simp only [e_1808_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 7) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2591)) := by
  have c110532 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.1
  have c110533 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.1
  have c110534 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.1
  have c110535 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.1
  have c110536 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.1
  have c110537 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_0 := arithEq_of_rows h (row := 1809) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_0
  simp only [← c110532, ← c110533, ← c110534] at e_1809_0
  have e_1809_1 := arithEq_of_rows h (row := 1809) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_1
  simp only [← c110535, ← c110536, k2, ← c110537] at e_1809_1
  have hr := e_1809_1
  simp only [e_1809_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 15) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2592)) := by
  have c110538 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110539 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110540 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110541 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110542 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110543 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_2 := arithEq_of_rows h (row := 1809) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_2
  simp only [← c110538, ← c110539, ← c110540] at e_1809_2
  have e_1809_3 := arithEq_of_rows h (row := 1809) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_3
  simp only [← c110541, ← c110542, k2, ← c110543] at e_1809_3
  have hr := e_1809_3
  simp only [e_1809_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 23) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2593)) := by
  have c110544 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110545 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110546 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110547 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110548 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110549 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_4 := arithEq_of_rows h (row := 1809) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_4
  simp only [← c110544, ← c110545, ← c110546] at e_1809_4
  have e_1809_5 := arithEq_of_rows h (row := 1809) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_5
  simp only [← c110547, ← c110548, k2, ← c110549] at e_1809_5
  have hr := e_1809_5
  simp only [e_1809_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 31) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2594)) := by
  have c110550 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110551 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110552 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110553 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110554 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110555 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_6 := arithEq_of_rows h (row := 1809) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_6
  simp only [← c110550, ← c110551, ← c110552] at e_1809_6
  have e_1809_7 := arithEq_of_rows h (row := 1809) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_7
  simp only [← c110553, ← c110554, k2, ← c110555] at e_1809_7
  have hr := e_1809_7
  simp only [e_1809_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 39) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2595)) := by
  have c110556 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110557 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110558 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110559 := (publicBatchWrapper89_copies3454 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110560 := (publicBatchWrapper89_copies3455 a h).1
  have c110561 := (publicBatchWrapper89_copies3455 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_8 := arithEq_of_rows h (row := 1809) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_8
  simp only [← c110556, ← c110557, ← c110558] at e_1809_8
  have e_1809_9 := arithEq_of_rows h (row := 1809) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_9
  simp only [← c110559, ← c110560, k2, ← c110561] at e_1809_9
  have hr := e_1809_9
  simp only [e_1809_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 47) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2596)) := by
  have c110562 := (publicBatchWrapper89_copies3455 a h).2.2.1
  have c110563 := (publicBatchWrapper89_copies3455 a h).2.2.2.1
  have c110564 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.1
  have c110565 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.1
  have c110566 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.1
  have c110567 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_10 := arithEq_of_rows h (row := 1809) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_10
  simp only [← c110562, ← c110563, ← c110564] at e_1809_10
  have e_1809_11 := arithEq_of_rows h (row := 1809) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_11
  simp only [← c110565, ← c110566, k2, ← c110567] at e_1809_11
  have hr := e_1809_11
  simp only [e_1809_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 55) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2597)) := by
  have c110568 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.1
  have c110569 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.1
  have c110570 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110571 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110572 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110573 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_12 := arithEq_of_rows h (row := 1809) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_12
  simp only [← c110568, ← c110569, ← c110570] at e_1809_12
  have e_1809_13 := arithEq_of_rows h (row := 1809) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_13
  simp only [← c110571, ← c110572, k2, ← c110573] at e_1809_13
  have hr := e_1809_13
  simp only [e_1809_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 63) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2598)) := by
  have c110574 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110575 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110576 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110577 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110578 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110579 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_14 := arithEq_of_rows h (row := 1809) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_14
  simp only [← c110574, ← c110575, ← c110576] at e_1809_14
  have e_1809_15 := arithEq_of_rows h (row := 1809) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_15
  simp only [← c110577, ← c110578, k2, ← c110579] at e_1809_15
  have hr := e_1809_15
  simp only [e_1809_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 71) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2599)) := by
  have c110580 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110581 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110582 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110583 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110584 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110585 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_16 := arithEq_of_rows h (row := 1809) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_16
  simp only [← c110580, ← c110581, ← c110582] at e_1809_16
  have e_1809_17 := arithEq_of_rows h (row := 1809) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_17
  simp only [← c110583, ← c110584, k2, ← c110585] at e_1809_17
  have hr := e_1809_17
  simp only [e_1809_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1809 79) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2600)) := by
  have c110586 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110587 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110588 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110589 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110590 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110591 := (publicBatchWrapper89_copies3455 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1809_18 := arithEq_of_rows h (row := 1809) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_18
  simp only [← c110586, ← c110587, ← c110588] at e_1809_18
  have e_1809_19 := arithEq_of_rows h (row := 1809) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1809_19
  simp only [← c110589, ← c110590, k2, ← c110591] at e_1809_19
  have hr := e_1809_19
  simp only [e_1809_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 7) = bselect (a (.wire 14 27)) (a (.virt 200)) (a (.virt 2601)) := by
  have c110592 := (publicBatchWrapper89_copies3456 a h).1
  have c110593 := (publicBatchWrapper89_copies3456 a h).2.1
  have c110594 := (publicBatchWrapper89_copies3456 a h).2.2.1
  have c110595 := (publicBatchWrapper89_copies3456 a h).2.2.2.1
  have c110596 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.1
  have c110597 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_0 := arithEq_of_rows h (row := 1810) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_0
  simp only [← c110592, ← c110593, ← c110594] at e_1810_0
  have e_1810_1 := arithEq_of_rows h (row := 1810) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_1
  simp only [← c110595, ← c110596, k2, ← c110597] at e_1810_1
  have hr := e_1810_1
  simp only [e_1810_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 15) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2770)) := by
  have c110598 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.1
  have c110599 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.1
  have c110600 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.1
  have c110601 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.1
  have c110602 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110603 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_2 := arithEq_of_rows h (row := 1810) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_2
  simp only [← c110598, ← c110599, ← c110600] at e_1810_2
  have e_1810_3 := arithEq_of_rows h (row := 1810) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_3
  simp only [← c110601, ← c110602, k2, ← c110603] at e_1810_3
  have hr := e_1810_3
  simp only [e_1810_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 23) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2771)) := by
  have c110604 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110605 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110606 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110607 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110608 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110609 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_4 := arithEq_of_rows h (row := 1810) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_4
  simp only [← c110604, ← c110605, ← c110606] at e_1810_4
  have e_1810_5 := arithEq_of_rows h (row := 1810) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_5
  simp only [← c110607, ← c110608, k2, ← c110609] at e_1810_5
  have hr := e_1810_5
  simp only [e_1810_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 31) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2772)) := by
  have c110610 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110611 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110612 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110613 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110614 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110615 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_6 := arithEq_of_rows h (row := 1810) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_6
  simp only [← c110610, ← c110611, ← c110612] at e_1810_6
  have e_1810_7 := arithEq_of_rows h (row := 1810) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_7
  simp only [← c110613, ← c110614, k2, ← c110615] at e_1810_7
  have hr := e_1810_7
  simp only [e_1810_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 39) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2773)) := by
  have c110616 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110617 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110618 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110619 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110620 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110621 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_8 := arithEq_of_rows h (row := 1810) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_8
  simp only [← c110616, ← c110617, ← c110618] at e_1810_8
  have e_1810_9 := arithEq_of_rows h (row := 1810) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_9
  simp only [← c110619, ← c110620, k2, ← c110621] at e_1810_9
  have hr := e_1810_9
  simp only [e_1810_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 47) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2774)) := by
  have c110622 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110623 := (publicBatchWrapper89_copies3456 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110624 := (publicBatchWrapper89_copies3457 a h).1
  have c110625 := (publicBatchWrapper89_copies3457 a h).2.1
  have c110626 := (publicBatchWrapper89_copies3457 a h).2.2.1
  have c110627 := (publicBatchWrapper89_copies3457 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_10 := arithEq_of_rows h (row := 1810) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_10
  simp only [← c110622, ← c110623, ← c110624] at e_1810_10
  have e_1810_11 := arithEq_of_rows h (row := 1810) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_11
  simp only [← c110625, ← c110626, k2, ← c110627] at e_1810_11
  have hr := e_1810_11
  simp only [e_1810_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 55) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2775)) := by
  have c110628 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.1
  have c110629 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.1
  have c110630 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.1
  have c110631 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.1
  have c110632 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.1
  have c110633 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_12 := arithEq_of_rows h (row := 1810) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_12
  simp only [← c110628, ← c110629, ← c110630] at e_1810_12
  have e_1810_13 := arithEq_of_rows h (row := 1810) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_13
  simp only [← c110631, ← c110632, k2, ← c110633] at e_1810_13
  have hr := e_1810_13
  simp only [e_1810_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 63) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2776)) := by
  have c110634 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110635 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110636 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110637 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110638 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110639 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_14 := arithEq_of_rows h (row := 1810) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_14
  simp only [← c110634, ← c110635, ← c110636] at e_1810_14
  have e_1810_15 := arithEq_of_rows h (row := 1810) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_15
  simp only [← c110637, ← c110638, k2, ← c110639] at e_1810_15
  have hr := e_1810_15
  simp only [e_1810_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 71) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2777)) := by
  have c110640 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110641 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110642 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110643 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110644 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110645 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_16 := arithEq_of_rows h (row := 1810) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_16
  simp only [← c110640, ← c110641, ← c110642] at e_1810_16
  have e_1810_17 := arithEq_of_rows h (row := 1810) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_17
  simp only [← c110643, ← c110644, k2, ← c110645] at e_1810_17
  have hr := e_1810_17
  simp only [e_1810_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1810 79) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2778)) := by
  have c110646 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110647 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110648 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110649 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110650 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110651 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1810_18 := arithEq_of_rows h (row := 1810) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_18
  simp only [← c110646, ← c110647, ← c110648] at e_1810_18
  have e_1810_19 := arithEq_of_rows h (row := 1810) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1810_19
  simp only [← c110649, ← c110650, k2, ← c110651] at e_1810_19
  have hr := e_1810_19
  simp only [e_1810_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 7) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2779)) := by
  have c110652 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110653 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110654 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110655 := (publicBatchWrapper89_copies3457 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110656 := (publicBatchWrapper89_copies3458 a h).1
  have c110657 := (publicBatchWrapper89_copies3458 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_0 := arithEq_of_rows h (row := 1811) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_0
  simp only [← c110652, ← c110653, ← c110654] at e_1811_0
  have e_1811_1 := arithEq_of_rows h (row := 1811) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_1
  simp only [← c110655, ← c110656, k2, ← c110657] at e_1811_1
  have hr := e_1811_1
  simp only [e_1811_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 15) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2780)) := by
  have c110658 := (publicBatchWrapper89_copies3458 a h).2.2.1
  have c110659 := (publicBatchWrapper89_copies3458 a h).2.2.2.1
  have c110660 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.1
  have c110661 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.1
  have c110662 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.1
  have c110663 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_2 := arithEq_of_rows h (row := 1811) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_2
  simp only [← c110658, ← c110659, ← c110660] at e_1811_2
  have e_1811_3 := arithEq_of_rows h (row := 1811) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_3
  simp only [← c110661, ← c110662, k2, ← c110663] at e_1811_3
  have hr := e_1811_3
  simp only [e_1811_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 23) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2781)) := by
  have c110664 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.1
  have c110665 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.1
  have c110666 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110667 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110668 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110669 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_4 := arithEq_of_rows h (row := 1811) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_4
  simp only [← c110664, ← c110665, ← c110666] at e_1811_4
  have e_1811_5 := arithEq_of_rows h (row := 1811) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_5
  simp only [← c110667, ← c110668, k2, ← c110669] at e_1811_5
  have hr := e_1811_5
  simp only [e_1811_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 31) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2782)) := by
  have c110670 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110671 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110672 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110673 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110674 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110675 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_6 := arithEq_of_rows h (row := 1811) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_6
  simp only [← c110670, ← c110671, ← c110672] at e_1811_6
  have e_1811_7 := arithEq_of_rows h (row := 1811) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_7
  simp only [← c110673, ← c110674, k2, ← c110675] at e_1811_7
  have hr := e_1811_7
  simp only [e_1811_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 39) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2783)) := by
  have c110676 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110677 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110678 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110679 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110680 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110681 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_8 := arithEq_of_rows h (row := 1811) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_8
  simp only [← c110676, ← c110677, ← c110678] at e_1811_8
  have e_1811_9 := arithEq_of_rows h (row := 1811) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_9
  simp only [← c110679, ← c110680, k2, ← c110681] at e_1811_9
  have hr := e_1811_9
  simp only [e_1811_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 47) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2784)) := by
  have c110682 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110683 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110684 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110685 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110686 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110687 := (publicBatchWrapper89_copies3458 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_10 := arithEq_of_rows h (row := 1811) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_10
  simp only [← c110682, ← c110683, ← c110684] at e_1811_10
  have e_1811_11 := arithEq_of_rows h (row := 1811) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_11
  simp only [← c110685, ← c110686, k2, ← c110687] at e_1811_11
  have hr := e_1811_11
  simp only [e_1811_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 55) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2785)) := by
  have c110688 := (publicBatchWrapper89_copies3459 a h).1
  have c110689 := (publicBatchWrapper89_copies3459 a h).2.1
  have c110690 := (publicBatchWrapper89_copies3459 a h).2.2.1
  have c110691 := (publicBatchWrapper89_copies3459 a h).2.2.2.1
  have c110692 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.1
  have c110693 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_12 := arithEq_of_rows h (row := 1811) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_12
  simp only [← c110688, ← c110689, ← c110690] at e_1811_12
  have e_1811_13 := arithEq_of_rows h (row := 1811) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_13
  simp only [← c110691, ← c110692, k2, ← c110693] at e_1811_13
  have hr := e_1811_13
  simp only [e_1811_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 63) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2786)) := by
  have c110694 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.1
  have c110695 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.1
  have c110696 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.1
  have c110697 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.1
  have c110698 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110699 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_14 := arithEq_of_rows h (row := 1811) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_14
  simp only [← c110694, ← c110695, ← c110696] at e_1811_14
  have e_1811_15 := arithEq_of_rows h (row := 1811) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_15
  simp only [← c110697, ← c110698, k2, ← c110699] at e_1811_15
  have hr := e_1811_15
  simp only [e_1811_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 71) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2787)) := by
  have c110700 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110701 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110702 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110703 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110704 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110705 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_16 := arithEq_of_rows h (row := 1811) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_16
  simp only [← c110700, ← c110701, ← c110702] at e_1811_16
  have e_1811_17 := arithEq_of_rows h (row := 1811) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_17
  simp only [← c110703, ← c110704, k2, ← c110705] at e_1811_17
  have hr := e_1811_17
  simp only [e_1811_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1811 79) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2788)) := by
  have c110706 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110707 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110708 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110709 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110710 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110711 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1811_18 := arithEq_of_rows h (row := 1811) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_18
  simp only [← c110706, ← c110707, ← c110708] at e_1811_18
  have e_1811_19 := arithEq_of_rows h (row := 1811) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1811_19
  simp only [← c110709, ← c110710, k2, ← c110711] at e_1811_19
  have hr := e_1811_19
  simp only [e_1811_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 7) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2789)) := by
  have c110712 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110713 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110714 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110715 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110716 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110717 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_0 := arithEq_of_rows h (row := 1812) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_0
  simp only [← c110712, ← c110713, ← c110714] at e_1812_0
  have e_1812_1 := arithEq_of_rows h (row := 1812) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_1
  simp only [← c110715, ← c110716, k2, ← c110717] at e_1812_1
  have hr := e_1812_1
  simp only [e_1812_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 15) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2790)) := by
  have c110718 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110719 := (publicBatchWrapper89_copies3459 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110720 := (publicBatchWrapper89_copies3460 a h).1
  have c110721 := (publicBatchWrapper89_copies3460 a h).2.1
  have c110722 := (publicBatchWrapper89_copies3460 a h).2.2.1
  have c110723 := (publicBatchWrapper89_copies3460 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_2 := arithEq_of_rows h (row := 1812) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_2
  simp only [← c110718, ← c110719, ← c110720] at e_1812_2
  have e_1812_3 := arithEq_of_rows h (row := 1812) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_3
  simp only [← c110721, ← c110722, k2, ← c110723] at e_1812_3
  have hr := e_1812_3
  simp only [e_1812_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 23) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2791)) := by
  have c110724 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.1
  have c110725 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.1
  have c110726 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.1
  have c110727 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.1
  have c110728 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.1
  have c110729 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_4 := arithEq_of_rows h (row := 1812) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_4
  simp only [← c110724, ← c110725, ← c110726] at e_1812_4
  have e_1812_5 := arithEq_of_rows h (row := 1812) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_5
  simp only [← c110727, ← c110728, k2, ← c110729] at e_1812_5
  have hr := e_1812_5
  simp only [e_1812_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 31) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2792)) := by
  have c110730 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110731 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110732 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110733 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110734 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110735 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_6 := arithEq_of_rows h (row := 1812) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_6
  simp only [← c110730, ← c110731, ← c110732] at e_1812_6
  have e_1812_7 := arithEq_of_rows h (row := 1812) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_7
  simp only [← c110733, ← c110734, k2, ← c110735] at e_1812_7
  have hr := e_1812_7
  simp only [e_1812_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 39) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2793)) := by
  have c110736 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110737 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110738 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110739 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110740 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110741 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_8 := arithEq_of_rows h (row := 1812) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_8
  simp only [← c110736, ← c110737, ← c110738] at e_1812_8
  have e_1812_9 := arithEq_of_rows h (row := 1812) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_9
  simp only [← c110739, ← c110740, k2, ← c110741] at e_1812_9
  have hr := e_1812_9
  simp only [e_1812_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 47) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2794)) := by
  have c110742 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110743 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110744 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110745 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110746 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110747 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_10 := arithEq_of_rows h (row := 1812) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_10
  simp only [← c110742, ← c110743, ← c110744] at e_1812_10
  have e_1812_11 := arithEq_of_rows h (row := 1812) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_11
  simp only [← c110745, ← c110746, k2, ← c110747] at e_1812_11
  have hr := e_1812_11
  simp only [e_1812_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 55) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2795)) := by
  have c110748 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110749 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110750 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110751 := (publicBatchWrapper89_copies3460 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110752 := (publicBatchWrapper89_copies3461 a h).1
  have c110753 := (publicBatchWrapper89_copies3461 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_12 := arithEq_of_rows h (row := 1812) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_12
  simp only [← c110748, ← c110749, ← c110750] at e_1812_12
  have e_1812_13 := arithEq_of_rows h (row := 1812) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_13
  simp only [← c110751, ← c110752, k2, ← c110753] at e_1812_13
  have hr := e_1812_13
  simp only [e_1812_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 63) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2796)) := by
  have c110754 := (publicBatchWrapper89_copies3461 a h).2.2.1
  have c110755 := (publicBatchWrapper89_copies3461 a h).2.2.2.1
  have c110756 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.1
  have c110757 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.1
  have c110758 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.1
  have c110759 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_14 := arithEq_of_rows h (row := 1812) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_14
  simp only [← c110754, ← c110755, ← c110756] at e_1812_14
  have e_1812_15 := arithEq_of_rows h (row := 1812) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_15
  simp only [← c110757, ← c110758, k2, ← c110759] at e_1812_15
  have hr := e_1812_15
  simp only [e_1812_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 71) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2797)) := by
  have c110760 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.1
  have c110761 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.1
  have c110762 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110763 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110764 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110765 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_16 := arithEq_of_rows h (row := 1812) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_16
  simp only [← c110760, ← c110761, ← c110762] at e_1812_16
  have e_1812_17 := arithEq_of_rows h (row := 1812) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_17
  simp only [← c110763, ← c110764, k2, ← c110765] at e_1812_17
  have hr := e_1812_17
  simp only [e_1812_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1812 79) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2798)) := by
  have c110766 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110767 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110768 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110769 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110770 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110771 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1812_18 := arithEq_of_rows h (row := 1812) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_18
  simp only [← c110766, ← c110767, ← c110768] at e_1812_18
  have e_1812_19 := arithEq_of_rows h (row := 1812) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1812_19
  simp only [← c110769, ← c110770, k2, ← c110771] at e_1812_19
  have hr := e_1812_19
  simp only [e_1812_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 7) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2799)) := by
  have c110772 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110773 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110774 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110775 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110776 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110777 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_0 := arithEq_of_rows h (row := 1813) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_0
  simp only [← c110772, ← c110773, ← c110774] at e_1813_0
  have e_1813_1 := arithEq_of_rows h (row := 1813) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_1
  simp only [← c110775, ← c110776, k2, ← c110777] at e_1813_1
  have hr := e_1813_1
  simp only [e_1813_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 15) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2800)) := by
  have c110778 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110779 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110780 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110781 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110782 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110783 := (publicBatchWrapper89_copies3461 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_2 := arithEq_of_rows h (row := 1813) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_2
  simp only [← c110778, ← c110779, ← c110780] at e_1813_2
  have e_1813_3 := arithEq_of_rows h (row := 1813) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_3
  simp only [← c110781, ← c110782, k2, ← c110783] at e_1813_3
  have hr := e_1813_3
  simp only [e_1813_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 23) = bselect (a (.wire 14 71)) (a (.virt 200)) (a (.virt 2801)) := by
  have c110784 := (publicBatchWrapper89_copies3462 a h).1
  have c110785 := (publicBatchWrapper89_copies3462 a h).2.1
  have c110786 := (publicBatchWrapper89_copies3462 a h).2.2.1
  have c110787 := (publicBatchWrapper89_copies3462 a h).2.2.2.1
  have c110788 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.1
  have c110789 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_4 := arithEq_of_rows h (row := 1813) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_4
  simp only [← c110784, ← c110785, ← c110786] at e_1813_4
  have e_1813_5 := arithEq_of_rows h (row := 1813) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_5
  simp only [← c110787, ← c110788, k2, ← c110789] at e_1813_5
  have hr := e_1813_5
  simp only [e_1813_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2970)) := by
  have c110790 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.1
  have c110791 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.1
  have c110792 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.1
  have c110793 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.1
  have c110794 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110795 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_6 := arithEq_of_rows h (row := 1813) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_6
  simp only [← c110790, ← c110791, ← c110792] at e_1813_6
  have e_1813_7 := arithEq_of_rows h (row := 1813) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_7
  simp only [← c110793, ← c110794, k2, ← c110795] at e_1813_7
  have hr := e_1813_7
  simp only [e_1813_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2971)) := by
  have c110796 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110797 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110798 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110799 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110800 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110801 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_8 := arithEq_of_rows h (row := 1813) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_8
  simp only [← c110796, ← c110797, ← c110798] at e_1813_8
  have e_1813_9 := arithEq_of_rows h (row := 1813) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_9
  simp only [← c110799, ← c110800, k2, ← c110801] at e_1813_9
  have hr := e_1813_9
  simp only [e_1813_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2972)) := by
  have c110802 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110803 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110804 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110805 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110806 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110807 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_10 := arithEq_of_rows h (row := 1813) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_10
  simp only [← c110802, ← c110803, ← c110804] at e_1813_10
  have e_1813_11 := arithEq_of_rows h (row := 1813) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_11
  simp only [← c110805, ← c110806, k2, ← c110807] at e_1813_11
  have hr := e_1813_11
  simp only [e_1813_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2973)) := by
  have c110808 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110809 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110810 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110811 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110812 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110813 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_12 := arithEq_of_rows h (row := 1813) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_12
  simp only [← c110808, ← c110809, ← c110810] at e_1813_12
  have e_1813_13 := arithEq_of_rows h (row := 1813) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_13
  simp only [← c110811, ← c110812, k2, ← c110813] at e_1813_13
  have hr := e_1813_13
  simp only [e_1813_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2974)) := by
  have c110814 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110815 := (publicBatchWrapper89_copies3462 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110816 := (publicBatchWrapper89_copies3463 a h).1
  have c110817 := (publicBatchWrapper89_copies3463 a h).2.1
  have c110818 := (publicBatchWrapper89_copies3463 a h).2.2.1
  have c110819 := (publicBatchWrapper89_copies3463 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_14 := arithEq_of_rows h (row := 1813) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_14
  simp only [← c110814, ← c110815, ← c110816] at e_1813_14
  have e_1813_15 := arithEq_of_rows h (row := 1813) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_15
  simp only [← c110817, ← c110818, k2, ← c110819] at e_1813_15
  have hr := e_1813_15
  simp only [e_1813_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2975)) := by
  have c110820 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.1
  have c110821 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.1
  have c110822 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.1
  have c110823 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.1
  have c110824 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.1
  have c110825 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_16 := arithEq_of_rows h (row := 1813) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_16
  simp only [← c110820, ← c110821, ← c110822] at e_1813_16
  have e_1813_17 := arithEq_of_rows h (row := 1813) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_17
  simp only [← c110823, ← c110824, k2, ← c110825] at e_1813_17
  have hr := e_1813_17
  simp only [e_1813_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1813 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2976)) := by
  have c110826 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110827 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110828 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110829 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110830 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110831 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1813_18 := arithEq_of_rows h (row := 1813) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_18
  simp only [← c110826, ← c110827, ← c110828] at e_1813_18
  have e_1813_19 := arithEq_of_rows h (row := 1813) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1813_19
  simp only [← c110829, ← c110830, k2, ← c110831] at e_1813_19
  have hr := e_1813_19
  simp only [e_1813_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2977)) := by
  have c110832 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110833 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110834 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110835 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110836 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110837 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_0 := arithEq_of_rows h (row := 1814) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_0
  simp only [← c110832, ← c110833, ← c110834] at e_1814_0
  have e_1814_1 := arithEq_of_rows h (row := 1814) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_1
  simp only [← c110835, ← c110836, k2, ← c110837] at e_1814_1
  have hr := e_1814_1
  simp only [e_1814_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2978)) := by
  have c110838 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110839 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110840 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110841 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110842 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110843 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_2 := arithEq_of_rows h (row := 1814) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_2
  simp only [← c110838, ← c110839, ← c110840] at e_1814_2
  have e_1814_3 := arithEq_of_rows h (row := 1814) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_3
  simp only [← c110841, ← c110842, k2, ← c110843] at e_1814_3
  have hr := e_1814_3
  simp only [e_1814_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2979)) := by
  have c110844 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110845 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110846 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110847 := (publicBatchWrapper89_copies3463 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110848 := (publicBatchWrapper89_copies3464 a h).1
  have c110849 := (publicBatchWrapper89_copies3464 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_4 := arithEq_of_rows h (row := 1814) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_4
  simp only [← c110844, ← c110845, ← c110846] at e_1814_4
  have e_1814_5 := arithEq_of_rows h (row := 1814) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_5
  simp only [← c110847, ← c110848, k2, ← c110849] at e_1814_5
  have hr := e_1814_5
  simp only [e_1814_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2980)) := by
  have c110850 := (publicBatchWrapper89_copies3464 a h).2.2.1
  have c110851 := (publicBatchWrapper89_copies3464 a h).2.2.2.1
  have c110852 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.1
  have c110853 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.1
  have c110854 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.1
  have c110855 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_6 := arithEq_of_rows h (row := 1814) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_6
  simp only [← c110850, ← c110851, ← c110852] at e_1814_6
  have e_1814_7 := arithEq_of_rows h (row := 1814) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_7
  simp only [← c110853, ← c110854, k2, ← c110855] at e_1814_7
  have hr := e_1814_7
  simp only [e_1814_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2981)) := by
  have c110856 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.1
  have c110857 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.1
  have c110858 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110859 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110860 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110861 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_8 := arithEq_of_rows h (row := 1814) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_8
  simp only [← c110856, ← c110857, ← c110858] at e_1814_8
  have e_1814_9 := arithEq_of_rows h (row := 1814) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_9
  simp only [← c110859, ← c110860, k2, ← c110861] at e_1814_9
  have hr := e_1814_9
  simp only [e_1814_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2982)) := by
  have c110862 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110863 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110864 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110865 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110866 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110867 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_10 := arithEq_of_rows h (row := 1814) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_10
  simp only [← c110862, ← c110863, ← c110864] at e_1814_10
  have e_1814_11 := arithEq_of_rows h (row := 1814) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_11
  simp only [← c110865, ← c110866, k2, ← c110867] at e_1814_11
  have hr := e_1814_11
  simp only [e_1814_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2983)) := by
  have c110868 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110869 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110870 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110871 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110872 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110873 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_12 := arithEq_of_rows h (row := 1814) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_12
  simp only [← c110868, ← c110869, ← c110870] at e_1814_12
  have e_1814_13 := arithEq_of_rows h (row := 1814) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_13
  simp only [← c110871, ← c110872, k2, ← c110873] at e_1814_13
  have hr := e_1814_13
  simp only [e_1814_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2984)) := by
  have c110874 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110875 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110876 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110877 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110878 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110879 := (publicBatchWrapper89_copies3464 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_14 := arithEq_of_rows h (row := 1814) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_14
  simp only [← c110874, ← c110875, ← c110876] at e_1814_14
  have e_1814_15 := arithEq_of_rows h (row := 1814) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_15
  simp only [← c110877, ← c110878, k2, ← c110879] at e_1814_15
  have hr := e_1814_15
  simp only [e_1814_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2985)) := by
  have c110880 := (publicBatchWrapper89_copies3465 a h).1
  have c110881 := (publicBatchWrapper89_copies3465 a h).2.1
  have c110882 := (publicBatchWrapper89_copies3465 a h).2.2.1
  have c110883 := (publicBatchWrapper89_copies3465 a h).2.2.2.1
  have c110884 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.1
  have c110885 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_16 := arithEq_of_rows h (row := 1814) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_16
  simp only [← c110880, ← c110881, ← c110882] at e_1814_16
  have e_1814_17 := arithEq_of_rows h (row := 1814) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_17
  simp only [← c110883, ← c110884, k2, ← c110885] at e_1814_17
  have hr := e_1814_17
  simp only [e_1814_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1814 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2986)) := by
  have c110886 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.1
  have c110887 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.1
  have c110888 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.1
  have c110889 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.1
  have c110890 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110891 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1814_18 := arithEq_of_rows h (row := 1814) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_18
  simp only [← c110886, ← c110887, ← c110888] at e_1814_18
  have e_1814_19 := arithEq_of_rows h (row := 1814) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1814_19
  simp only [← c110889, ← c110890, k2, ← c110891] at e_1814_19
  have hr := e_1814_19
  simp only [e_1814_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2987)) := by
  have c110892 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110893 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110894 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110895 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110896 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110897 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_0 := arithEq_of_rows h (row := 1815) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_0
  simp only [← c110892, ← c110893, ← c110894] at e_1815_0
  have e_1815_1 := arithEq_of_rows h (row := 1815) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_1
  simp only [← c110895, ← c110896, k2, ← c110897] at e_1815_1
  have hr := e_1815_1
  simp only [e_1815_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2988)) := by
  have c110898 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110899 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110900 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110901 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110902 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110903 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_2 := arithEq_of_rows h (row := 1815) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_2
  simp only [← c110898, ← c110899, ← c110900] at e_1815_2
  have e_1815_3 := arithEq_of_rows h (row := 1815) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_3
  simp only [← c110901, ← c110902, k2, ← c110903] at e_1815_3
  have hr := e_1815_3
  simp only [e_1815_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2989)) := by
  have c110904 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110905 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110906 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110907 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110908 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110909 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_4 := arithEq_of_rows h (row := 1815) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_4
  simp only [← c110904, ← c110905, ← c110906] at e_1815_4
  have e_1815_5 := arithEq_of_rows h (row := 1815) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_5
  simp only [← c110907, ← c110908, k2, ← c110909] at e_1815_5
  have hr := e_1815_5
  simp only [e_1815_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2990)) := by
  have c110910 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110911 := (publicBatchWrapper89_copies3465 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110912 := (publicBatchWrapper89_copies3466 a h).1
  have c110913 := (publicBatchWrapper89_copies3466 a h).2.1
  have c110914 := (publicBatchWrapper89_copies3466 a h).2.2.1
  have c110915 := (publicBatchWrapper89_copies3466 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_6 := arithEq_of_rows h (row := 1815) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_6
  simp only [← c110910, ← c110911, ← c110912] at e_1815_6
  have e_1815_7 := arithEq_of_rows h (row := 1815) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_7
  simp only [← c110913, ← c110914, k2, ← c110915] at e_1815_7
  have hr := e_1815_7
  simp only [e_1815_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2991)) := by
  have c110916 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.1
  have c110917 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.1
  have c110918 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.1
  have c110919 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.1
  have c110920 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.1
  have c110921 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_8 := arithEq_of_rows h (row := 1815) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_8
  simp only [← c110916, ← c110917, ← c110918] at e_1815_8
  have e_1815_9 := arithEq_of_rows h (row := 1815) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_9
  simp only [← c110919, ← c110920, k2, ← c110921] at e_1815_9
  have hr := e_1815_9
  simp only [e_1815_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 47) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2992)) := by
  have c110922 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110923 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110924 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110925 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110926 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110927 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_10 := arithEq_of_rows h (row := 1815) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_10
  simp only [← c110922, ← c110923, ← c110924] at e_1815_10
  have e_1815_11 := arithEq_of_rows h (row := 1815) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_11
  simp only [← c110925, ← c110926, k2, ← c110927] at e_1815_11
  have hr := e_1815_11
  simp only [e_1815_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 55) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2993)) := by
  have c110928 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110929 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110930 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110931 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110932 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110933 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_12 := arithEq_of_rows h (row := 1815) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_12
  simp only [← c110928, ← c110929, ← c110930] at e_1815_12
  have e_1815_13 := arithEq_of_rows h (row := 1815) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_13
  simp only [← c110931, ← c110932, k2, ← c110933] at e_1815_13
  have hr := e_1815_13
  simp only [e_1815_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 63) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2994)) := by
  have c110934 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110935 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110936 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110937 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110938 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110939 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_14 := arithEq_of_rows h (row := 1815) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_14
  simp only [← c110934, ← c110935, ← c110936] at e_1815_14
  have e_1815_15 := arithEq_of_rows h (row := 1815) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_15
  simp only [← c110937, ← c110938, k2, ← c110939] at e_1815_15
  have hr := e_1815_15
  simp only [e_1815_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 71) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2995)) := by
  have c110940 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110941 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110942 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110943 := (publicBatchWrapper89_copies3466 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c110944 := (publicBatchWrapper89_copies3467 a h).1
  have c110945 := (publicBatchWrapper89_copies3467 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_16 := arithEq_of_rows h (row := 1815) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_16
  simp only [← c110940, ← c110941, ← c110942] at e_1815_16
  have e_1815_17 := arithEq_of_rows h (row := 1815) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_17
  simp only [← c110943, ← c110944, k2, ← c110945] at e_1815_17
  have hr := e_1815_17
  simp only [e_1815_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1815 79) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2996)) := by
  have c110946 := (publicBatchWrapper89_copies3467 a h).2.2.1
  have c110947 := (publicBatchWrapper89_copies3467 a h).2.2.2.1
  have c110948 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.1
  have c110949 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.1
  have c110950 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.1
  have c110951 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1815_18 := arithEq_of_rows h (row := 1815) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_18
  simp only [← c110946, ← c110947, ← c110948] at e_1815_18
  have e_1815_19 := arithEq_of_rows h (row := 1815) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1815_19
  simp only [← c110949, ← c110950, k2, ← c110951] at e_1815_19
  have hr := e_1815_19
  simp only [e_1815_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 7) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2997)) := by
  have c110952 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.1
  have c110953 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.1
  have c110954 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110955 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c110956 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110957 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_0 := arithEq_of_rows h (row := 1816) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_0
  simp only [← c110952, ← c110953, ← c110954] at e_1816_0
  have e_1816_1 := arithEq_of_rows h (row := 1816) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_1
  simp only [← c110955, ← c110956, k2, ← c110957] at e_1816_1
  have hr := e_1816_1
  simp only [e_1816_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 15) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2998)) := by
  have c110958 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110959 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110960 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110961 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110962 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110963 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_2 := arithEq_of_rows h (row := 1816) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_2
  simp only [← c110958, ← c110959, ← c110960] at e_1816_2
  have e_1816_3 := arithEq_of_rows h (row := 1816) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_3
  simp only [← c110961, ← c110962, k2, ← c110963] at e_1816_3
  have hr := e_1816_3
  simp only [e_1816_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 23) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 2999)) := by
  have c110964 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110965 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110966 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110967 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110968 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110969 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_4 := arithEq_of_rows h (row := 1816) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_4
  simp only [← c110964, ← c110965, ← c110966] at e_1816_4
  have e_1816_5 := arithEq_of_rows h (row := 1816) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_5
  simp only [← c110967, ← c110968, k2, ← c110969] at e_1816_5
  have hr := e_1816_5
  simp only [e_1816_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 31) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 3000)) := by
  have c110970 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110971 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110972 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110973 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110974 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110975 := (publicBatchWrapper89_copies3467 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_6 := arithEq_of_rows h (row := 1816) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_6
  simp only [← c110970, ← c110971, ← c110972] at e_1816_6
  have e_1816_7 := arithEq_of_rows h (row := 1816) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_7
  simp only [← c110973, ← c110974, k2, ← c110975] at e_1816_7
  have hr := e_1816_7
  simp only [e_1816_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 39) = bselect (a (.wire 15 35)) (a (.virt 200)) (a (.virt 3001)) := by
  have c110976 := (publicBatchWrapper89_copies3468 a h).1
  have c110977 := (publicBatchWrapper89_copies3468 a h).2.1
  have c110978 := (publicBatchWrapper89_copies3468 a h).2.2.1
  have c110979 := (publicBatchWrapper89_copies3468 a h).2.2.2.1
  have c110980 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.1
  have c110981 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_8 := arithEq_of_rows h (row := 1816) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_8
  simp only [← c110976, ← c110977, ← c110978] at e_1816_8
  have e_1816_9 := arithEq_of_rows h (row := 1816) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_9
  simp only [← c110979, ← c110980, k2, ← c110981] at e_1816_9
  have hr := e_1816_9
  simp only [e_1816_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3170)) := by
  have c110982 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.1
  have c110983 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.1
  have c110984 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.1
  have c110985 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.1
  have c110986 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.1
  have c110987 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_10 := arithEq_of_rows h (row := 1816) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_10
  simp only [← c110982, ← c110983, ← c110984] at e_1816_10
  have e_1816_11 := arithEq_of_rows h (row := 1816) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_11
  simp only [← c110985, ← c110986, k2, ← c110987] at e_1816_11
  have hr := e_1816_11
  simp only [e_1816_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3171)) := by
  have c110988 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110989 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110990 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110991 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110992 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110993 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_12 := arithEq_of_rows h (row := 1816) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_12
  simp only [← c110988, ← c110989, ← c110990] at e_1816_12
  have e_1816_13 := arithEq_of_rows h (row := 1816) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_13
  simp only [← c110991, ← c110992, k2, ← c110993] at e_1816_13
  have hr := e_1816_13
  simp only [e_1816_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3172)) := by
  have c110994 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110995 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110996 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110997 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110998 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c110999 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_14 := arithEq_of_rows h (row := 1816) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_14
  simp only [← c110994, ← c110995, ← c110996] at e_1816_14
  have e_1816_15 := arithEq_of_rows h (row := 1816) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_15
  simp only [← c110997, ← c110998, k2, ← c110999] at e_1816_15
  have hr := e_1816_15
  simp only [e_1816_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3173)) := by
  have c111000 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111001 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111002 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111003 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111004 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111005 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_16 := arithEq_of_rows h (row := 1816) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_16
  simp only [← c111000, ← c111001, ← c111002] at e_1816_16
  have e_1816_17 := arithEq_of_rows h (row := 1816) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_17
  simp only [← c111003, ← c111004, k2, ← c111005] at e_1816_17
  have hr := e_1816_17
  simp only [e_1816_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1816 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3174)) := by
  have c111006 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111007 := (publicBatchWrapper89_copies3468 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111008 := (publicBatchWrapper89_copies3469 a h).1
  have c111009 := (publicBatchWrapper89_copies3469 a h).2.1
  have c111010 := (publicBatchWrapper89_copies3469 a h).2.2.1
  have c111011 := (publicBatchWrapper89_copies3469 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1816_18 := arithEq_of_rows h (row := 1816) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_18
  simp only [← c111006, ← c111007, ← c111008] at e_1816_18
  have e_1816_19 := arithEq_of_rows h (row := 1816) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1816_19
  simp only [← c111009, ← c111010, k2, ← c111011] at e_1816_19
  have hr := e_1816_19
  simp only [e_1816_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3175)) := by
  have c111012 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.1
  have c111013 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.1
  have c111014 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.1
  have c111015 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.1
  have c111016 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.1
  have c111017 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_0 := arithEq_of_rows h (row := 1817) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_0
  simp only [← c111012, ← c111013, ← c111014] at e_1817_0
  have e_1817_1 := arithEq_of_rows h (row := 1817) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_1
  simp only [← c111015, ← c111016, k2, ← c111017] at e_1817_1
  have hr := e_1817_1
  simp only [e_1817_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3176)) := by
  have c111018 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111019 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111020 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111021 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111022 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111023 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_2 := arithEq_of_rows h (row := 1817) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_2
  simp only [← c111018, ← c111019, ← c111020] at e_1817_2
  have e_1817_3 := arithEq_of_rows h (row := 1817) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_3
  simp only [← c111021, ← c111022, k2, ← c111023] at e_1817_3
  have hr := e_1817_3
  simp only [e_1817_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3177)) := by
  have c111024 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111025 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111026 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111027 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111028 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111029 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_4 := arithEq_of_rows h (row := 1817) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_4
  simp only [← c111024, ← c111025, ← c111026] at e_1817_4
  have e_1817_5 := arithEq_of_rows h (row := 1817) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_5
  simp only [← c111027, ← c111028, k2, ← c111029] at e_1817_5
  have hr := e_1817_5
  simp only [e_1817_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3178)) := by
  have c111030 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111031 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111032 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111033 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111034 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111035 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_6 := arithEq_of_rows h (row := 1817) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_6
  simp only [← c111030, ← c111031, ← c111032] at e_1817_6
  have e_1817_7 := arithEq_of_rows h (row := 1817) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_7
  simp only [← c111033, ← c111034, k2, ← c111035] at e_1817_7
  have hr := e_1817_7
  simp only [e_1817_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3179)) := by
  have c111036 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111037 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111038 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111039 := (publicBatchWrapper89_copies3469 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111040 := (publicBatchWrapper89_copies3470 a h).1
  have c111041 := (publicBatchWrapper89_copies3470 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_8 := arithEq_of_rows h (row := 1817) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_8
  simp only [← c111036, ← c111037, ← c111038] at e_1817_8
  have e_1817_9 := arithEq_of_rows h (row := 1817) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_9
  simp only [← c111039, ← c111040, k2, ← c111041] at e_1817_9
  have hr := e_1817_9
  simp only [e_1817_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3180)) := by
  have c111042 := (publicBatchWrapper89_copies3470 a h).2.2.1
  have c111043 := (publicBatchWrapper89_copies3470 a h).2.2.2.1
  have c111044 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.1
  have c111045 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.1
  have c111046 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.1
  have c111047 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_10 := arithEq_of_rows h (row := 1817) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_10
  simp only [← c111042, ← c111043, ← c111044] at e_1817_10
  have e_1817_11 := arithEq_of_rows h (row := 1817) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_11
  simp only [← c111045, ← c111046, k2, ← c111047] at e_1817_11
  have hr := e_1817_11
  simp only [e_1817_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 55) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3181)) := by
  have c111048 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.1
  have c111049 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.1
  have c111050 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111051 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c111052 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111053 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_12 := arithEq_of_rows h (row := 1817) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_12
  simp only [← c111048, ← c111049, ← c111050] at e_1817_12
  have e_1817_13 := arithEq_of_rows h (row := 1817) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_13
  simp only [← c111051, ← c111052, k2, ← c111053] at e_1817_13
  have hr := e_1817_13
  simp only [e_1817_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 63) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3182)) := by
  have c111054 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111055 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111056 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111057 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111058 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111059 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_14 := arithEq_of_rows h (row := 1817) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_14
  simp only [← c111054, ← c111055, ← c111056] at e_1817_14
  have e_1817_15 := arithEq_of_rows h (row := 1817) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_15
  simp only [← c111057, ← c111058, k2, ← c111059] at e_1817_15
  have hr := e_1817_15
  simp only [e_1817_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 71) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3183)) := by
  have c111060 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111061 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111062 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111063 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111064 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111065 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_16 := arithEq_of_rows h (row := 1817) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_16
  simp only [← c111060, ← c111061, ← c111062] at e_1817_16
  have e_1817_17 := arithEq_of_rows h (row := 1817) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_17
  simp only [← c111063, ← c111064, k2, ← c111065] at e_1817_17
  have hr := e_1817_17
  simp only [e_1817_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1817 79) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3184)) := by
  have c111066 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111067 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111068 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111069 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111070 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111071 := (publicBatchWrapper89_copies3470 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1817_18 := arithEq_of_rows h (row := 1817) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_18
  simp only [← c111066, ← c111067, ← c111068] at e_1817_18
  have e_1817_19 := arithEq_of_rows h (row := 1817) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1817_19
  simp only [← c111069, ← c111070, k2, ← c111071] at e_1817_19
  have hr := e_1817_19
  simp only [e_1817_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 7) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3185)) := by
  have c111072 := (publicBatchWrapper89_copies3471 a h).1
  have c111073 := (publicBatchWrapper89_copies3471 a h).2.1
  have c111074 := (publicBatchWrapper89_copies3471 a h).2.2.1
  have c111075 := (publicBatchWrapper89_copies3471 a h).2.2.2.1
  have c111076 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.1
  have c111077 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_0 := arithEq_of_rows h (row := 1818) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_0
  simp only [← c111072, ← c111073, ← c111074] at e_1818_0
  have e_1818_1 := arithEq_of_rows h (row := 1818) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_1
  simp only [← c111075, ← c111076, k2, ← c111077] at e_1818_1
  have hr := e_1818_1
  simp only [e_1818_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 15) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3186)) := by
  have c111078 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.1
  have c111079 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.1
  have c111080 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.1
  have c111081 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.1
  have c111082 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.1
  have c111083 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_2 := arithEq_of_rows h (row := 1818) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_2
  simp only [← c111078, ← c111079, ← c111080] at e_1818_2
  have e_1818_3 := arithEq_of_rows h (row := 1818) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_3
  simp only [← c111081, ← c111082, k2, ← c111083] at e_1818_3
  have hr := e_1818_3
  simp only [e_1818_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 23) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3187)) := by
  have c111084 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111085 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111086 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111087 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111088 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111089 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_4 := arithEq_of_rows h (row := 1818) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_4
  simp only [← c111084, ← c111085, ← c111086] at e_1818_4
  have e_1818_5 := arithEq_of_rows h (row := 1818) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_5
  simp only [← c111087, ← c111088, k2, ← c111089] at e_1818_5
  have hr := e_1818_5
  simp only [e_1818_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 31) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3188)) := by
  have c111090 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111091 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111092 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111093 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111094 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111095 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_6 := arithEq_of_rows h (row := 1818) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_6
  simp only [← c111090, ← c111091, ← c111092] at e_1818_6
  have e_1818_7 := arithEq_of_rows h (row := 1818) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_7
  simp only [← c111093, ← c111094, k2, ← c111095] at e_1818_7
  have hr := e_1818_7
  simp only [e_1818_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 39) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3189)) := by
  have c111096 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111097 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111098 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111099 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111100 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111101 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_8 := arithEq_of_rows h (row := 1818) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_8
  simp only [← c111096, ← c111097, ← c111098] at e_1818_8
  have e_1818_9 := arithEq_of_rows h (row := 1818) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_9
  simp only [← c111099, ← c111100, k2, ← c111101] at e_1818_9
  have hr := e_1818_9
  simp only [e_1818_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f17766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1818 47) = bselect (a (.wire 15 79)) (a (.virt 200)) (a (.virt 3190)) := by
  have c111102 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c111103 := (publicBatchWrapper89_copies3471 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c111104 := (publicBatchWrapper89_copies3472 a h).1
  have c111105 := (publicBatchWrapper89_copies3472 a h).2.1
  have c111106 := (publicBatchWrapper89_copies3472 a h).2.2.1
  have c111107 := (publicBatchWrapper89_copies3472 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1818_10 := arithEq_of_rows h (row := 1818) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_10
  simp only [← c111102, ← c111103, ← c111104] at e_1818_10
  have e_1818_11 := arithEq_of_rows h (row := 1818) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1818_11
  simp only [← c111105, ← c111106, k2, ← c111107] at e_1818_11
  have hr := e_1818_11
  simp only [e_1818_10] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
