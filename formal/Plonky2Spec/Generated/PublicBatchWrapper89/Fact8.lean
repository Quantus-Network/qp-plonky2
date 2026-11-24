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

theorem publicBatchWrapper89_f6013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 643 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3717)) := by
  have c40584 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.1
  have c40585 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.1
  have c40586 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40587 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40588 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40589 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_643_4 := arithEq_of_rows h (row := 643) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_4
  simp only [← c40584, ← c40585, ← c40586] at e_643_4
  have e_643_5 := arithEq_of_rows h (row := 643) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_5
  simp only [← c40587, ← c40588, k2, ← c40589] at e_643_5
  have hr := e_643_5
  simp only [e_643_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 643 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3718)) := by
  have c40590 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40591 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40592 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40593 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40594 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40595 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_643_6 := arithEq_of_rows h (row := 643) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_6
  simp only [← c40590, ← c40591, ← c40592] at e_643_6
  have e_643_7 := arithEq_of_rows h (row := 643) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_7
  simp only [← c40593, ← c40594, k2, ← c40595] at e_643_7
  have hr := e_643_7
  simp only [e_643_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 643 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3719)) := by
  have c40596 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40597 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40598 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40599 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40600 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40601 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_643_8 := arithEq_of_rows h (row := 643) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_8
  simp only [← c40596, ← c40597, ← c40598] at e_643_8
  have e_643_9 := arithEq_of_rows h (row := 643) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_9
  simp only [← c40599, ← c40600, k2, ← c40601] at e_643_9
  have hr := e_643_9
  simp only [e_643_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 643 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3720)) := by
  have c40602 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40603 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40604 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40605 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40606 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40607 := (publicBatchWrapper89_copies1268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_643_10 := arithEq_of_rows h (row := 643) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_10
  simp only [← c40602, ← c40603, ← c40604] at e_643_10
  have e_643_11 := arithEq_of_rows h (row := 643) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_11
  simp only [← c40605, ← c40606, k2, ← c40607] at e_643_11
  have hr := e_643_11
  simp only [e_643_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 643 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3721)) := by
  have c40608 := (publicBatchWrapper89_copies1269 a h).1
  have c40609 := (publicBatchWrapper89_copies1269 a h).2.1
  have c40610 := (publicBatchWrapper89_copies1269 a h).2.2.1
  have c40611 := (publicBatchWrapper89_copies1269 a h).2.2.2.1
  have c40612 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.1
  have c40613 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_643_12 := arithEq_of_rows h (row := 643) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_12
  simp only [← c40608, ← c40609, ← c40610] at e_643_12
  have e_643_13 := arithEq_of_rows h (row := 643) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_13
  simp only [← c40611, ← c40612, k2, ← c40613] at e_643_13
  have hr := e_643_13
  simp only [e_643_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 643 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3722)) := by
  have c40614 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.1
  have c40615 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.1
  have c40616 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.1
  have c40617 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.1
  have c40618 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40619 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_643_14 := arithEq_of_rows h (row := 643) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_14
  simp only [← c40614, ← c40615, ← c40616] at e_643_14
  have e_643_15 := arithEq_of_rows h (row := 643) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_15
  simp only [← c40617, ← c40618, k2, ← c40619] at e_643_15
  have hr := e_643_15
  simp only [e_643_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 643 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3723)) := by
  have c40620 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40621 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40622 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40623 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40624 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40625 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_643_16 := arithEq_of_rows h (row := 643) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_16
  simp only [← c40620, ← c40621, ← c40622] at e_643_16
  have e_643_17 := arithEq_of_rows h (row := 643) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_17
  simp only [← c40623, ← c40624, k2, ← c40625] at e_643_17
  have hr := e_643_17
  simp only [e_643_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 643 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3724)) := by
  have c40626 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40627 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40628 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40629 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40630 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40631 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_643_18 := arithEq_of_rows h (row := 643) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_18
  simp only [← c40626, ← c40627, ← c40628] at e_643_18
  have e_643_19 := arithEq_of_rows h (row := 643) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_643_19
  simp only [← c40629, ← c40630, k2, ← c40631] at e_643_19
  have hr := e_643_19
  simp only [e_643_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 644 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3725)) := by
  have c40632 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40633 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40634 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40635 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40636 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40637 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_644_0 := arithEq_of_rows h (row := 644) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_0
  simp only [← c40632, ← c40633, ← c40634] at e_644_0
  have e_644_1 := arithEq_of_rows h (row := 644) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_1
  simp only [← c40635, ← c40636, k2, ← c40637] at e_644_1
  have hr := e_644_1
  simp only [e_644_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 644 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3726)) := by
  have c40638 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40639 := (publicBatchWrapper89_copies1269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40640 := (publicBatchWrapper89_copies1270 a h).1
  have c40641 := (publicBatchWrapper89_copies1270 a h).2.1
  have c40642 := (publicBatchWrapper89_copies1270 a h).2.2.1
  have c40643 := (publicBatchWrapper89_copies1270 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_644_2 := arithEq_of_rows h (row := 644) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_2
  simp only [← c40638, ← c40639, ← c40640] at e_644_2
  have e_644_3 := arithEq_of_rows h (row := 644) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_3
  simp only [← c40641, ← c40642, k2, ← c40643] at e_644_3
  have hr := e_644_3
  simp only [e_644_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 644 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3727)) := by
  have c40644 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.1
  have c40645 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.1
  have c40646 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.1
  have c40647 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.1
  have c40648 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.1
  have c40649 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_644_4 := arithEq_of_rows h (row := 644) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_4
  simp only [← c40644, ← c40645, ← c40646] at e_644_4
  have e_644_5 := arithEq_of_rows h (row := 644) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_5
  simp only [← c40647, ← c40648, k2, ← c40649] at e_644_5
  have hr := e_644_5
  simp only [e_644_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 644 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3728)) := by
  have c40650 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40651 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40652 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40653 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40654 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40655 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_644_6 := arithEq_of_rows h (row := 644) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_6
  simp only [← c40650, ← c40651, ← c40652] at e_644_6
  have e_644_7 := arithEq_of_rows h (row := 644) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_7
  simp only [← c40653, ← c40654, k2, ← c40655] at e_644_7
  have hr := e_644_7
  simp only [e_644_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 644 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3729)) := by
  have c40656 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40657 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40658 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40659 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40660 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40661 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_644_8 := arithEq_of_rows h (row := 644) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_8
  simp only [← c40656, ← c40657, ← c40658] at e_644_8
  have e_644_9 := arithEq_of_rows h (row := 644) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_9
  simp only [← c40659, ← c40660, k2, ← c40661] at e_644_9
  have hr := e_644_9
  simp only [e_644_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 644 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3730)) := by
  have c40662 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40663 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40664 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40665 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40666 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40667 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_644_10 := arithEq_of_rows h (row := 644) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_10
  simp only [← c40662, ← c40663, ← c40664] at e_644_10
  have e_644_11 := arithEq_of_rows h (row := 644) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_11
  simp only [← c40665, ← c40666, k2, ← c40667] at e_644_11
  have hr := e_644_11
  simp only [e_644_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 644 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3731)) := by
  have c40668 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40669 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40670 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40671 := (publicBatchWrapper89_copies1270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40672 := (publicBatchWrapper89_copies1271 a h).1
  have c40673 := (publicBatchWrapper89_copies1271 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_644_12 := arithEq_of_rows h (row := 644) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_12
  simp only [← c40668, ← c40669, ← c40670] at e_644_12
  have e_644_13 := arithEq_of_rows h (row := 644) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_13
  simp only [← c40671, ← c40672, k2, ← c40673] at e_644_13
  have hr := e_644_13
  simp only [e_644_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 644 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3732)) := by
  have c40674 := (publicBatchWrapper89_copies1271 a h).2.2.1
  have c40675 := (publicBatchWrapper89_copies1271 a h).2.2.2.1
  have c40676 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.1
  have c40677 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.1
  have c40678 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.1
  have c40679 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_644_14 := arithEq_of_rows h (row := 644) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_14
  simp only [← c40674, ← c40675, ← c40676] at e_644_14
  have e_644_15 := arithEq_of_rows h (row := 644) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_15
  simp only [← c40677, ← c40678, k2, ← c40679] at e_644_15
  have hr := e_644_15
  simp only [e_644_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 644 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3733)) := by
  have c40680 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.1
  have c40681 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.1
  have c40682 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40683 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40684 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40685 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_644_16 := arithEq_of_rows h (row := 644) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_16
  simp only [← c40680, ← c40681, ← c40682] at e_644_16
  have e_644_17 := arithEq_of_rows h (row := 644) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_17
  simp only [← c40683, ← c40684, k2, ← c40685] at e_644_17
  have hr := e_644_17
  simp only [e_644_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 644 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3734)) := by
  have c40686 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40687 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40688 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40689 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40690 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40691 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_644_18 := arithEq_of_rows h (row := 644) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_18
  simp only [← c40686, ← c40687, ← c40688] at e_644_18
  have e_644_19 := arithEq_of_rows h (row := 644) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_644_19
  simp only [← c40689, ← c40690, k2, ← c40691] at e_644_19
  have hr := e_644_19
  simp only [e_644_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 645 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3735)) := by
  have c40692 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40693 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40694 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40695 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40696 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40697 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_645_0 := arithEq_of_rows h (row := 645) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_0
  simp only [← c40692, ← c40693, ← c40694] at e_645_0
  have e_645_1 := arithEq_of_rows h (row := 645) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_1
  simp only [← c40695, ← c40696, k2, ← c40697] at e_645_1
  have hr := e_645_1
  simp only [e_645_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 645 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3736)) := by
  have c40698 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40699 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40700 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40701 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40702 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40703 := (publicBatchWrapper89_copies1271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_645_2 := arithEq_of_rows h (row := 645) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_2
  simp only [← c40698, ← c40699, ← c40700] at e_645_2
  have e_645_3 := arithEq_of_rows h (row := 645) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_3
  simp only [← c40701, ← c40702, k2, ← c40703] at e_645_3
  have hr := e_645_3
  simp only [e_645_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 645 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3737)) := by
  have c40704 := (publicBatchWrapper89_copies1272 a h).1
  have c40705 := (publicBatchWrapper89_copies1272 a h).2.1
  have c40706 := (publicBatchWrapper89_copies1272 a h).2.2.1
  have c40707 := (publicBatchWrapper89_copies1272 a h).2.2.2.1
  have c40708 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.1
  have c40709 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_645_4 := arithEq_of_rows h (row := 645) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_4
  simp only [← c40704, ← c40705, ← c40706] at e_645_4
  have e_645_5 := arithEq_of_rows h (row := 645) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_5
  simp only [← c40707, ← c40708, k2, ← c40709] at e_645_5
  have hr := e_645_5
  simp only [e_645_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 645 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3738)) := by
  have c40710 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.1
  have c40711 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.1
  have c40712 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.1
  have c40713 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.1
  have c40714 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40715 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_645_6 := arithEq_of_rows h (row := 645) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_6
  simp only [← c40710, ← c40711, ← c40712] at e_645_6
  have e_645_7 := arithEq_of_rows h (row := 645) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_7
  simp only [← c40713, ← c40714, k2, ← c40715] at e_645_7
  have hr := e_645_7
  simp only [e_645_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 645 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3739)) := by
  have c40716 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40717 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40718 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40719 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40720 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40721 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_645_8 := arithEq_of_rows h (row := 645) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_8
  simp only [← c40716, ← c40717, ← c40718] at e_645_8
  have e_645_9 := arithEq_of_rows h (row := 645) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_9
  simp only [← c40719, ← c40720, k2, ← c40721] at e_645_9
  have hr := e_645_9
  simp only [e_645_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 645 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3740)) := by
  have c40722 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40723 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40724 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40725 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40726 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40727 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_645_10 := arithEq_of_rows h (row := 645) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_10
  simp only [← c40722, ← c40723, ← c40724] at e_645_10
  have e_645_11 := arithEq_of_rows h (row := 645) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_11
  simp only [← c40725, ← c40726, k2, ← c40727] at e_645_11
  have hr := e_645_11
  simp only [e_645_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 645 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3741)) := by
  have c40728 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40729 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40730 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40731 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40732 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40733 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_645_12 := arithEq_of_rows h (row := 645) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_12
  simp only [← c40728, ← c40729, ← c40730] at e_645_12
  have e_645_13 := arithEq_of_rows h (row := 645) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_13
  simp only [← c40731, ← c40732, k2, ← c40733] at e_645_13
  have hr := e_645_13
  simp only [e_645_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 645 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3742)) := by
  have c40734 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40735 := (publicBatchWrapper89_copies1272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40736 := (publicBatchWrapper89_copies1273 a h).1
  have c40737 := (publicBatchWrapper89_copies1273 a h).2.1
  have c40738 := (publicBatchWrapper89_copies1273 a h).2.2.1
  have c40739 := (publicBatchWrapper89_copies1273 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_645_14 := arithEq_of_rows h (row := 645) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_14
  simp only [← c40734, ← c40735, ← c40736] at e_645_14
  have e_645_15 := arithEq_of_rows h (row := 645) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_15
  simp only [← c40737, ← c40738, k2, ← c40739] at e_645_15
  have hr := e_645_15
  simp only [e_645_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 645 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3743)) := by
  have c40740 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.1
  have c40741 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.1
  have c40742 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.1
  have c40743 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.1
  have c40744 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.1
  have c40745 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_645_16 := arithEq_of_rows h (row := 645) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_16
  simp only [← c40740, ← c40741, ← c40742] at e_645_16
  have e_645_17 := arithEq_of_rows h (row := 645) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_17
  simp only [← c40743, ← c40744, k2, ← c40745] at e_645_17
  have hr := e_645_17
  simp only [e_645_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 645 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3744)) := by
  have c40746 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40747 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40748 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40749 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40750 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40751 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_645_18 := arithEq_of_rows h (row := 645) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_18
  simp only [← c40746, ← c40747, ← c40748] at e_645_18
  have e_645_19 := arithEq_of_rows h (row := 645) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_645_19
  simp only [← c40749, ← c40750, k2, ← c40751] at e_645_19
  have hr := e_645_19
  simp only [e_645_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 646 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3745)) := by
  have c40752 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40753 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40754 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40755 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40756 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40757 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_646_0 := arithEq_of_rows h (row := 646) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_0
  simp only [← c40752, ← c40753, ← c40754] at e_646_0
  have e_646_1 := arithEq_of_rows h (row := 646) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_1
  simp only [← c40755, ← c40756, k2, ← c40757] at e_646_1
  have hr := e_646_1
  simp only [e_646_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 646 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3746)) := by
  have c40758 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40759 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40760 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40761 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40762 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40763 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_646_2 := arithEq_of_rows h (row := 646) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_2
  simp only [← c40758, ← c40759, ← c40760] at e_646_2
  have e_646_3 := arithEq_of_rows h (row := 646) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_3
  simp only [← c40761, ← c40762, k2, ← c40763] at e_646_3
  have hr := e_646_3
  simp only [e_646_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 646 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3747)) := by
  have c40764 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40765 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40766 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40767 := (publicBatchWrapper89_copies1273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40768 := (publicBatchWrapper89_copies1274 a h).1
  have c40769 := (publicBatchWrapper89_copies1274 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_646_4 := arithEq_of_rows h (row := 646) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_4
  simp only [← c40764, ← c40765, ← c40766] at e_646_4
  have e_646_5 := arithEq_of_rows h (row := 646) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_5
  simp only [← c40767, ← c40768, k2, ← c40769] at e_646_5
  have hr := e_646_5
  simp only [e_646_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 646 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3748)) := by
  have c40770 := (publicBatchWrapper89_copies1274 a h).2.2.1
  have c40771 := (publicBatchWrapper89_copies1274 a h).2.2.2.1
  have c40772 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.1
  have c40773 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.1
  have c40774 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.1
  have c40775 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_646_6 := arithEq_of_rows h (row := 646) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_6
  simp only [← c40770, ← c40771, ← c40772] at e_646_6
  have e_646_7 := arithEq_of_rows h (row := 646) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_7
  simp only [← c40773, ← c40774, k2, ← c40775] at e_646_7
  have hr := e_646_7
  simp only [e_646_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 646 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3749)) := by
  have c40776 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.1
  have c40777 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.1
  have c40778 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40779 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40780 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40781 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_646_8 := arithEq_of_rows h (row := 646) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_8
  simp only [← c40776, ← c40777, ← c40778] at e_646_8
  have e_646_9 := arithEq_of_rows h (row := 646) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_9
  simp only [← c40779, ← c40780, k2, ← c40781] at e_646_9
  have hr := e_646_9
  simp only [e_646_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 646 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3750)) := by
  have c40782 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40783 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40784 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40785 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40786 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40787 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_646_10 := arithEq_of_rows h (row := 646) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_10
  simp only [← c40782, ← c40783, ← c40784] at e_646_10
  have e_646_11 := arithEq_of_rows h (row := 646) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_11
  simp only [← c40785, ← c40786, k2, ← c40787] at e_646_11
  have hr := e_646_11
  simp only [e_646_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 646 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3751)) := by
  have c40788 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40789 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40790 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40791 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40792 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40793 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_646_12 := arithEq_of_rows h (row := 646) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_12
  simp only [← c40788, ← c40789, ← c40790] at e_646_12
  have e_646_13 := arithEq_of_rows h (row := 646) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_13
  simp only [← c40791, ← c40792, k2, ← c40793] at e_646_13
  have hr := e_646_13
  simp only [e_646_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 646 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3752)) := by
  have c40794 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40795 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40796 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40797 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40798 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40799 := (publicBatchWrapper89_copies1274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_646_14 := arithEq_of_rows h (row := 646) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_14
  simp only [← c40794, ← c40795, ← c40796] at e_646_14
  have e_646_15 := arithEq_of_rows h (row := 646) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_15
  simp only [← c40797, ← c40798, k2, ← c40799] at e_646_15
  have hr := e_646_15
  simp only [e_646_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 646 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3753)) := by
  have c40800 := (publicBatchWrapper89_copies1275 a h).1
  have c40801 := (publicBatchWrapper89_copies1275 a h).2.1
  have c40802 := (publicBatchWrapper89_copies1275 a h).2.2.1
  have c40803 := (publicBatchWrapper89_copies1275 a h).2.2.2.1
  have c40804 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.1
  have c40805 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_646_16 := arithEq_of_rows h (row := 646) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_16
  simp only [← c40800, ← c40801, ← c40802] at e_646_16
  have e_646_17 := arithEq_of_rows h (row := 646) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_17
  simp only [← c40803, ← c40804, k2, ← c40805] at e_646_17
  have hr := e_646_17
  simp only [e_646_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 646 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3754)) := by
  have c40806 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.1
  have c40807 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.1
  have c40808 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.1
  have c40809 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.1
  have c40810 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40811 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_646_18 := arithEq_of_rows h (row := 646) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_18
  simp only [← c40806, ← c40807, ← c40808] at e_646_18
  have e_646_19 := arithEq_of_rows h (row := 646) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_646_19
  simp only [← c40809, ← c40810, k2, ← c40811] at e_646_19
  have hr := e_646_19
  simp only [e_646_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 647 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3755)) := by
  have c40812 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40813 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40814 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40815 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40816 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40817 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_647_0 := arithEq_of_rows h (row := 647) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_0
  simp only [← c40812, ← c40813, ← c40814] at e_647_0
  have e_647_1 := arithEq_of_rows h (row := 647) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_1
  simp only [← c40815, ← c40816, k2, ← c40817] at e_647_1
  have hr := e_647_1
  simp only [e_647_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 647 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3756)) := by
  have c40818 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40819 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40820 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40821 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40822 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40823 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_647_2 := arithEq_of_rows h (row := 647) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_2
  simp only [← c40818, ← c40819, ← c40820] at e_647_2
  have e_647_3 := arithEq_of_rows h (row := 647) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_3
  simp only [← c40821, ← c40822, k2, ← c40823] at e_647_3
  have hr := e_647_3
  simp only [e_647_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 647 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3757)) := by
  have c40824 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40825 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40826 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40827 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40828 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40829 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_647_4 := arithEq_of_rows h (row := 647) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_4
  simp only [← c40824, ← c40825, ← c40826] at e_647_4
  have e_647_5 := arithEq_of_rows h (row := 647) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_5
  simp only [← c40827, ← c40828, k2, ← c40829] at e_647_5
  have hr := e_647_5
  simp only [e_647_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 647 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3758)) := by
  have c40830 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40831 := (publicBatchWrapper89_copies1275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40832 := (publicBatchWrapper89_copies1276 a h).1
  have c40833 := (publicBatchWrapper89_copies1276 a h).2.1
  have c40834 := (publicBatchWrapper89_copies1276 a h).2.2.1
  have c40835 := (publicBatchWrapper89_copies1276 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_647_6 := arithEq_of_rows h (row := 647) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_6
  simp only [← c40830, ← c40831, ← c40832] at e_647_6
  have e_647_7 := arithEq_of_rows h (row := 647) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_7
  simp only [← c40833, ← c40834, k2, ← c40835] at e_647_7
  have hr := e_647_7
  simp only [e_647_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 647 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3759)) := by
  have c40836 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.1
  have c40837 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.1
  have c40838 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.1
  have c40839 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.1
  have c40840 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.1
  have c40841 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_647_8 := arithEq_of_rows h (row := 647) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_8
  simp only [← c40836, ← c40837, ← c40838] at e_647_8
  have e_647_9 := arithEq_of_rows h (row := 647) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_9
  simp only [← c40839, ← c40840, k2, ← c40841] at e_647_9
  have hr := e_647_9
  simp only [e_647_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 647 47) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3760)) := by
  have c40842 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40843 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40844 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40845 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40846 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40847 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_647_10 := arithEq_of_rows h (row := 647) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_10
  simp only [← c40842, ← c40843, ← c40844] at e_647_10
  have e_647_11 := arithEq_of_rows h (row := 647) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_11
  simp only [← c40845, ← c40846, k2, ← c40847] at e_647_11
  have hr := e_647_11
  simp only [e_647_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 647 55) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3761)) := by
  have c40848 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40849 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40850 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40851 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40852 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40853 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_647_12 := arithEq_of_rows h (row := 647) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_12
  simp only [← c40848, ← c40849, ← c40850] at e_647_12
  have e_647_13 := arithEq_of_rows h (row := 647) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_13
  simp only [← c40851, ← c40852, k2, ← c40853] at e_647_13
  have hr := e_647_13
  simp only [e_647_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 647 63) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3762)) := by
  have c40854 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40855 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40856 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40857 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40858 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40859 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_647_14 := arithEq_of_rows h (row := 647) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_14
  simp only [← c40854, ← c40855, ← c40856] at e_647_14
  have e_647_15 := arithEq_of_rows h (row := 647) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_15
  simp only [← c40857, ← c40858, k2, ← c40859] at e_647_15
  have hr := e_647_15
  simp only [e_647_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 647 71) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3763)) := by
  have c40860 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40861 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40862 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40863 := (publicBatchWrapper89_copies1276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40864 := (publicBatchWrapper89_copies1277 a h).1
  have c40865 := (publicBatchWrapper89_copies1277 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_647_16 := arithEq_of_rows h (row := 647) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_16
  simp only [← c40860, ← c40861, ← c40862] at e_647_16
  have e_647_17 := arithEq_of_rows h (row := 647) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_17
  simp only [← c40863, ← c40864, k2, ← c40865] at e_647_17
  have hr := e_647_17
  simp only [e_647_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 647 79) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3764)) := by
  have c40866 := (publicBatchWrapper89_copies1277 a h).2.2.1
  have c40867 := (publicBatchWrapper89_copies1277 a h).2.2.2.1
  have c40868 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.1
  have c40869 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.1
  have c40870 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.1
  have c40871 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_647_18 := arithEq_of_rows h (row := 647) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_18
  simp only [← c40866, ← c40867, ← c40868] at e_647_18
  have e_647_19 := arithEq_of_rows h (row := 647) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_647_19
  simp only [← c40869, ← c40870, k2, ← c40871] at e_647_19
  have hr := e_647_19
  simp only [e_647_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 648 7) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3765)) := by
  have c40872 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.1
  have c40873 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.1
  have c40874 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40875 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40876 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40877 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_648_0 := arithEq_of_rows h (row := 648) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_0
  simp only [← c40872, ← c40873, ← c40874] at e_648_0
  have e_648_1 := arithEq_of_rows h (row := 648) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_1
  simp only [← c40875, ← c40876, k2, ← c40877] at e_648_1
  have hr := e_648_1
  simp only [e_648_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 648 15) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3766)) := by
  have c40878 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40879 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40880 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40881 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40882 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40883 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_648_2 := arithEq_of_rows h (row := 648) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_2
  simp only [← c40878, ← c40879, ← c40880] at e_648_2
  have e_648_3 := arithEq_of_rows h (row := 648) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_3
  simp only [← c40881, ← c40882, k2, ← c40883] at e_648_3
  have hr := e_648_3
  simp only [e_648_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 648 23) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3767)) := by
  have c40884 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40885 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40886 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40887 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40888 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40889 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_648_4 := arithEq_of_rows h (row := 648) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_4
  simp only [← c40884, ← c40885, ← c40886] at e_648_4
  have e_648_5 := arithEq_of_rows h (row := 648) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_5
  simp only [← c40887, ← c40888, k2, ← c40889] at e_648_5
  have hr := e_648_5
  simp only [e_648_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 648 31) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3768)) := by
  have c40890 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40891 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40892 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40893 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40894 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40895 := (publicBatchWrapper89_copies1277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_648_6 := arithEq_of_rows h (row := 648) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_6
  simp only [← c40890, ← c40891, ← c40892] at e_648_6
  have e_648_7 := arithEq_of_rows h (row := 648) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_7
  simp only [← c40893, ← c40894, k2, ← c40895] at e_648_7
  have hr := e_648_7
  simp only [e_648_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 648 39) = bselect (a (.wire 19 51)) (a (.virt 200)) (a (.virt 3769)) := by
  have c40896 := (publicBatchWrapper89_copies1278 a h).1
  have c40897 := (publicBatchWrapper89_copies1278 a h).2.1
  have c40898 := (publicBatchWrapper89_copies1278 a h).2.2.1
  have c40899 := (publicBatchWrapper89_copies1278 a h).2.2.2.1
  have c40900 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.1
  have c40901 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_648_8 := arithEq_of_rows h (row := 648) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_8
  simp only [← c40896, ← c40897, ← c40898] at e_648_8
  have e_648_9 := arithEq_of_rows h (row := 648) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_9
  simp only [← c40899, ← c40900, k2, ← c40901] at e_648_9
  have hr := e_648_9
  simp only [e_648_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 648 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3810)) := by
  have c40902 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.1
  have c40903 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.1
  have c40904 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.1
  have c40905 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.1
  have c40906 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40907 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_648_10 := arithEq_of_rows h (row := 648) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_10
  simp only [← c40902, ← c40903, ← c40904] at e_648_10
  have e_648_11 := arithEq_of_rows h (row := 648) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_11
  simp only [← c40905, ← c40906, k2, ← c40907] at e_648_11
  have hr := e_648_11
  simp only [e_648_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 648 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3811)) := by
  have c40908 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40909 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40910 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40911 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40912 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40913 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_648_12 := arithEq_of_rows h (row := 648) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_12
  simp only [← c40908, ← c40909, ← c40910] at e_648_12
  have e_648_13 := arithEq_of_rows h (row := 648) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_13
  simp only [← c40911, ← c40912, k2, ← c40913] at e_648_13
  have hr := e_648_13
  simp only [e_648_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 648 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3812)) := by
  have c40914 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40915 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40916 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40917 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40918 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40919 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_648_14 := arithEq_of_rows h (row := 648) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_14
  simp only [← c40914, ← c40915, ← c40916] at e_648_14
  have e_648_15 := arithEq_of_rows h (row := 648) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_15
  simp only [← c40917, ← c40918, k2, ← c40919] at e_648_15
  have hr := e_648_15
  simp only [e_648_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 648 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3813)) := by
  have c40920 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40921 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40922 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40923 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40924 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40925 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_648_16 := arithEq_of_rows h (row := 648) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_16
  simp only [← c40920, ← c40921, ← c40922] at e_648_16
  have e_648_17 := arithEq_of_rows h (row := 648) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_17
  simp only [← c40923, ← c40924, k2, ← c40925] at e_648_17
  have hr := e_648_17
  simp only [e_648_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 648 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3814)) := by
  have c40926 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40927 := (publicBatchWrapper89_copies1278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40928 := (publicBatchWrapper89_copies1279 a h).1
  have c40929 := (publicBatchWrapper89_copies1279 a h).2.1
  have c40930 := (publicBatchWrapper89_copies1279 a h).2.2.1
  have c40931 := (publicBatchWrapper89_copies1279 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_648_18 := arithEq_of_rows h (row := 648) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_18
  simp only [← c40926, ← c40927, ← c40928] at e_648_18
  have e_648_19 := arithEq_of_rows h (row := 648) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_648_19
  simp only [← c40929, ← c40930, k2, ← c40931] at e_648_19
  have hr := e_648_19
  simp only [e_648_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 649 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3815)) := by
  have c40932 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.1
  have c40933 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.1
  have c40934 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.1
  have c40935 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.1
  have c40936 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.1
  have c40937 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_649_0 := arithEq_of_rows h (row := 649) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_0
  simp only [← c40932, ← c40933, ← c40934] at e_649_0
  have e_649_1 := arithEq_of_rows h (row := 649) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_1
  simp only [← c40935, ← c40936, k2, ← c40937] at e_649_1
  have hr := e_649_1
  simp only [e_649_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 649 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3816)) := by
  have c40938 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40939 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40940 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40941 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40942 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40943 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_649_2 := arithEq_of_rows h (row := 649) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_2
  simp only [← c40938, ← c40939, ← c40940] at e_649_2
  have e_649_3 := arithEq_of_rows h (row := 649) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_3
  simp only [← c40941, ← c40942, k2, ← c40943] at e_649_3
  have hr := e_649_3
  simp only [e_649_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 649 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3817)) := by
  have c40944 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40945 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40946 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40947 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40948 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40949 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_649_4 := arithEq_of_rows h (row := 649) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_4
  simp only [← c40944, ← c40945, ← c40946] at e_649_4
  have e_649_5 := arithEq_of_rows h (row := 649) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_5
  simp only [← c40947, ← c40948, k2, ← c40949] at e_649_5
  have hr := e_649_5
  simp only [e_649_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 649 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3818)) := by
  have c40950 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40951 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40952 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40953 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40954 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40955 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_649_6 := arithEq_of_rows h (row := 649) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_6
  simp only [← c40950, ← c40951, ← c40952] at e_649_6
  have e_649_7 := arithEq_of_rows h (row := 649) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_7
  simp only [← c40953, ← c40954, k2, ← c40955] at e_649_7
  have hr := e_649_7
  simp only [e_649_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 649 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3819)) := by
  have c40956 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40957 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40958 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40959 := (publicBatchWrapper89_copies1279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c40960 := (publicBatchWrapper89_copies1280 a h).1
  have c40961 := (publicBatchWrapper89_copies1280 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_649_8 := arithEq_of_rows h (row := 649) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_8
  simp only [← c40956, ← c40957, ← c40958] at e_649_8
  have e_649_9 := arithEq_of_rows h (row := 649) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_9
  simp only [← c40959, ← c40960, k2, ← c40961] at e_649_9
  have hr := e_649_9
  simp only [e_649_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 649 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3820)) := by
  have c40962 := (publicBatchWrapper89_copies1280 a h).2.2.1
  have c40963 := (publicBatchWrapper89_copies1280 a h).2.2.2.1
  have c40964 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.1
  have c40965 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.1
  have c40966 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.1
  have c40967 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_649_10 := arithEq_of_rows h (row := 649) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_10
  simp only [← c40962, ← c40963, ← c40964] at e_649_10
  have e_649_11 := arithEq_of_rows h (row := 649) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_11
  simp only [← c40965, ← c40966, k2, ← c40967] at e_649_11
  have hr := e_649_11
  simp only [e_649_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 649 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3821)) := by
  have c40968 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.1
  have c40969 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.1
  have c40970 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.1
  have c40971 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c40972 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40973 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_649_12 := arithEq_of_rows h (row := 649) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_12
  simp only [← c40968, ← c40969, ← c40970] at e_649_12
  have e_649_13 := arithEq_of_rows h (row := 649) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_13
  simp only [← c40971, ← c40972, k2, ← c40973] at e_649_13
  have hr := e_649_13
  simp only [e_649_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 649 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3822)) := by
  have c40974 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40975 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40976 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40977 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40978 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40979 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_649_14 := arithEq_of_rows h (row := 649) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_14
  simp only [← c40974, ← c40975, ← c40976] at e_649_14
  have e_649_15 := arithEq_of_rows h (row := 649) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_15
  simp only [← c40977, ← c40978, k2, ← c40979] at e_649_15
  have hr := e_649_15
  simp only [e_649_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 649 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3823)) := by
  have c40980 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40981 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40982 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40983 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40984 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40985 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_649_16 := arithEq_of_rows h (row := 649) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_16
  simp only [← c40980, ← c40981, ← c40982] at e_649_16
  have e_649_17 := arithEq_of_rows h (row := 649) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_17
  simp only [← c40983, ← c40984, k2, ← c40985] at e_649_17
  have hr := e_649_17
  simp only [e_649_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 649 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3824)) := by
  have c40986 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40987 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40988 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40989 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40990 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c40991 := (publicBatchWrapper89_copies1280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_649_18 := arithEq_of_rows h (row := 649) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_18
  simp only [← c40986, ← c40987, ← c40988] at e_649_18
  have e_649_19 := arithEq_of_rows h (row := 649) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_649_19
  simp only [← c40989, ← c40990, k2, ← c40991] at e_649_19
  have hr := e_649_19
  simp only [e_649_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 650 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3825)) := by
  have c40992 := (publicBatchWrapper89_copies1281 a h).1
  have c40993 := (publicBatchWrapper89_copies1281 a h).2.1
  have c40994 := (publicBatchWrapper89_copies1281 a h).2.2.1
  have c40995 := (publicBatchWrapper89_copies1281 a h).2.2.2.1
  have c40996 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.1
  have c40997 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_650_0 := arithEq_of_rows h (row := 650) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_0
  simp only [← c40992, ← c40993, ← c40994] at e_650_0
  have e_650_1 := arithEq_of_rows h (row := 650) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_1
  simp only [← c40995, ← c40996, k2, ← c40997] at e_650_1
  have hr := e_650_1
  simp only [e_650_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 650 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3826)) := by
  have c40998 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.1
  have c40999 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.1
  have c41000 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.1
  have c41001 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.1
  have c41002 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41003 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_650_2 := arithEq_of_rows h (row := 650) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_2
  simp only [← c40998, ← c40999, ← c41000] at e_650_2
  have e_650_3 := arithEq_of_rows h (row := 650) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_3
  simp only [← c41001, ← c41002, k2, ← c41003] at e_650_3
  have hr := e_650_3
  simp only [e_650_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 650 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3827)) := by
  have c41004 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41005 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41006 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41007 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41008 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41009 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_650_4 := arithEq_of_rows h (row := 650) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_4
  simp only [← c41004, ← c41005, ← c41006] at e_650_4
  have e_650_5 := arithEq_of_rows h (row := 650) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_5
  simp only [← c41007, ← c41008, k2, ← c41009] at e_650_5
  have hr := e_650_5
  simp only [e_650_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 650 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3828)) := by
  have c41010 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41011 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41012 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41013 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41014 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41015 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_650_6 := arithEq_of_rows h (row := 650) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_6
  simp only [← c41010, ← c41011, ← c41012] at e_650_6
  have e_650_7 := arithEq_of_rows h (row := 650) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_7
  simp only [← c41013, ← c41014, k2, ← c41015] at e_650_7
  have hr := e_650_7
  simp only [e_650_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 650 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3829)) := by
  have c41016 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41017 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41018 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41019 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41020 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41021 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_650_8 := arithEq_of_rows h (row := 650) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_8
  simp only [← c41016, ← c41017, ← c41018] at e_650_8
  have e_650_9 := arithEq_of_rows h (row := 650) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_9
  simp only [← c41019, ← c41020, k2, ← c41021] at e_650_9
  have hr := e_650_9
  simp only [e_650_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 650 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3830)) := by
  have c41022 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41023 := (publicBatchWrapper89_copies1281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41024 := (publicBatchWrapper89_copies1282 a h).1
  have c41025 := (publicBatchWrapper89_copies1282 a h).2.1
  have c41026 := (publicBatchWrapper89_copies1282 a h).2.2.1
  have c41027 := (publicBatchWrapper89_copies1282 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_650_10 := arithEq_of_rows h (row := 650) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_10
  simp only [← c41022, ← c41023, ← c41024] at e_650_10
  have e_650_11 := arithEq_of_rows h (row := 650) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_11
  simp only [← c41025, ← c41026, k2, ← c41027] at e_650_11
  have hr := e_650_11
  simp only [e_650_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 650 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3831)) := by
  have c41028 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.1
  have c41029 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.1
  have c41030 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.1
  have c41031 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.1
  have c41032 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.1
  have c41033 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_650_12 := arithEq_of_rows h (row := 650) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_12
  simp only [← c41028, ← c41029, ← c41030] at e_650_12
  have e_650_13 := arithEq_of_rows h (row := 650) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_13
  simp only [← c41031, ← c41032, k2, ← c41033] at e_650_13
  have hr := e_650_13
  simp only [e_650_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 650 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3832)) := by
  have c41034 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41035 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41036 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41037 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41038 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41039 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_650_14 := arithEq_of_rows h (row := 650) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_14
  simp only [← c41034, ← c41035, ← c41036] at e_650_14
  have e_650_15 := arithEq_of_rows h (row := 650) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_15
  simp only [← c41037, ← c41038, k2, ← c41039] at e_650_15
  have hr := e_650_15
  simp only [e_650_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 650 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3833)) := by
  have c41040 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41041 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41042 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41043 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41044 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41045 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_650_16 := arithEq_of_rows h (row := 650) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_16
  simp only [← c41040, ← c41041, ← c41042] at e_650_16
  have e_650_17 := arithEq_of_rows h (row := 650) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_17
  simp only [← c41043, ← c41044, k2, ← c41045] at e_650_17
  have hr := e_650_17
  simp only [e_650_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 650 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3834)) := by
  have c41046 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41047 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41048 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41049 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41050 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41051 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_650_18 := arithEq_of_rows h (row := 650) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_18
  simp only [← c41046, ← c41047, ← c41048] at e_650_18
  have e_650_19 := arithEq_of_rows h (row := 650) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_650_19
  simp only [← c41049, ← c41050, k2, ← c41051] at e_650_19
  have hr := e_650_19
  simp only [e_650_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 651 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3835)) := by
  have c41052 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41053 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41054 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41055 := (publicBatchWrapper89_copies1282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41056 := (publicBatchWrapper89_copies1283 a h).1
  have c41057 := (publicBatchWrapper89_copies1283 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_651_0 := arithEq_of_rows h (row := 651) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_0
  simp only [← c41052, ← c41053, ← c41054] at e_651_0
  have e_651_1 := arithEq_of_rows h (row := 651) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_1
  simp only [← c41055, ← c41056, k2, ← c41057] at e_651_1
  have hr := e_651_1
  simp only [e_651_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 651 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3836)) := by
  have c41058 := (publicBatchWrapper89_copies1283 a h).2.2.1
  have c41059 := (publicBatchWrapper89_copies1283 a h).2.2.2.1
  have c41060 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.1
  have c41061 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.1
  have c41062 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.1
  have c41063 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_651_2 := arithEq_of_rows h (row := 651) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_2
  simp only [← c41058, ← c41059, ← c41060] at e_651_2
  have e_651_3 := arithEq_of_rows h (row := 651) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_3
  simp only [← c41061, ← c41062, k2, ← c41063] at e_651_3
  have hr := e_651_3
  simp only [e_651_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 651 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3837)) := by
  have c41064 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.1
  have c41065 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.1
  have c41066 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41067 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41068 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41069 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_651_4 := arithEq_of_rows h (row := 651) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_4
  simp only [← c41064, ← c41065, ← c41066] at e_651_4
  have e_651_5 := arithEq_of_rows h (row := 651) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_5
  simp only [← c41067, ← c41068, k2, ← c41069] at e_651_5
  have hr := e_651_5
  simp only [e_651_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 651 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3838)) := by
  have c41070 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41071 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41072 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41073 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41074 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41075 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_651_6 := arithEq_of_rows h (row := 651) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_6
  simp only [← c41070, ← c41071, ← c41072] at e_651_6
  have e_651_7 := arithEq_of_rows h (row := 651) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_7
  simp only [← c41073, ← c41074, k2, ← c41075] at e_651_7
  have hr := e_651_7
  simp only [e_651_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 651 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3839)) := by
  have c41076 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41077 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41078 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41079 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41080 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41081 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_651_8 := arithEq_of_rows h (row := 651) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_8
  simp only [← c41076, ← c41077, ← c41078] at e_651_8
  have e_651_9 := arithEq_of_rows h (row := 651) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_9
  simp only [← c41079, ← c41080, k2, ← c41081] at e_651_9
  have hr := e_651_9
  simp only [e_651_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 651 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3840)) := by
  have c41082 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41083 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41084 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41085 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41086 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41087 := (publicBatchWrapper89_copies1283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_651_10 := arithEq_of_rows h (row := 651) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_10
  simp only [← c41082, ← c41083, ← c41084] at e_651_10
  have e_651_11 := arithEq_of_rows h (row := 651) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_11
  simp only [← c41085, ← c41086, k2, ← c41087] at e_651_11
  have hr := e_651_11
  simp only [e_651_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 651 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3841)) := by
  have c41088 := (publicBatchWrapper89_copies1284 a h).1
  have c41089 := (publicBatchWrapper89_copies1284 a h).2.1
  have c41090 := (publicBatchWrapper89_copies1284 a h).2.2.1
  have c41091 := (publicBatchWrapper89_copies1284 a h).2.2.2.1
  have c41092 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.1
  have c41093 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_651_12 := arithEq_of_rows h (row := 651) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_12
  simp only [← c41088, ← c41089, ← c41090] at e_651_12
  have e_651_13 := arithEq_of_rows h (row := 651) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_13
  simp only [← c41091, ← c41092, k2, ← c41093] at e_651_13
  have hr := e_651_13
  simp only [e_651_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 651 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3842)) := by
  have c41094 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.1
  have c41095 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.1
  have c41096 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.1
  have c41097 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.1
  have c41098 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41099 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_651_14 := arithEq_of_rows h (row := 651) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_14
  simp only [← c41094, ← c41095, ← c41096] at e_651_14
  have e_651_15 := arithEq_of_rows h (row := 651) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_15
  simp only [← c41097, ← c41098, k2, ← c41099] at e_651_15
  have hr := e_651_15
  simp only [e_651_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 651 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3843)) := by
  have c41100 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41101 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41102 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41103 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41104 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41105 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_651_16 := arithEq_of_rows h (row := 651) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_16
  simp only [← c41100, ← c41101, ← c41102] at e_651_16
  have e_651_17 := arithEq_of_rows h (row := 651) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_17
  simp only [← c41103, ← c41104, k2, ← c41105] at e_651_17
  have hr := e_651_17
  simp only [e_651_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 651 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3844)) := by
  have c41106 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41107 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41108 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41109 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41110 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41111 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_651_18 := arithEq_of_rows h (row := 651) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_18
  simp only [← c41106, ← c41107, ← c41108] at e_651_18
  have e_651_19 := arithEq_of_rows h (row := 651) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_651_19
  simp only [← c41109, ← c41110, k2, ← c41111] at e_651_19
  have hr := e_651_19
  simp only [e_651_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 652 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3845)) := by
  have c41112 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41113 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41114 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41115 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41116 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41117 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_652_0 := arithEq_of_rows h (row := 652) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_0
  simp only [← c41112, ← c41113, ← c41114] at e_652_0
  have e_652_1 := arithEq_of_rows h (row := 652) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_1
  simp only [← c41115, ← c41116, k2, ← c41117] at e_652_1
  have hr := e_652_1
  simp only [e_652_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 652 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3846)) := by
  have c41118 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41119 := (publicBatchWrapper89_copies1284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41120 := (publicBatchWrapper89_copies1285 a h).1
  have c41121 := (publicBatchWrapper89_copies1285 a h).2.1
  have c41122 := (publicBatchWrapper89_copies1285 a h).2.2.1
  have c41123 := (publicBatchWrapper89_copies1285 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_652_2 := arithEq_of_rows h (row := 652) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_2
  simp only [← c41118, ← c41119, ← c41120] at e_652_2
  have e_652_3 := arithEq_of_rows h (row := 652) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_3
  simp only [← c41121, ← c41122, k2, ← c41123] at e_652_3
  have hr := e_652_3
  simp only [e_652_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 652 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3847)) := by
  have c41124 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.1
  have c41125 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.1
  have c41126 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.1
  have c41127 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.1
  have c41128 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.1
  have c41129 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_652_4 := arithEq_of_rows h (row := 652) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_4
  simp only [← c41124, ← c41125, ← c41126] at e_652_4
  have e_652_5 := arithEq_of_rows h (row := 652) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_5
  simp only [← c41127, ← c41128, k2, ← c41129] at e_652_5
  have hr := e_652_5
  simp only [e_652_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 652 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3848)) := by
  have c41130 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41131 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41132 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41133 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41134 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41135 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_652_6 := arithEq_of_rows h (row := 652) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_6
  simp only [← c41130, ← c41131, ← c41132] at e_652_6
  have e_652_7 := arithEq_of_rows h (row := 652) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_7
  simp only [← c41133, ← c41134, k2, ← c41135] at e_652_7
  have hr := e_652_7
  simp only [e_652_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 652 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3849)) := by
  have c41136 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41137 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41138 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41139 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41140 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41141 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_652_8 := arithEq_of_rows h (row := 652) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_8
  simp only [← c41136, ← c41137, ← c41138] at e_652_8
  have e_652_9 := arithEq_of_rows h (row := 652) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_9
  simp only [← c41139, ← c41140, k2, ← c41141] at e_652_9
  have hr := e_652_9
  simp only [e_652_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 652 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3850)) := by
  have c41142 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41143 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41144 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41145 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41146 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41147 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_652_10 := arithEq_of_rows h (row := 652) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_10
  simp only [← c41142, ← c41143, ← c41144] at e_652_10
  have e_652_11 := arithEq_of_rows h (row := 652) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_11
  simp only [← c41145, ← c41146, k2, ← c41147] at e_652_11
  have hr := e_652_11
  simp only [e_652_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 652 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3851)) := by
  have c41148 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41149 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41150 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41151 := (publicBatchWrapper89_copies1285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41152 := (publicBatchWrapper89_copies1286 a h).1
  have c41153 := (publicBatchWrapper89_copies1286 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_652_12 := arithEq_of_rows h (row := 652) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_12
  simp only [← c41148, ← c41149, ← c41150] at e_652_12
  have e_652_13 := arithEq_of_rows h (row := 652) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_13
  simp only [← c41151, ← c41152, k2, ← c41153] at e_652_13
  have hr := e_652_13
  simp only [e_652_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 652 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3852)) := by
  have c41154 := (publicBatchWrapper89_copies1286 a h).2.2.1
  have c41155 := (publicBatchWrapper89_copies1286 a h).2.2.2.1
  have c41156 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.1
  have c41157 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.1
  have c41158 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.1
  have c41159 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_652_14 := arithEq_of_rows h (row := 652) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_14
  simp only [← c41154, ← c41155, ← c41156] at e_652_14
  have e_652_15 := arithEq_of_rows h (row := 652) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_15
  simp only [← c41157, ← c41158, k2, ← c41159] at e_652_15
  have hr := e_652_15
  simp only [e_652_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 652 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3853)) := by
  have c41160 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.1
  have c41161 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.1
  have c41162 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41163 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41164 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41165 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_652_16 := arithEq_of_rows h (row := 652) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_16
  simp only [← c41160, ← c41161, ← c41162] at e_652_16
  have e_652_17 := arithEq_of_rows h (row := 652) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_17
  simp only [← c41163, ← c41164, k2, ← c41165] at e_652_17
  have hr := e_652_17
  simp only [e_652_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 652 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3854)) := by
  have c41166 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41167 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41168 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41169 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41170 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41171 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_652_18 := arithEq_of_rows h (row := 652) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_18
  simp only [← c41166, ← c41167, ← c41168] at e_652_18
  have e_652_19 := arithEq_of_rows h (row := 652) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_652_19
  simp only [← c41169, ← c41170, k2, ← c41171] at e_652_19
  have hr := e_652_19
  simp only [e_652_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 653 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3855)) := by
  have c41172 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41173 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41174 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41175 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41176 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41177 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_653_0 := arithEq_of_rows h (row := 653) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_0
  simp only [← c41172, ← c41173, ← c41174] at e_653_0
  have e_653_1 := arithEq_of_rows h (row := 653) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_1
  simp only [← c41175, ← c41176, k2, ← c41177] at e_653_1
  have hr := e_653_1
  simp only [e_653_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 653 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3856)) := by
  have c41178 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41179 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41180 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41181 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41182 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41183 := (publicBatchWrapper89_copies1286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_653_2 := arithEq_of_rows h (row := 653) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_2
  simp only [← c41178, ← c41179, ← c41180] at e_653_2
  have e_653_3 := arithEq_of_rows h (row := 653) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_3
  simp only [← c41181, ← c41182, k2, ← c41183] at e_653_3
  have hr := e_653_3
  simp only [e_653_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 653 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3857)) := by
  have c41184 := (publicBatchWrapper89_copies1287 a h).1
  have c41185 := (publicBatchWrapper89_copies1287 a h).2.1
  have c41186 := (publicBatchWrapper89_copies1287 a h).2.2.1
  have c41187 := (publicBatchWrapper89_copies1287 a h).2.2.2.1
  have c41188 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.1
  have c41189 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_653_4 := arithEq_of_rows h (row := 653) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_4
  simp only [← c41184, ← c41185, ← c41186] at e_653_4
  have e_653_5 := arithEq_of_rows h (row := 653) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_5
  simp only [← c41187, ← c41188, k2, ← c41189] at e_653_5
  have hr := e_653_5
  simp only [e_653_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 653 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3858)) := by
  have c41190 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.1
  have c41191 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.1
  have c41192 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.1
  have c41193 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.1
  have c41194 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41195 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_653_6 := arithEq_of_rows h (row := 653) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_6
  simp only [← c41190, ← c41191, ← c41192] at e_653_6
  have e_653_7 := arithEq_of_rows h (row := 653) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_7
  simp only [← c41193, ← c41194, k2, ← c41195] at e_653_7
  have hr := e_653_7
  simp only [e_653_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 653 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3859)) := by
  have c41196 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41197 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41198 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41199 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41200 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41201 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_653_8 := arithEq_of_rows h (row := 653) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_8
  simp only [← c41196, ← c41197, ← c41198] at e_653_8
  have e_653_9 := arithEq_of_rows h (row := 653) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_9
  simp only [← c41199, ← c41200, k2, ← c41201] at e_653_9
  have hr := e_653_9
  simp only [e_653_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 653 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3860)) := by
  have c41202 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41203 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41204 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41205 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41206 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41207 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_653_10 := arithEq_of_rows h (row := 653) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_10
  simp only [← c41202, ← c41203, ← c41204] at e_653_10
  have e_653_11 := arithEq_of_rows h (row := 653) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_11
  simp only [← c41205, ← c41206, k2, ← c41207] at e_653_11
  have hr := e_653_11
  simp only [e_653_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 653 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3861)) := by
  have c41208 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41209 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41210 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41211 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41212 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41213 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_653_12 := arithEq_of_rows h (row := 653) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_12
  simp only [← c41208, ← c41209, ← c41210] at e_653_12
  have e_653_13 := arithEq_of_rows h (row := 653) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_13
  simp only [← c41211, ← c41212, k2, ← c41213] at e_653_13
  have hr := e_653_13
  simp only [e_653_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 653 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3862)) := by
  have c41214 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41215 := (publicBatchWrapper89_copies1287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41216 := (publicBatchWrapper89_copies1288 a h).1
  have c41217 := (publicBatchWrapper89_copies1288 a h).2.1
  have c41218 := (publicBatchWrapper89_copies1288 a h).2.2.1
  have c41219 := (publicBatchWrapper89_copies1288 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_653_14 := arithEq_of_rows h (row := 653) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_14
  simp only [← c41214, ← c41215, ← c41216] at e_653_14
  have e_653_15 := arithEq_of_rows h (row := 653) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_15
  simp only [← c41217, ← c41218, k2, ← c41219] at e_653_15
  have hr := e_653_15
  simp only [e_653_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 653 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3863)) := by
  have c41220 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.1
  have c41221 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.1
  have c41222 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.1
  have c41223 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.1
  have c41224 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.1
  have c41225 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_653_16 := arithEq_of_rows h (row := 653) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_16
  simp only [← c41220, ← c41221, ← c41222] at e_653_16
  have e_653_17 := arithEq_of_rows h (row := 653) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_17
  simp only [← c41223, ← c41224, k2, ← c41225] at e_653_17
  have hr := e_653_17
  simp only [e_653_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 653 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3864)) := by
  have c41226 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41227 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41228 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41229 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41230 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41231 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_653_18 := arithEq_of_rows h (row := 653) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_18
  simp only [← c41226, ← c41227, ← c41228] at e_653_18
  have e_653_19 := arithEq_of_rows h (row := 653) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_653_19
  simp only [← c41229, ← c41230, k2, ← c41231] at e_653_19
  have hr := e_653_19
  simp only [e_653_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 654 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3865)) := by
  have c41232 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41233 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41234 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41235 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41236 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41237 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_654_0 := arithEq_of_rows h (row := 654) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_0
  simp only [← c41232, ← c41233, ← c41234] at e_654_0
  have e_654_1 := arithEq_of_rows h (row := 654) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_1
  simp only [← c41235, ← c41236, k2, ← c41237] at e_654_1
  have hr := e_654_1
  simp only [e_654_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 654 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3866)) := by
  have c41238 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41239 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41240 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41241 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41242 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41243 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_654_2 := arithEq_of_rows h (row := 654) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_2
  simp only [← c41238, ← c41239, ← c41240] at e_654_2
  have e_654_3 := arithEq_of_rows h (row := 654) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_3
  simp only [← c41241, ← c41242, k2, ← c41243] at e_654_3
  have hr := e_654_3
  simp only [e_654_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 654 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3867)) := by
  have c41244 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41245 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41246 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41247 := (publicBatchWrapper89_copies1288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41248 := (publicBatchWrapper89_copies1289 a h).1
  have c41249 := (publicBatchWrapper89_copies1289 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_654_4 := arithEq_of_rows h (row := 654) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_4
  simp only [← c41244, ← c41245, ← c41246] at e_654_4
  have e_654_5 := arithEq_of_rows h (row := 654) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_5
  simp only [← c41247, ← c41248, k2, ← c41249] at e_654_5
  have hr := e_654_5
  simp only [e_654_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 654 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3868)) := by
  have c41250 := (publicBatchWrapper89_copies1289 a h).2.2.1
  have c41251 := (publicBatchWrapper89_copies1289 a h).2.2.2.1
  have c41252 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.1
  have c41253 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.1
  have c41254 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.1
  have c41255 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_654_6 := arithEq_of_rows h (row := 654) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_6
  simp only [← c41250, ← c41251, ← c41252] at e_654_6
  have e_654_7 := arithEq_of_rows h (row := 654) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_7
  simp only [← c41253, ← c41254, k2, ← c41255] at e_654_7
  have hr := e_654_7
  simp only [e_654_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 654 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3869)) := by
  have c41256 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.1
  have c41257 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.1
  have c41258 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41259 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41260 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41261 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_654_8 := arithEq_of_rows h (row := 654) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_8
  simp only [← c41256, ← c41257, ← c41258] at e_654_8
  have e_654_9 := arithEq_of_rows h (row := 654) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_9
  simp only [← c41259, ← c41260, k2, ← c41261] at e_654_9
  have hr := e_654_9
  simp only [e_654_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 654 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3870)) := by
  have c41262 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41263 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41264 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41265 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41266 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41267 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_654_10 := arithEq_of_rows h (row := 654) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_10
  simp only [← c41262, ← c41263, ← c41264] at e_654_10
  have e_654_11 := arithEq_of_rows h (row := 654) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_11
  simp only [← c41265, ← c41266, k2, ← c41267] at e_654_11
  have hr := e_654_11
  simp only [e_654_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 654 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3871)) := by
  have c41268 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41269 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41270 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41271 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41272 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41273 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_654_12 := arithEq_of_rows h (row := 654) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_12
  simp only [← c41268, ← c41269, ← c41270] at e_654_12
  have e_654_13 := arithEq_of_rows h (row := 654) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_13
  simp only [← c41271, ← c41272, k2, ← c41273] at e_654_13
  have hr := e_654_13
  simp only [e_654_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 654 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3872)) := by
  have c41274 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41275 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41276 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41277 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41278 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41279 := (publicBatchWrapper89_copies1289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_654_14 := arithEq_of_rows h (row := 654) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_14
  simp only [← c41274, ← c41275, ← c41276] at e_654_14
  have e_654_15 := arithEq_of_rows h (row := 654) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_15
  simp only [← c41277, ← c41278, k2, ← c41279] at e_654_15
  have hr := e_654_15
  simp only [e_654_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 654 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3873)) := by
  have c41280 := (publicBatchWrapper89_copies1290 a h).1
  have c41281 := (publicBatchWrapper89_copies1290 a h).2.1
  have c41282 := (publicBatchWrapper89_copies1290 a h).2.2.1
  have c41283 := (publicBatchWrapper89_copies1290 a h).2.2.2.1
  have c41284 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.1
  have c41285 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_654_16 := arithEq_of_rows h (row := 654) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_16
  simp only [← c41280, ← c41281, ← c41282] at e_654_16
  have e_654_17 := arithEq_of_rows h (row := 654) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_17
  simp only [← c41283, ← c41284, k2, ← c41285] at e_654_17
  have hr := e_654_17
  simp only [e_654_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 654 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3874)) := by
  have c41286 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.1
  have c41287 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.1
  have c41288 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.1
  have c41289 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.1
  have c41290 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41291 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_654_18 := arithEq_of_rows h (row := 654) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_18
  simp only [← c41286, ← c41287, ← c41288] at e_654_18
  have e_654_19 := arithEq_of_rows h (row := 654) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_654_19
  simp only [← c41289, ← c41290, k2, ← c41291] at e_654_19
  have hr := e_654_19
  simp only [e_654_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 655 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3875)) := by
  have c41292 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41293 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41294 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41295 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41296 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41297 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_655_0 := arithEq_of_rows h (row := 655) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_0
  simp only [← c41292, ← c41293, ← c41294] at e_655_0
  have e_655_1 := arithEq_of_rows h (row := 655) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_1
  simp only [← c41295, ← c41296, k2, ← c41297] at e_655_1
  have hr := e_655_1
  simp only [e_655_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 655 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3876)) := by
  have c41298 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41299 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41300 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41301 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41302 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41303 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_655_2 := arithEq_of_rows h (row := 655) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_2
  simp only [← c41298, ← c41299, ← c41300] at e_655_2
  have e_655_3 := arithEq_of_rows h (row := 655) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_3
  simp only [← c41301, ← c41302, k2, ← c41303] at e_655_3
  have hr := e_655_3
  simp only [e_655_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 655 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3877)) := by
  have c41304 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41305 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41306 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41307 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41308 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41309 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_655_4 := arithEq_of_rows h (row := 655) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_4
  simp only [← c41304, ← c41305, ← c41306] at e_655_4
  have e_655_5 := arithEq_of_rows h (row := 655) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_5
  simp only [← c41307, ← c41308, k2, ← c41309] at e_655_5
  have hr := e_655_5
  simp only [e_655_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 655 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3878)) := by
  have c41310 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41311 := (publicBatchWrapper89_copies1290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41312 := (publicBatchWrapper89_copies1291 a h).1
  have c41313 := (publicBatchWrapper89_copies1291 a h).2.1
  have c41314 := (publicBatchWrapper89_copies1291 a h).2.2.1
  have c41315 := (publicBatchWrapper89_copies1291 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_655_6 := arithEq_of_rows h (row := 655) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_6
  simp only [← c41310, ← c41311, ← c41312] at e_655_6
  have e_655_7 := arithEq_of_rows h (row := 655) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_7
  simp only [← c41313, ← c41314, k2, ← c41315] at e_655_7
  have hr := e_655_7
  simp only [e_655_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 655 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3879)) := by
  have c41316 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.1
  have c41317 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.1
  have c41318 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.1
  have c41319 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.1
  have c41320 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.1
  have c41321 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_655_8 := arithEq_of_rows h (row := 655) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_8
  simp only [← c41316, ← c41317, ← c41318] at e_655_8
  have e_655_9 := arithEq_of_rows h (row := 655) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_9
  simp only [← c41319, ← c41320, k2, ← c41321] at e_655_9
  have hr := e_655_9
  simp only [e_655_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 655 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3880)) := by
  have c41322 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41323 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41324 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41325 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41326 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41327 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_655_10 := arithEq_of_rows h (row := 655) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_10
  simp only [← c41322, ← c41323, ← c41324] at e_655_10
  have e_655_11 := arithEq_of_rows h (row := 655) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_11
  simp only [← c41325, ← c41326, k2, ← c41327] at e_655_11
  have hr := e_655_11
  simp only [e_655_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 655 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3881)) := by
  have c41328 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41329 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41330 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41331 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41332 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41333 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_655_12 := arithEq_of_rows h (row := 655) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_12
  simp only [← c41328, ← c41329, ← c41330] at e_655_12
  have e_655_13 := arithEq_of_rows h (row := 655) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_13
  simp only [← c41331, ← c41332, k2, ← c41333] at e_655_13
  have hr := e_655_13
  simp only [e_655_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 655 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3882)) := by
  have c41334 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41335 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41336 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41337 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41338 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41339 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_655_14 := arithEq_of_rows h (row := 655) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_14
  simp only [← c41334, ← c41335, ← c41336] at e_655_14
  have e_655_15 := arithEq_of_rows h (row := 655) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_15
  simp only [← c41337, ← c41338, k2, ← c41339] at e_655_15
  have hr := e_655_15
  simp only [e_655_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 655 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3883)) := by
  have c41340 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41341 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41342 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41343 := (publicBatchWrapper89_copies1291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41344 := (publicBatchWrapper89_copies1292 a h).1
  have c41345 := (publicBatchWrapper89_copies1292 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_655_16 := arithEq_of_rows h (row := 655) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_16
  simp only [← c41340, ← c41341, ← c41342] at e_655_16
  have e_655_17 := arithEq_of_rows h (row := 655) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_17
  simp only [← c41343, ← c41344, k2, ← c41345] at e_655_17
  have hr := e_655_17
  simp only [e_655_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 655 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3884)) := by
  have c41346 := (publicBatchWrapper89_copies1292 a h).2.2.1
  have c41347 := (publicBatchWrapper89_copies1292 a h).2.2.2.1
  have c41348 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.1
  have c41349 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.1
  have c41350 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.1
  have c41351 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_655_18 := arithEq_of_rows h (row := 655) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_18
  simp only [← c41346, ← c41347, ← c41348] at e_655_18
  have e_655_19 := arithEq_of_rows h (row := 655) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_655_19
  simp only [← c41349, ← c41350, k2, ← c41351] at e_655_19
  have hr := e_655_19
  simp only [e_655_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 656 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3885)) := by
  have c41352 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.1
  have c41353 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.1
  have c41354 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41355 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41356 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41357 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_656_0 := arithEq_of_rows h (row := 656) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_0
  simp only [← c41352, ← c41353, ← c41354] at e_656_0
  have e_656_1 := arithEq_of_rows h (row := 656) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_1
  simp only [← c41355, ← c41356, k2, ← c41357] at e_656_1
  have hr := e_656_1
  simp only [e_656_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 656 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3886)) := by
  have c41358 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41359 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41360 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41361 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41362 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41363 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_656_2 := arithEq_of_rows h (row := 656) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_2
  simp only [← c41358, ← c41359, ← c41360] at e_656_2
  have e_656_3 := arithEq_of_rows h (row := 656) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_3
  simp only [← c41361, ← c41362, k2, ← c41363] at e_656_3
  have hr := e_656_3
  simp only [e_656_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 656 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3887)) := by
  have c41364 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41365 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41366 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41367 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41368 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41369 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_656_4 := arithEq_of_rows h (row := 656) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_4
  simp only [← c41364, ← c41365, ← c41366] at e_656_4
  have e_656_5 := arithEq_of_rows h (row := 656) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_5
  simp only [← c41367, ← c41368, k2, ← c41369] at e_656_5
  have hr := e_656_5
  simp only [e_656_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 656 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3888)) := by
  have c41370 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41371 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41372 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41373 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41374 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41375 := (publicBatchWrapper89_copies1292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_656_6 := arithEq_of_rows h (row := 656) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_6
  simp only [← c41370, ← c41371, ← c41372] at e_656_6
  have e_656_7 := arithEq_of_rows h (row := 656) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_7
  simp only [← c41373, ← c41374, k2, ← c41375] at e_656_7
  have hr := e_656_7
  simp only [e_656_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 656 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3889)) := by
  have c41376 := (publicBatchWrapper89_copies1293 a h).1
  have c41377 := (publicBatchWrapper89_copies1293 a h).2.1
  have c41378 := (publicBatchWrapper89_copies1293 a h).2.2.1
  have c41379 := (publicBatchWrapper89_copies1293 a h).2.2.2.1
  have c41380 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.1
  have c41381 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_656_8 := arithEq_of_rows h (row := 656) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_8
  simp only [← c41376, ← c41377, ← c41378] at e_656_8
  have e_656_9 := arithEq_of_rows h (row := 656) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_9
  simp only [← c41379, ← c41380, k2, ← c41381] at e_656_9
  have hr := e_656_9
  simp only [e_656_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 656 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3890)) := by
  have c41382 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.1
  have c41383 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.1
  have c41384 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.1
  have c41385 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.1
  have c41386 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41387 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_656_10 := arithEq_of_rows h (row := 656) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_10
  simp only [← c41382, ← c41383, ← c41384] at e_656_10
  have e_656_11 := arithEq_of_rows h (row := 656) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_11
  simp only [← c41385, ← c41386, k2, ← c41387] at e_656_11
  have hr := e_656_11
  simp only [e_656_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 656 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3891)) := by
  have c41388 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41389 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41390 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41391 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41392 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41393 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_656_12 := arithEq_of_rows h (row := 656) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_12
  simp only [← c41388, ← c41389, ← c41390] at e_656_12
  have e_656_13 := arithEq_of_rows h (row := 656) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_13
  simp only [← c41391, ← c41392, k2, ← c41393] at e_656_13
  have hr := e_656_13
  simp only [e_656_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 656 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3892)) := by
  have c41394 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41395 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41396 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41397 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41398 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41399 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_656_14 := arithEq_of_rows h (row := 656) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_14
  simp only [← c41394, ← c41395, ← c41396] at e_656_14
  have e_656_15 := arithEq_of_rows h (row := 656) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_15
  simp only [← c41397, ← c41398, k2, ← c41399] at e_656_15
  have hr := e_656_15
  simp only [e_656_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 656 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3893)) := by
  have c41400 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41401 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41402 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41403 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41404 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41405 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_656_16 := arithEq_of_rows h (row := 656) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_16
  simp only [← c41400, ← c41401, ← c41402] at e_656_16
  have e_656_17 := arithEq_of_rows h (row := 656) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_17
  simp only [← c41403, ← c41404, k2, ← c41405] at e_656_17
  have hr := e_656_17
  simp only [e_656_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 656 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3894)) := by
  have c41406 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41407 := (publicBatchWrapper89_copies1293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41408 := (publicBatchWrapper89_copies1294 a h).1
  have c41409 := (publicBatchWrapper89_copies1294 a h).2.1
  have c41410 := (publicBatchWrapper89_copies1294 a h).2.2.1
  have c41411 := (publicBatchWrapper89_copies1294 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_656_18 := arithEq_of_rows h (row := 656) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_18
  simp only [← c41406, ← c41407, ← c41408] at e_656_18
  have e_656_19 := arithEq_of_rows h (row := 656) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_656_19
  simp only [← c41409, ← c41410, k2, ← c41411] at e_656_19
  have hr := e_656_19
  simp only [e_656_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 657 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3895)) := by
  have c41412 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.1
  have c41413 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.1
  have c41414 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.1
  have c41415 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.1
  have c41416 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.1
  have c41417 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_657_0 := arithEq_of_rows h (row := 657) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_0
  simp only [← c41412, ← c41413, ← c41414] at e_657_0
  have e_657_1 := arithEq_of_rows h (row := 657) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_1
  simp only [← c41415, ← c41416, k2, ← c41417] at e_657_1
  have hr := e_657_1
  simp only [e_657_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 657 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3896)) := by
  have c41418 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41419 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41420 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41421 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41422 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41423 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_657_2 := arithEq_of_rows h (row := 657) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_2
  simp only [← c41418, ← c41419, ← c41420] at e_657_2
  have e_657_3 := arithEq_of_rows h (row := 657) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_3
  simp only [← c41421, ← c41422, k2, ← c41423] at e_657_3
  have hr := e_657_3
  simp only [e_657_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 657 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3897)) := by
  have c41424 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41425 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41426 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41427 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41428 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41429 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_657_4 := arithEq_of_rows h (row := 657) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_4
  simp only [← c41424, ← c41425, ← c41426] at e_657_4
  have e_657_5 := arithEq_of_rows h (row := 657) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_5
  simp only [← c41427, ← c41428, k2, ← c41429] at e_657_5
  have hr := e_657_5
  simp only [e_657_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 657 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3898)) := by
  have c41430 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41431 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41432 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41433 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41434 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41435 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_657_6 := arithEq_of_rows h (row := 657) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_6
  simp only [← c41430, ← c41431, ← c41432] at e_657_6
  have e_657_7 := arithEq_of_rows h (row := 657) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_7
  simp only [← c41433, ← c41434, k2, ← c41435] at e_657_7
  have hr := e_657_7
  simp only [e_657_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 657 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3899)) := by
  have c41436 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41437 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41438 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41439 := (publicBatchWrapper89_copies1294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41440 := (publicBatchWrapper89_copies1295 a h).1
  have c41441 := (publicBatchWrapper89_copies1295 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_657_8 := arithEq_of_rows h (row := 657) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_8
  simp only [← c41436, ← c41437, ← c41438] at e_657_8
  have e_657_9 := arithEq_of_rows h (row := 657) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_9
  simp only [← c41439, ← c41440, k2, ← c41441] at e_657_9
  have hr := e_657_9
  simp only [e_657_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 657 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3900)) := by
  have c41442 := (publicBatchWrapper89_copies1295 a h).2.2.1
  have c41443 := (publicBatchWrapper89_copies1295 a h).2.2.2.1
  have c41444 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.1
  have c41445 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.1
  have c41446 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.1
  have c41447 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_657_10 := arithEq_of_rows h (row := 657) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_10
  simp only [← c41442, ← c41443, ← c41444] at e_657_10
  have e_657_11 := arithEq_of_rows h (row := 657) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_11
  simp only [← c41445, ← c41446, k2, ← c41447] at e_657_11
  have hr := e_657_11
  simp only [e_657_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 657 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3901)) := by
  have c41448 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.1
  have c41449 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.1
  have c41450 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41451 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41452 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41453 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_657_12 := arithEq_of_rows h (row := 657) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_12
  simp only [← c41448, ← c41449, ← c41450] at e_657_12
  have e_657_13 := arithEq_of_rows h (row := 657) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_13
  simp only [← c41451, ← c41452, k2, ← c41453] at e_657_13
  have hr := e_657_13
  simp only [e_657_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 657 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3902)) := by
  have c41454 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41455 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41456 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41457 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41458 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41459 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_657_14 := arithEq_of_rows h (row := 657) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_14
  simp only [← c41454, ← c41455, ← c41456] at e_657_14
  have e_657_15 := arithEq_of_rows h (row := 657) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_15
  simp only [← c41457, ← c41458, k2, ← c41459] at e_657_15
  have hr := e_657_15
  simp only [e_657_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 657 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3903)) := by
  have c41460 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41461 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41462 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41463 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41464 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41465 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_657_16 := arithEq_of_rows h (row := 657) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_16
  simp only [← c41460, ← c41461, ← c41462] at e_657_16
  have e_657_17 := arithEq_of_rows h (row := 657) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_17
  simp only [← c41463, ← c41464, k2, ← c41465] at e_657_17
  have hr := e_657_17
  simp only [e_657_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 657 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3904)) := by
  have c41466 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41467 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41468 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41469 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41470 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41471 := (publicBatchWrapper89_copies1295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_657_18 := arithEq_of_rows h (row := 657) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_18
  simp only [← c41466, ← c41467, ← c41468] at e_657_18
  have e_657_19 := arithEq_of_rows h (row := 657) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_657_19
  simp only [← c41469, ← c41470, k2, ← c41471] at e_657_19
  have hr := e_657_19
  simp only [e_657_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 658 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3905)) := by
  have c41472 := (publicBatchWrapper89_copies1296 a h).1
  have c41473 := (publicBatchWrapper89_copies1296 a h).2.1
  have c41474 := (publicBatchWrapper89_copies1296 a h).2.2.1
  have c41475 := (publicBatchWrapper89_copies1296 a h).2.2.2.1
  have c41476 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.1
  have c41477 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_658_0 := arithEq_of_rows h (row := 658) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_0
  simp only [← c41472, ← c41473, ← c41474] at e_658_0
  have e_658_1 := arithEq_of_rows h (row := 658) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_1
  simp only [← c41475, ← c41476, k2, ← c41477] at e_658_1
  have hr := e_658_1
  simp only [e_658_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 658 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3906)) := by
  have c41478 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.1
  have c41479 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.1
  have c41480 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.1
  have c41481 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.1
  have c41482 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41483 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_658_2 := arithEq_of_rows h (row := 658) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_2
  simp only [← c41478, ← c41479, ← c41480] at e_658_2
  have e_658_3 := arithEq_of_rows h (row := 658) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_3
  simp only [← c41481, ← c41482, k2, ← c41483] at e_658_3
  have hr := e_658_3
  simp only [e_658_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 658 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3907)) := by
  have c41484 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41485 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41486 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41487 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41488 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41489 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_658_4 := arithEq_of_rows h (row := 658) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_4
  simp only [← c41484, ← c41485, ← c41486] at e_658_4
  have e_658_5 := arithEq_of_rows h (row := 658) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_5
  simp only [← c41487, ← c41488, k2, ← c41489] at e_658_5
  have hr := e_658_5
  simp only [e_658_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 658 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3908)) := by
  have c41490 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41491 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41492 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41493 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41494 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41495 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_658_6 := arithEq_of_rows h (row := 658) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_6
  simp only [← c41490, ← c41491, ← c41492] at e_658_6
  have e_658_7 := arithEq_of_rows h (row := 658) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_7
  simp only [← c41493, ← c41494, k2, ← c41495] at e_658_7
  have hr := e_658_7
  simp only [e_658_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 658 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3909)) := by
  have c41496 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41497 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41498 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41499 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41500 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41501 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_658_8 := arithEq_of_rows h (row := 658) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_8
  simp only [← c41496, ← c41497, ← c41498] at e_658_8
  have e_658_9 := arithEq_of_rows h (row := 658) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_9
  simp only [← c41499, ← c41500, k2, ← c41501] at e_658_9
  have hr := e_658_9
  simp only [e_658_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 658 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3910)) := by
  have c41502 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41503 := (publicBatchWrapper89_copies1296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41504 := (publicBatchWrapper89_copies1297 a h).1
  have c41505 := (publicBatchWrapper89_copies1297 a h).2.1
  have c41506 := (publicBatchWrapper89_copies1297 a h).2.2.1
  have c41507 := (publicBatchWrapper89_copies1297 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_658_10 := arithEq_of_rows h (row := 658) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_10
  simp only [← c41502, ← c41503, ← c41504] at e_658_10
  have e_658_11 := arithEq_of_rows h (row := 658) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_11
  simp only [← c41505, ← c41506, k2, ← c41507] at e_658_11
  have hr := e_658_11
  simp only [e_658_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 658 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3911)) := by
  have c41508 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.1
  have c41509 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.1
  have c41510 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.1
  have c41511 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.1
  have c41512 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.1
  have c41513 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_658_12 := arithEq_of_rows h (row := 658) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_12
  simp only [← c41508, ← c41509, ← c41510] at e_658_12
  have e_658_13 := arithEq_of_rows h (row := 658) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_13
  simp only [← c41511, ← c41512, k2, ← c41513] at e_658_13
  have hr := e_658_13
  simp only [e_658_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 658 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3912)) := by
  have c41514 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41515 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41516 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41517 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41518 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41519 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_658_14 := arithEq_of_rows h (row := 658) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_14
  simp only [← c41514, ← c41515, ← c41516] at e_658_14
  have e_658_15 := arithEq_of_rows h (row := 658) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_15
  simp only [← c41517, ← c41518, k2, ← c41519] at e_658_15
  have hr := e_658_15
  simp only [e_658_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 658 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3913)) := by
  have c41520 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41521 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41522 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41523 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41524 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41525 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_658_16 := arithEq_of_rows h (row := 658) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_16
  simp only [← c41520, ← c41521, ← c41522] at e_658_16
  have e_658_17 := arithEq_of_rows h (row := 658) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_17
  simp only [← c41523, ← c41524, k2, ← c41525] at e_658_17
  have hr := e_658_17
  simp only [e_658_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 658 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3914)) := by
  have c41526 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41527 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41528 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41529 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41530 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41531 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_658_18 := arithEq_of_rows h (row := 658) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_18
  simp only [← c41526, ← c41527, ← c41528] at e_658_18
  have e_658_19 := arithEq_of_rows h (row := 658) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_658_19
  simp only [← c41529, ← c41530, k2, ← c41531] at e_658_19
  have hr := e_658_19
  simp only [e_658_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 659 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3915)) := by
  have c41532 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41533 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41534 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41535 := (publicBatchWrapper89_copies1297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41536 := (publicBatchWrapper89_copies1298 a h).1
  have c41537 := (publicBatchWrapper89_copies1298 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_659_0 := arithEq_of_rows h (row := 659) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_0
  simp only [← c41532, ← c41533, ← c41534] at e_659_0
  have e_659_1 := arithEq_of_rows h (row := 659) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_1
  simp only [← c41535, ← c41536, k2, ← c41537] at e_659_1
  have hr := e_659_1
  simp only [e_659_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 659 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3916)) := by
  have c41538 := (publicBatchWrapper89_copies1298 a h).2.2.1
  have c41539 := (publicBatchWrapper89_copies1298 a h).2.2.2.1
  have c41540 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.1
  have c41541 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.1
  have c41542 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.1
  have c41543 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_659_2 := arithEq_of_rows h (row := 659) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_2
  simp only [← c41538, ← c41539, ← c41540] at e_659_2
  have e_659_3 := arithEq_of_rows h (row := 659) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_3
  simp only [← c41541, ← c41542, k2, ← c41543] at e_659_3
  have hr := e_659_3
  simp only [e_659_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 659 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3917)) := by
  have c41544 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.1
  have c41545 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.1
  have c41546 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41547 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41548 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41549 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_659_4 := arithEq_of_rows h (row := 659) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_4
  simp only [← c41544, ← c41545, ← c41546] at e_659_4
  have e_659_5 := arithEq_of_rows h (row := 659) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_5
  simp only [← c41547, ← c41548, k2, ← c41549] at e_659_5
  have hr := e_659_5
  simp only [e_659_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 659 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3918)) := by
  have c41550 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41551 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41552 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41553 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41554 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41555 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_659_6 := arithEq_of_rows h (row := 659) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_6
  simp only [← c41550, ← c41551, ← c41552] at e_659_6
  have e_659_7 := arithEq_of_rows h (row := 659) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_7
  simp only [← c41553, ← c41554, k2, ← c41555] at e_659_7
  have hr := e_659_7
  simp only [e_659_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 659 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3919)) := by
  have c41556 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41557 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41558 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41559 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41560 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41561 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_659_8 := arithEq_of_rows h (row := 659) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_8
  simp only [← c41556, ← c41557, ← c41558] at e_659_8
  have e_659_9 := arithEq_of_rows h (row := 659) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_9
  simp only [← c41559, ← c41560, k2, ← c41561] at e_659_9
  have hr := e_659_9
  simp only [e_659_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 659 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3920)) := by
  have c41562 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41563 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41564 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41565 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41566 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41567 := (publicBatchWrapper89_copies1298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_659_10 := arithEq_of_rows h (row := 659) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_10
  simp only [← c41562, ← c41563, ← c41564] at e_659_10
  have e_659_11 := arithEq_of_rows h (row := 659) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_11
  simp only [← c41565, ← c41566, k2, ← c41567] at e_659_11
  have hr := e_659_11
  simp only [e_659_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 659 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3921)) := by
  have c41568 := (publicBatchWrapper89_copies1299 a h).1
  have c41569 := (publicBatchWrapper89_copies1299 a h).2.1
  have c41570 := (publicBatchWrapper89_copies1299 a h).2.2.1
  have c41571 := (publicBatchWrapper89_copies1299 a h).2.2.2.1
  have c41572 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.1
  have c41573 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_659_12 := arithEq_of_rows h (row := 659) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_12
  simp only [← c41568, ← c41569, ← c41570] at e_659_12
  have e_659_13 := arithEq_of_rows h (row := 659) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_13
  simp only [← c41571, ← c41572, k2, ← c41573] at e_659_13
  have hr := e_659_13
  simp only [e_659_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 659 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3922)) := by
  have c41574 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.1
  have c41575 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.1
  have c41576 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.1
  have c41577 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.1
  have c41578 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41579 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_659_14 := arithEq_of_rows h (row := 659) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_14
  simp only [← c41574, ← c41575, ← c41576] at e_659_14
  have e_659_15 := arithEq_of_rows h (row := 659) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_15
  simp only [← c41577, ← c41578, k2, ← c41579] at e_659_15
  have hr := e_659_15
  simp only [e_659_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 659 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3923)) := by
  have c41580 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41581 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41582 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41583 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41584 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41585 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_659_16 := arithEq_of_rows h (row := 659) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_16
  simp only [← c41580, ← c41581, ← c41582] at e_659_16
  have e_659_17 := arithEq_of_rows h (row := 659) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_17
  simp only [← c41583, ← c41584, k2, ← c41585] at e_659_17
  have hr := e_659_17
  simp only [e_659_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 659 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3924)) := by
  have c41586 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41587 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41588 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41589 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41590 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41591 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_659_18 := arithEq_of_rows h (row := 659) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_18
  simp only [← c41586, ← c41587, ← c41588] at e_659_18
  have e_659_19 := arithEq_of_rows h (row := 659) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_659_19
  simp only [← c41589, ← c41590, k2, ← c41591] at e_659_19
  have hr := e_659_19
  simp only [e_659_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 660 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3925)) := by
  have c41592 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41593 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41594 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41595 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41596 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41597 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_660_0 := arithEq_of_rows h (row := 660) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_0
  simp only [← c41592, ← c41593, ← c41594] at e_660_0
  have e_660_1 := arithEq_of_rows h (row := 660) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_1
  simp only [← c41595, ← c41596, k2, ← c41597] at e_660_1
  have hr := e_660_1
  simp only [e_660_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 660 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3926)) := by
  have c41598 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41599 := (publicBatchWrapper89_copies1299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41600 := (publicBatchWrapper89_copies1300 a h).1
  have c41601 := (publicBatchWrapper89_copies1300 a h).2.1
  have c41602 := (publicBatchWrapper89_copies1300 a h).2.2.1
  have c41603 := (publicBatchWrapper89_copies1300 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_660_2 := arithEq_of_rows h (row := 660) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_2
  simp only [← c41598, ← c41599, ← c41600] at e_660_2
  have e_660_3 := arithEq_of_rows h (row := 660) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_3
  simp only [← c41601, ← c41602, k2, ← c41603] at e_660_3
  have hr := e_660_3
  simp only [e_660_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 660 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3927)) := by
  have c41604 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.1
  have c41605 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.1
  have c41606 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.1
  have c41607 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.1
  have c41608 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.1
  have c41609 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_660_4 := arithEq_of_rows h (row := 660) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_4
  simp only [← c41604, ← c41605, ← c41606] at e_660_4
  have e_660_5 := arithEq_of_rows h (row := 660) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_5
  simp only [← c41607, ← c41608, k2, ← c41609] at e_660_5
  have hr := e_660_5
  simp only [e_660_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 660 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3928)) := by
  have c41610 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41611 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41612 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41613 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41614 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41615 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_660_6 := arithEq_of_rows h (row := 660) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_6
  simp only [← c41610, ← c41611, ← c41612] at e_660_6
  have e_660_7 := arithEq_of_rows h (row := 660) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_7
  simp only [← c41613, ← c41614, k2, ← c41615] at e_660_7
  have hr := e_660_7
  simp only [e_660_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 660 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3929)) := by
  have c41616 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41617 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41618 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41619 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41620 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41621 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_660_8 := arithEq_of_rows h (row := 660) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_8
  simp only [← c41616, ← c41617, ← c41618] at e_660_8
  have e_660_9 := arithEq_of_rows h (row := 660) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_9
  simp only [← c41619, ← c41620, k2, ← c41621] at e_660_9
  have hr := e_660_9
  simp only [e_660_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 660 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3930)) := by
  have c41622 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41623 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41624 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41625 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41626 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41627 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_660_10 := arithEq_of_rows h (row := 660) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_10
  simp only [← c41622, ← c41623, ← c41624] at e_660_10
  have e_660_11 := arithEq_of_rows h (row := 660) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_11
  simp only [← c41625, ← c41626, k2, ← c41627] at e_660_11
  have hr := e_660_11
  simp only [e_660_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 660 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3931)) := by
  have c41628 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41629 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41630 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41631 := (publicBatchWrapper89_copies1300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41632 := (publicBatchWrapper89_copies1301 a h).1
  have c41633 := (publicBatchWrapper89_copies1301 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_660_12 := arithEq_of_rows h (row := 660) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_12
  simp only [← c41628, ← c41629, ← c41630] at e_660_12
  have e_660_13 := arithEq_of_rows h (row := 660) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_13
  simp only [← c41631, ← c41632, k2, ← c41633] at e_660_13
  have hr := e_660_13
  simp only [e_660_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 660 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3932)) := by
  have c41634 := (publicBatchWrapper89_copies1301 a h).2.2.1
  have c41635 := (publicBatchWrapper89_copies1301 a h).2.2.2.1
  have c41636 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.1
  have c41637 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.1
  have c41638 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.1
  have c41639 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_660_14 := arithEq_of_rows h (row := 660) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_14
  simp only [← c41634, ← c41635, ← c41636] at e_660_14
  have e_660_15 := arithEq_of_rows h (row := 660) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_15
  simp only [← c41637, ← c41638, k2, ← c41639] at e_660_15
  have hr := e_660_15
  simp only [e_660_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 660 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3933)) := by
  have c41640 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.1
  have c41641 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.1
  have c41642 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41643 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41644 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41645 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_660_16 := arithEq_of_rows h (row := 660) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_16
  simp only [← c41640, ← c41641, ← c41642] at e_660_16
  have e_660_17 := arithEq_of_rows h (row := 660) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_17
  simp only [← c41643, ← c41644, k2, ← c41645] at e_660_17
  have hr := e_660_17
  simp only [e_660_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 660 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3934)) := by
  have c41646 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41647 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41648 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41649 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41650 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41651 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_660_18 := arithEq_of_rows h (row := 660) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_18
  simp only [← c41646, ← c41647, ← c41648] at e_660_18
  have e_660_19 := arithEq_of_rows h (row := 660) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_660_19
  simp only [← c41649, ← c41650, k2, ← c41651] at e_660_19
  have hr := e_660_19
  simp only [e_660_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 661 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3935)) := by
  have c41652 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41653 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41654 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41655 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41656 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41657 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_661_0 := arithEq_of_rows h (row := 661) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_0
  simp only [← c41652, ← c41653, ← c41654] at e_661_0
  have e_661_1 := arithEq_of_rows h (row := 661) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_1
  simp only [← c41655, ← c41656, k2, ← c41657] at e_661_1
  have hr := e_661_1
  simp only [e_661_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 661 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3936)) := by
  have c41658 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41659 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41660 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41661 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41662 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41663 := (publicBatchWrapper89_copies1301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_661_2 := arithEq_of_rows h (row := 661) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_2
  simp only [← c41658, ← c41659, ← c41660] at e_661_2
  have e_661_3 := arithEq_of_rows h (row := 661) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_3
  simp only [← c41661, ← c41662, k2, ← c41663] at e_661_3
  have hr := e_661_3
  simp only [e_661_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 661 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3937)) := by
  have c41664 := (publicBatchWrapper89_copies1302 a h).1
  have c41665 := (publicBatchWrapper89_copies1302 a h).2.1
  have c41666 := (publicBatchWrapper89_copies1302 a h).2.2.1
  have c41667 := (publicBatchWrapper89_copies1302 a h).2.2.2.1
  have c41668 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.1
  have c41669 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_661_4 := arithEq_of_rows h (row := 661) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_4
  simp only [← c41664, ← c41665, ← c41666] at e_661_4
  have e_661_5 := arithEq_of_rows h (row := 661) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_5
  simp only [← c41667, ← c41668, k2, ← c41669] at e_661_5
  have hr := e_661_5
  simp only [e_661_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 661 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3938)) := by
  have c41670 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.1
  have c41671 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.1
  have c41672 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.1
  have c41673 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.1
  have c41674 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41675 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_661_6 := arithEq_of_rows h (row := 661) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_6
  simp only [← c41670, ← c41671, ← c41672] at e_661_6
  have e_661_7 := arithEq_of_rows h (row := 661) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_7
  simp only [← c41673, ← c41674, k2, ← c41675] at e_661_7
  have hr := e_661_7
  simp only [e_661_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 661 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3939)) := by
  have c41676 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41677 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41678 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41679 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41680 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41681 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_661_8 := arithEq_of_rows h (row := 661) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_8
  simp only [← c41676, ← c41677, ← c41678] at e_661_8
  have e_661_9 := arithEq_of_rows h (row := 661) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_9
  simp only [← c41679, ← c41680, k2, ← c41681] at e_661_9
  have hr := e_661_9
  simp only [e_661_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 661 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3940)) := by
  have c41682 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41683 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41684 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41685 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41686 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41687 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_661_10 := arithEq_of_rows h (row := 661) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_10
  simp only [← c41682, ← c41683, ← c41684] at e_661_10
  have e_661_11 := arithEq_of_rows h (row := 661) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_11
  simp only [← c41685, ← c41686, k2, ← c41687] at e_661_11
  have hr := e_661_11
  simp only [e_661_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 661 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3941)) := by
  have c41688 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41689 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41690 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41691 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41692 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41693 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_661_12 := arithEq_of_rows h (row := 661) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_12
  simp only [← c41688, ← c41689, ← c41690] at e_661_12
  have e_661_13 := arithEq_of_rows h (row := 661) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_13
  simp only [← c41691, ← c41692, k2, ← c41693] at e_661_13
  have hr := e_661_13
  simp only [e_661_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 661 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3942)) := by
  have c41694 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41695 := (publicBatchWrapper89_copies1302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41696 := (publicBatchWrapper89_copies1303 a h).1
  have c41697 := (publicBatchWrapper89_copies1303 a h).2.1
  have c41698 := (publicBatchWrapper89_copies1303 a h).2.2.1
  have c41699 := (publicBatchWrapper89_copies1303 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_661_14 := arithEq_of_rows h (row := 661) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_14
  simp only [← c41694, ← c41695, ← c41696] at e_661_14
  have e_661_15 := arithEq_of_rows h (row := 661) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_15
  simp only [← c41697, ← c41698, k2, ← c41699] at e_661_15
  have hr := e_661_15
  simp only [e_661_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 661 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3943)) := by
  have c41700 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.1
  have c41701 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.1
  have c41702 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.1
  have c41703 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.1
  have c41704 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.1
  have c41705 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_661_16 := arithEq_of_rows h (row := 661) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_16
  simp only [← c41700, ← c41701, ← c41702] at e_661_16
  have e_661_17 := arithEq_of_rows h (row := 661) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_17
  simp only [← c41703, ← c41704, k2, ← c41705] at e_661_17
  have hr := e_661_17
  simp only [e_661_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 661 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3944)) := by
  have c41706 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41707 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41708 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41709 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41710 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41711 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_661_18 := arithEq_of_rows h (row := 661) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_18
  simp only [← c41706, ← c41707, ← c41708] at e_661_18
  have e_661_19 := arithEq_of_rows h (row := 661) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_661_19
  simp only [← c41709, ← c41710, k2, ← c41711] at e_661_19
  have hr := e_661_19
  simp only [e_661_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 662 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3945)) := by
  have c41712 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41713 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41714 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41715 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41716 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41717 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_662_0 := arithEq_of_rows h (row := 662) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_0
  simp only [← c41712, ← c41713, ← c41714] at e_662_0
  have e_662_1 := arithEq_of_rows h (row := 662) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_1
  simp only [← c41715, ← c41716, k2, ← c41717] at e_662_1
  have hr := e_662_1
  simp only [e_662_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 662 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3946)) := by
  have c41718 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41719 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41720 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41721 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41722 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41723 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_662_2 := arithEq_of_rows h (row := 662) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_2
  simp only [← c41718, ← c41719, ← c41720] at e_662_2
  have e_662_3 := arithEq_of_rows h (row := 662) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_3
  simp only [← c41721, ← c41722, k2, ← c41723] at e_662_3
  have hr := e_662_3
  simp only [e_662_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 662 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3947)) := by
  have c41724 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41725 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41726 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41727 := (publicBatchWrapper89_copies1303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41728 := (publicBatchWrapper89_copies1304 a h).1
  have c41729 := (publicBatchWrapper89_copies1304 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_662_4 := arithEq_of_rows h (row := 662) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_4
  simp only [← c41724, ← c41725, ← c41726] at e_662_4
  have e_662_5 := arithEq_of_rows h (row := 662) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_5
  simp only [← c41727, ← c41728, k2, ← c41729] at e_662_5
  have hr := e_662_5
  simp only [e_662_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 662 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3948)) := by
  have c41730 := (publicBatchWrapper89_copies1304 a h).2.2.1
  have c41731 := (publicBatchWrapper89_copies1304 a h).2.2.2.1
  have c41732 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.1
  have c41733 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.1
  have c41734 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.1
  have c41735 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_662_6 := arithEq_of_rows h (row := 662) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_6
  simp only [← c41730, ← c41731, ← c41732] at e_662_6
  have e_662_7 := arithEq_of_rows h (row := 662) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_7
  simp only [← c41733, ← c41734, k2, ← c41735] at e_662_7
  have hr := e_662_7
  simp only [e_662_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 662 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3949)) := by
  have c41736 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.1
  have c41737 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.1
  have c41738 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41739 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41740 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41741 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_662_8 := arithEq_of_rows h (row := 662) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_8
  simp only [← c41736, ← c41737, ← c41738] at e_662_8
  have e_662_9 := arithEq_of_rows h (row := 662) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_9
  simp only [← c41739, ← c41740, k2, ← c41741] at e_662_9
  have hr := e_662_9
  simp only [e_662_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 662 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3950)) := by
  have c41742 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41743 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41744 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41745 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41746 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41747 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_662_10 := arithEq_of_rows h (row := 662) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_10
  simp only [← c41742, ← c41743, ← c41744] at e_662_10
  have e_662_11 := arithEq_of_rows h (row := 662) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_11
  simp only [← c41745, ← c41746, k2, ← c41747] at e_662_11
  have hr := e_662_11
  simp only [e_662_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 662 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3951)) := by
  have c41748 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41749 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41750 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41751 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41752 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41753 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_662_12 := arithEq_of_rows h (row := 662) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_12
  simp only [← c41748, ← c41749, ← c41750] at e_662_12
  have e_662_13 := arithEq_of_rows h (row := 662) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_13
  simp only [← c41751, ← c41752, k2, ← c41753] at e_662_13
  have hr := e_662_13
  simp only [e_662_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 662 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3952)) := by
  have c41754 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41755 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41756 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41757 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41758 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41759 := (publicBatchWrapper89_copies1304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_662_14 := arithEq_of_rows h (row := 662) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_14
  simp only [← c41754, ← c41755, ← c41756] at e_662_14
  have e_662_15 := arithEq_of_rows h (row := 662) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_15
  simp only [← c41757, ← c41758, k2, ← c41759] at e_662_15
  have hr := e_662_15
  simp only [e_662_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 662 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3953)) := by
  have c41760 := (publicBatchWrapper89_copies1305 a h).1
  have c41761 := (publicBatchWrapper89_copies1305 a h).2.1
  have c41762 := (publicBatchWrapper89_copies1305 a h).2.2.1
  have c41763 := (publicBatchWrapper89_copies1305 a h).2.2.2.1
  have c41764 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.1
  have c41765 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_662_16 := arithEq_of_rows h (row := 662) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_16
  simp only [← c41760, ← c41761, ← c41762] at e_662_16
  have e_662_17 := arithEq_of_rows h (row := 662) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_17
  simp only [← c41763, ← c41764, k2, ← c41765] at e_662_17
  have hr := e_662_17
  simp only [e_662_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 662 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3954)) := by
  have c41766 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.1
  have c41767 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.1
  have c41768 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.1
  have c41769 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.1
  have c41770 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41771 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_662_18 := arithEq_of_rows h (row := 662) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_18
  simp only [← c41766, ← c41767, ← c41768] at e_662_18
  have e_662_19 := arithEq_of_rows h (row := 662) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_662_19
  simp only [← c41769, ← c41770, k2, ← c41771] at e_662_19
  have hr := e_662_19
  simp only [e_662_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 663 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3955)) := by
  have c41772 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41773 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41774 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41775 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41776 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41777 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_663_0 := arithEq_of_rows h (row := 663) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_0
  simp only [← c41772, ← c41773, ← c41774] at e_663_0
  have e_663_1 := arithEq_of_rows h (row := 663) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_1
  simp only [← c41775, ← c41776, k2, ← c41777] at e_663_1
  have hr := e_663_1
  simp only [e_663_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 663 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3956)) := by
  have c41778 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41779 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41780 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41781 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41782 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41783 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_663_2 := arithEq_of_rows h (row := 663) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_2
  simp only [← c41778, ← c41779, ← c41780] at e_663_2
  have e_663_3 := arithEq_of_rows h (row := 663) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_3
  simp only [← c41781, ← c41782, k2, ← c41783] at e_663_3
  have hr := e_663_3
  simp only [e_663_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 663 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3957)) := by
  have c41784 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41785 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41786 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41787 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41788 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41789 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_663_4 := arithEq_of_rows h (row := 663) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_4
  simp only [← c41784, ← c41785, ← c41786] at e_663_4
  have e_663_5 := arithEq_of_rows h (row := 663) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_5
  simp only [← c41787, ← c41788, k2, ← c41789] at e_663_5
  have hr := e_663_5
  simp only [e_663_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 663 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3958)) := by
  have c41790 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41791 := (publicBatchWrapper89_copies1305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41792 := (publicBatchWrapper89_copies1306 a h).1
  have c41793 := (publicBatchWrapper89_copies1306 a h).2.1
  have c41794 := (publicBatchWrapper89_copies1306 a h).2.2.1
  have c41795 := (publicBatchWrapper89_copies1306 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_663_6 := arithEq_of_rows h (row := 663) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_6
  simp only [← c41790, ← c41791, ← c41792] at e_663_6
  have e_663_7 := arithEq_of_rows h (row := 663) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_7
  simp only [← c41793, ← c41794, k2, ← c41795] at e_663_7
  have hr := e_663_7
  simp only [e_663_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 663 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3959)) := by
  have c41796 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.1
  have c41797 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.1
  have c41798 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.1
  have c41799 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.1
  have c41800 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.1
  have c41801 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_663_8 := arithEq_of_rows h (row := 663) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_8
  simp only [← c41796, ← c41797, ← c41798] at e_663_8
  have e_663_9 := arithEq_of_rows h (row := 663) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_9
  simp only [← c41799, ← c41800, k2, ← c41801] at e_663_9
  have hr := e_663_9
  simp only [e_663_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 663 47) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3960)) := by
  have c41802 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41803 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41804 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41805 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41806 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41807 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_663_10 := arithEq_of_rows h (row := 663) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_10
  simp only [← c41802, ← c41803, ← c41804] at e_663_10
  have e_663_11 := arithEq_of_rows h (row := 663) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_11
  simp only [← c41805, ← c41806, k2, ← c41807] at e_663_11
  have hr := e_663_11
  simp only [e_663_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 663 55) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3961)) := by
  have c41808 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41809 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41810 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41811 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41812 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41813 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_663_12 := arithEq_of_rows h (row := 663) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_12
  simp only [← c41808, ← c41809, ← c41810] at e_663_12
  have e_663_13 := arithEq_of_rows h (row := 663) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_13
  simp only [← c41811, ← c41812, k2, ← c41813] at e_663_13
  have hr := e_663_13
  simp only [e_663_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 663 63) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3962)) := by
  have c41814 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41815 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41816 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41817 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41818 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41819 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_663_14 := arithEq_of_rows h (row := 663) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_14
  simp only [← c41814, ← c41815, ← c41816] at e_663_14
  have e_663_15 := arithEq_of_rows h (row := 663) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_15
  simp only [← c41817, ← c41818, k2, ← c41819] at e_663_15
  have hr := e_663_15
  simp only [e_663_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 663 71) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3963)) := by
  have c41820 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41821 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41822 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41823 := (publicBatchWrapper89_copies1306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41824 := (publicBatchWrapper89_copies1307 a h).1
  have c41825 := (publicBatchWrapper89_copies1307 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_663_16 := arithEq_of_rows h (row := 663) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_16
  simp only [← c41820, ← c41821, ← c41822] at e_663_16
  have e_663_17 := arithEq_of_rows h (row := 663) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_17
  simp only [← c41823, ← c41824, k2, ← c41825] at e_663_17
  have hr := e_663_17
  simp only [e_663_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 663 79) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3964)) := by
  have c41826 := (publicBatchWrapper89_copies1307 a h).2.2.1
  have c41827 := (publicBatchWrapper89_copies1307 a h).2.2.2.1
  have c41828 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.1
  have c41829 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.1
  have c41830 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.1
  have c41831 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_663_18 := arithEq_of_rows h (row := 663) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_18
  simp only [← c41826, ← c41827, ← c41828] at e_663_18
  have e_663_19 := arithEq_of_rows h (row := 663) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_663_19
  simp only [← c41829, ← c41830, k2, ← c41831] at e_663_19
  have hr := e_663_19
  simp only [e_663_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 664 7) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3965)) := by
  have c41832 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.1
  have c41833 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.1
  have c41834 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41835 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41836 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41837 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_664_0 := arithEq_of_rows h (row := 664) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_0
  simp only [← c41832, ← c41833, ← c41834] at e_664_0
  have e_664_1 := arithEq_of_rows h (row := 664) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_1
  simp only [← c41835, ← c41836, k2, ← c41837] at e_664_1
  have hr := e_664_1
  simp only [e_664_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 664 15) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3966)) := by
  have c41838 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41839 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41840 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41841 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41842 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41843 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_664_2 := arithEq_of_rows h (row := 664) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_2
  simp only [← c41838, ← c41839, ← c41840] at e_664_2
  have e_664_3 := arithEq_of_rows h (row := 664) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_3
  simp only [← c41841, ← c41842, k2, ← c41843] at e_664_3
  have hr := e_664_3
  simp only [e_664_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 664 23) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3967)) := by
  have c41844 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41845 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41846 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41847 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41848 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41849 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_664_4 := arithEq_of_rows h (row := 664) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_4
  simp only [← c41844, ← c41845, ← c41846] at e_664_4
  have e_664_5 := arithEq_of_rows h (row := 664) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_5
  simp only [← c41847, ← c41848, k2, ← c41849] at e_664_5
  have hr := e_664_5
  simp only [e_664_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 664 31) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3968)) := by
  have c41850 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41851 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41852 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41853 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41854 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41855 := (publicBatchWrapper89_copies1307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_664_6 := arithEq_of_rows h (row := 664) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_6
  simp only [← c41850, ← c41851, ← c41852] at e_664_6
  have e_664_7 := arithEq_of_rows h (row := 664) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_7
  simp only [← c41853, ← c41854, k2, ← c41855] at e_664_7
  have hr := e_664_7
  simp only [e_664_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 664 39) = bselect (a (.wire 20 15)) (a (.virt 200)) (a (.virt 3969)) := by
  have c41856 := (publicBatchWrapper89_copies1308 a h).1
  have c41857 := (publicBatchWrapper89_copies1308 a h).2.1
  have c41858 := (publicBatchWrapper89_copies1308 a h).2.2.1
  have c41859 := (publicBatchWrapper89_copies1308 a h).2.2.2.1
  have c41860 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.1
  have c41861 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_664_8 := arithEq_of_rows h (row := 664) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_8
  simp only [← c41856, ← c41857, ← c41858] at e_664_8
  have e_664_9 := arithEq_of_rows h (row := 664) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_9
  simp only [← c41859, ← c41860, k2, ← c41861] at e_664_9
  have hr := e_664_9
  simp only [e_664_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 664 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4010)) := by
  have c41862 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.1
  have c41863 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.1
  have c41864 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.1
  have c41865 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.1
  have c41866 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41867 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_664_10 := arithEq_of_rows h (row := 664) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_10
  simp only [← c41862, ← c41863, ← c41864] at e_664_10
  have e_664_11 := arithEq_of_rows h (row := 664) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_11
  simp only [← c41865, ← c41866, k2, ← c41867] at e_664_11
  have hr := e_664_11
  simp only [e_664_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 664 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4011)) := by
  have c41868 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41869 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41870 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41871 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41872 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41873 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_664_12 := arithEq_of_rows h (row := 664) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_12
  simp only [← c41868, ← c41869, ← c41870] at e_664_12
  have e_664_13 := arithEq_of_rows h (row := 664) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_13
  simp only [← c41871, ← c41872, k2, ← c41873] at e_664_13
  have hr := e_664_13
  simp only [e_664_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 664 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4012)) := by
  have c41874 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41875 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41876 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41877 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41878 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41879 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_664_14 := arithEq_of_rows h (row := 664) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_14
  simp only [← c41874, ← c41875, ← c41876] at e_664_14
  have e_664_15 := arithEq_of_rows h (row := 664) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_15
  simp only [← c41877, ← c41878, k2, ← c41879] at e_664_15
  have hr := e_664_15
  simp only [e_664_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 664 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4013)) := by
  have c41880 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41881 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41882 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41883 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41884 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41885 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_664_16 := arithEq_of_rows h (row := 664) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_16
  simp only [← c41880, ← c41881, ← c41882] at e_664_16
  have e_664_17 := arithEq_of_rows h (row := 664) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_17
  simp only [← c41883, ← c41884, k2, ← c41885] at e_664_17
  have hr := e_664_17
  simp only [e_664_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 664 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4014)) := by
  have c41886 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41887 := (publicBatchWrapper89_copies1308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41888 := (publicBatchWrapper89_copies1309 a h).1
  have c41889 := (publicBatchWrapper89_copies1309 a h).2.1
  have c41890 := (publicBatchWrapper89_copies1309 a h).2.2.1
  have c41891 := (publicBatchWrapper89_copies1309 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_664_18 := arithEq_of_rows h (row := 664) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_18
  simp only [← c41886, ← c41887, ← c41888] at e_664_18
  have e_664_19 := arithEq_of_rows h (row := 664) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_664_19
  simp only [← c41889, ← c41890, k2, ← c41891] at e_664_19
  have hr := e_664_19
  simp only [e_664_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 665 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4015)) := by
  have c41892 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.1
  have c41893 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.1
  have c41894 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.1
  have c41895 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.1
  have c41896 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.1
  have c41897 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_665_0 := arithEq_of_rows h (row := 665) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_0
  simp only [← c41892, ← c41893, ← c41894] at e_665_0
  have e_665_1 := arithEq_of_rows h (row := 665) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_1
  simp only [← c41895, ← c41896, k2, ← c41897] at e_665_1
  have hr := e_665_1
  simp only [e_665_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 665 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4016)) := by
  have c41898 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41899 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41900 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41901 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41902 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41903 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_665_2 := arithEq_of_rows h (row := 665) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_2
  simp only [← c41898, ← c41899, ← c41900] at e_665_2
  have e_665_3 := arithEq_of_rows h (row := 665) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_3
  simp only [← c41901, ← c41902, k2, ← c41903] at e_665_3
  have hr := e_665_3
  simp only [e_665_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 665 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4017)) := by
  have c41904 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41905 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41906 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41907 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41908 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41909 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_665_4 := arithEq_of_rows h (row := 665) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_4
  simp only [← c41904, ← c41905, ← c41906] at e_665_4
  have e_665_5 := arithEq_of_rows h (row := 665) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_5
  simp only [← c41907, ← c41908, k2, ← c41909] at e_665_5
  have hr := e_665_5
  simp only [e_665_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 665 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4018)) := by
  have c41910 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41911 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41912 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41913 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41914 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41915 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_665_6 := arithEq_of_rows h (row := 665) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_6
  simp only [← c41910, ← c41911, ← c41912] at e_665_6
  have e_665_7 := arithEq_of_rows h (row := 665) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_7
  simp only [← c41913, ← c41914, k2, ← c41915] at e_665_7
  have hr := e_665_7
  simp only [e_665_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 665 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4019)) := by
  have c41916 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41917 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41918 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41919 := (publicBatchWrapper89_copies1309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41920 := (publicBatchWrapper89_copies1310 a h).1
  have c41921 := (publicBatchWrapper89_copies1310 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_665_8 := arithEq_of_rows h (row := 665) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_8
  simp only [← c41916, ← c41917, ← c41918] at e_665_8
  have e_665_9 := arithEq_of_rows h (row := 665) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_9
  simp only [← c41919, ← c41920, k2, ← c41921] at e_665_9
  have hr := e_665_9
  simp only [e_665_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 665 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4020)) := by
  have c41922 := (publicBatchWrapper89_copies1310 a h).2.2.1
  have c41923 := (publicBatchWrapper89_copies1310 a h).2.2.2.1
  have c41924 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.1
  have c41925 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.1
  have c41926 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.1
  have c41927 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_665_10 := arithEq_of_rows h (row := 665) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_10
  simp only [← c41922, ← c41923, ← c41924] at e_665_10
  have e_665_11 := arithEq_of_rows h (row := 665) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_11
  simp only [← c41925, ← c41926, k2, ← c41927] at e_665_11
  have hr := e_665_11
  simp only [e_665_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 665 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4021)) := by
  have c41928 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.1
  have c41929 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.1
  have c41930 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41931 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41932 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41933 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_665_12 := arithEq_of_rows h (row := 665) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_12
  simp only [← c41928, ← c41929, ← c41930] at e_665_12
  have e_665_13 := arithEq_of_rows h (row := 665) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_13
  simp only [← c41931, ← c41932, k2, ← c41933] at e_665_13
  have hr := e_665_13
  simp only [e_665_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 665 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4022)) := by
  have c41934 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41935 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41936 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41937 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41938 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41939 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_665_14 := arithEq_of_rows h (row := 665) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_14
  simp only [← c41934, ← c41935, ← c41936] at e_665_14
  have e_665_15 := arithEq_of_rows h (row := 665) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_15
  simp only [← c41937, ← c41938, k2, ← c41939] at e_665_15
  have hr := e_665_15
  simp only [e_665_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 665 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4023)) := by
  have c41940 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41941 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41942 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41943 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41944 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41945 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_665_16 := arithEq_of_rows h (row := 665) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_16
  simp only [← c41940, ← c41941, ← c41942] at e_665_16
  have e_665_17 := arithEq_of_rows h (row := 665) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_17
  simp only [← c41943, ← c41944, k2, ← c41945] at e_665_17
  have hr := e_665_17
  simp only [e_665_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 665 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4024)) := by
  have c41946 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41947 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41948 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41949 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41950 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41951 := (publicBatchWrapper89_copies1310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_665_18 := arithEq_of_rows h (row := 665) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_18
  simp only [← c41946, ← c41947, ← c41948] at e_665_18
  have e_665_19 := arithEq_of_rows h (row := 665) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_665_19
  simp only [← c41949, ← c41950, k2, ← c41951] at e_665_19
  have hr := e_665_19
  simp only [e_665_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 666 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4025)) := by
  have c41952 := (publicBatchWrapper89_copies1311 a h).1
  have c41953 := (publicBatchWrapper89_copies1311 a h).2.1
  have c41954 := (publicBatchWrapper89_copies1311 a h).2.2.1
  have c41955 := (publicBatchWrapper89_copies1311 a h).2.2.2.1
  have c41956 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.1
  have c41957 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_666_0 := arithEq_of_rows h (row := 666) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_0
  simp only [← c41952, ← c41953, ← c41954] at e_666_0
  have e_666_1 := arithEq_of_rows h (row := 666) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_1
  simp only [← c41955, ← c41956, k2, ← c41957] at e_666_1
  have hr := e_666_1
  simp only [e_666_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 666 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4026)) := by
  have c41958 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.1
  have c41959 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.1
  have c41960 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.1
  have c41961 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.1
  have c41962 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41963 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_666_2 := arithEq_of_rows h (row := 666) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_2
  simp only [← c41958, ← c41959, ← c41960] at e_666_2
  have e_666_3 := arithEq_of_rows h (row := 666) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_3
  simp only [← c41961, ← c41962, k2, ← c41963] at e_666_3
  have hr := e_666_3
  simp only [e_666_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 666 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4027)) := by
  have c41964 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41965 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41966 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41967 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41968 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41969 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_666_4 := arithEq_of_rows h (row := 666) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_4
  simp only [← c41964, ← c41965, ← c41966] at e_666_4
  have e_666_5 := arithEq_of_rows h (row := 666) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_5
  simp only [← c41967, ← c41968, k2, ← c41969] at e_666_5
  have hr := e_666_5
  simp only [e_666_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 666 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4028)) := by
  have c41970 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41971 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41972 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41973 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41974 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41975 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_666_6 := arithEq_of_rows h (row := 666) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_6
  simp only [← c41970, ← c41971, ← c41972] at e_666_6
  have e_666_7 := arithEq_of_rows h (row := 666) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_7
  simp only [← c41973, ← c41974, k2, ← c41975] at e_666_7
  have hr := e_666_7
  simp only [e_666_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 666 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4029)) := by
  have c41976 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41977 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41978 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41979 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41980 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41981 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_666_8 := arithEq_of_rows h (row := 666) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_8
  simp only [← c41976, ← c41977, ← c41978] at e_666_8
  have e_666_9 := arithEq_of_rows h (row := 666) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_9
  simp only [← c41979, ← c41980, k2, ← c41981] at e_666_9
  have hr := e_666_9
  simp only [e_666_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 666 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4030)) := by
  have c41982 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41983 := (publicBatchWrapper89_copies1311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c41984 := (publicBatchWrapper89_copies1312 a h).1
  have c41985 := (publicBatchWrapper89_copies1312 a h).2.1
  have c41986 := (publicBatchWrapper89_copies1312 a h).2.2.1
  have c41987 := (publicBatchWrapper89_copies1312 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_666_10 := arithEq_of_rows h (row := 666) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_10
  simp only [← c41982, ← c41983, ← c41984] at e_666_10
  have e_666_11 := arithEq_of_rows h (row := 666) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_11
  simp only [← c41985, ← c41986, k2, ← c41987] at e_666_11
  have hr := e_666_11
  simp only [e_666_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 666 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4031)) := by
  have c41988 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.1
  have c41989 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.1
  have c41990 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.1
  have c41991 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.1
  have c41992 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.1
  have c41993 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_666_12 := arithEq_of_rows h (row := 666) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_12
  simp only [← c41988, ← c41989, ← c41990] at e_666_12
  have e_666_13 := arithEq_of_rows h (row := 666) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_13
  simp only [← c41991, ← c41992, k2, ← c41993] at e_666_13
  have hr := e_666_13
  simp only [e_666_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 666 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4032)) := by
  have c41994 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.1
  have c41995 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c41996 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41997 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41998 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c41999 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_666_14 := arithEq_of_rows h (row := 666) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_14
  simp only [← c41994, ← c41995, ← c41996] at e_666_14
  have e_666_15 := arithEq_of_rows h (row := 666) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_15
  simp only [← c41997, ← c41998, k2, ← c41999] at e_666_15
  have hr := e_666_15
  simp only [e_666_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 666 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4033)) := by
  have c42000 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42001 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42002 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42003 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42004 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42005 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_666_16 := arithEq_of_rows h (row := 666) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_16
  simp only [← c42000, ← c42001, ← c42002] at e_666_16
  have e_666_17 := arithEq_of_rows h (row := 666) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_17
  simp only [← c42003, ← c42004, k2, ← c42005] at e_666_17
  have hr := e_666_17
  simp only [e_666_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 666 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4034)) := by
  have c42006 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42007 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42008 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42009 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42010 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42011 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_666_18 := arithEq_of_rows h (row := 666) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_18
  simp only [← c42006, ← c42007, ← c42008] at e_666_18
  have e_666_19 := arithEq_of_rows h (row := 666) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_666_19
  simp only [← c42009, ← c42010, k2, ← c42011] at e_666_19
  have hr := e_666_19
  simp only [e_666_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 667 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4035)) := by
  have c42012 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42013 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42014 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42015 := (publicBatchWrapper89_copies1312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42016 := (publicBatchWrapper89_copies1313 a h).1
  have c42017 := (publicBatchWrapper89_copies1313 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_667_0 := arithEq_of_rows h (row := 667) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_0
  simp only [← c42012, ← c42013, ← c42014] at e_667_0
  have e_667_1 := arithEq_of_rows h (row := 667) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_1
  simp only [← c42015, ← c42016, k2, ← c42017] at e_667_1
  have hr := e_667_1
  simp only [e_667_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 667 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4036)) := by
  have c42018 := (publicBatchWrapper89_copies1313 a h).2.2.1
  have c42019 := (publicBatchWrapper89_copies1313 a h).2.2.2.1
  have c42020 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.1
  have c42021 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.1
  have c42022 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.1
  have c42023 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_667_2 := arithEq_of_rows h (row := 667) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_2
  simp only [← c42018, ← c42019, ← c42020] at e_667_2
  have e_667_3 := arithEq_of_rows h (row := 667) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_3
  simp only [← c42021, ← c42022, k2, ← c42023] at e_667_3
  have hr := e_667_3
  simp only [e_667_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 667 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4037)) := by
  have c42024 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.1
  have c42025 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.1
  have c42026 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42027 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42028 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42029 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_667_4 := arithEq_of_rows h (row := 667) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_4
  simp only [← c42024, ← c42025, ← c42026] at e_667_4
  have e_667_5 := arithEq_of_rows h (row := 667) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_5
  simp only [← c42027, ← c42028, k2, ← c42029] at e_667_5
  have hr := e_667_5
  simp only [e_667_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 667 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4038)) := by
  have c42030 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42031 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42032 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42033 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42034 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42035 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_667_6 := arithEq_of_rows h (row := 667) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_6
  simp only [← c42030, ← c42031, ← c42032] at e_667_6
  have e_667_7 := arithEq_of_rows h (row := 667) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_7
  simp only [← c42033, ← c42034, k2, ← c42035] at e_667_7
  have hr := e_667_7
  simp only [e_667_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 667 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4039)) := by
  have c42036 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42037 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42038 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42039 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42040 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42041 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_667_8 := arithEq_of_rows h (row := 667) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_8
  simp only [← c42036, ← c42037, ← c42038] at e_667_8
  have e_667_9 := arithEq_of_rows h (row := 667) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_9
  simp only [← c42039, ← c42040, k2, ← c42041] at e_667_9
  have hr := e_667_9
  simp only [e_667_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 667 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4040)) := by
  have c42042 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42043 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42044 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42045 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42046 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42047 := (publicBatchWrapper89_copies1313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_667_10 := arithEq_of_rows h (row := 667) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_10
  simp only [← c42042, ← c42043, ← c42044] at e_667_10
  have e_667_11 := arithEq_of_rows h (row := 667) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_11
  simp only [← c42045, ← c42046, k2, ← c42047] at e_667_11
  have hr := e_667_11
  simp only [e_667_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 667 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4041)) := by
  have c42048 := (publicBatchWrapper89_copies1314 a h).1
  have c42049 := (publicBatchWrapper89_copies1314 a h).2.1
  have c42050 := (publicBatchWrapper89_copies1314 a h).2.2.1
  have c42051 := (publicBatchWrapper89_copies1314 a h).2.2.2.1
  have c42052 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.1
  have c42053 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_667_12 := arithEq_of_rows h (row := 667) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_12
  simp only [← c42048, ← c42049, ← c42050] at e_667_12
  have e_667_13 := arithEq_of_rows h (row := 667) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_13
  simp only [← c42051, ← c42052, k2, ← c42053] at e_667_13
  have hr := e_667_13
  simp only [e_667_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 667 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4042)) := by
  have c42054 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.1
  have c42055 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.1
  have c42056 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.1
  have c42057 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.1
  have c42058 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42059 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_667_14 := arithEq_of_rows h (row := 667) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_14
  simp only [← c42054, ← c42055, ← c42056] at e_667_14
  have e_667_15 := arithEq_of_rows h (row := 667) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_15
  simp only [← c42057, ← c42058, k2, ← c42059] at e_667_15
  have hr := e_667_15
  simp only [e_667_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 667 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4043)) := by
  have c42060 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42061 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42062 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42063 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42064 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42065 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_667_16 := arithEq_of_rows h (row := 667) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_16
  simp only [← c42060, ← c42061, ← c42062] at e_667_16
  have e_667_17 := arithEq_of_rows h (row := 667) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_17
  simp only [← c42063, ← c42064, k2, ← c42065] at e_667_17
  have hr := e_667_17
  simp only [e_667_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 667 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4044)) := by
  have c42066 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42067 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42068 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42069 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42070 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42071 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_667_18 := arithEq_of_rows h (row := 667) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_18
  simp only [← c42066, ← c42067, ← c42068] at e_667_18
  have e_667_19 := arithEq_of_rows h (row := 667) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_667_19
  simp only [← c42069, ← c42070, k2, ← c42071] at e_667_19
  have hr := e_667_19
  simp only [e_667_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 668 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4045)) := by
  have c42072 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42073 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42074 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42075 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42076 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42077 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_668_0 := arithEq_of_rows h (row := 668) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_0
  simp only [← c42072, ← c42073, ← c42074] at e_668_0
  have e_668_1 := arithEq_of_rows h (row := 668) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_1
  simp only [← c42075, ← c42076, k2, ← c42077] at e_668_1
  have hr := e_668_1
  simp only [e_668_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 668 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4046)) := by
  have c42078 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42079 := (publicBatchWrapper89_copies1314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42080 := (publicBatchWrapper89_copies1315 a h).1
  have c42081 := (publicBatchWrapper89_copies1315 a h).2.1
  have c42082 := (publicBatchWrapper89_copies1315 a h).2.2.1
  have c42083 := (publicBatchWrapper89_copies1315 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_668_2 := arithEq_of_rows h (row := 668) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_2
  simp only [← c42078, ← c42079, ← c42080] at e_668_2
  have e_668_3 := arithEq_of_rows h (row := 668) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_3
  simp only [← c42081, ← c42082, k2, ← c42083] at e_668_3
  have hr := e_668_3
  simp only [e_668_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 668 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4047)) := by
  have c42084 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.1
  have c42085 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.1
  have c42086 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.1
  have c42087 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.1
  have c42088 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.1
  have c42089 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_668_4 := arithEq_of_rows h (row := 668) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_4
  simp only [← c42084, ← c42085, ← c42086] at e_668_4
  have e_668_5 := arithEq_of_rows h (row := 668) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_5
  simp only [← c42087, ← c42088, k2, ← c42089] at e_668_5
  have hr := e_668_5
  simp only [e_668_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 668 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4048)) := by
  have c42090 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42091 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42092 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42093 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42094 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42095 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_668_6 := arithEq_of_rows h (row := 668) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_6
  simp only [← c42090, ← c42091, ← c42092] at e_668_6
  have e_668_7 := arithEq_of_rows h (row := 668) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_7
  simp only [← c42093, ← c42094, k2, ← c42095] at e_668_7
  have hr := e_668_7
  simp only [e_668_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 668 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4049)) := by
  have c42096 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42097 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42098 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42099 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42100 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42101 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_668_8 := arithEq_of_rows h (row := 668) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_8
  simp only [← c42096, ← c42097, ← c42098] at e_668_8
  have e_668_9 := arithEq_of_rows h (row := 668) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_9
  simp only [← c42099, ← c42100, k2, ← c42101] at e_668_9
  have hr := e_668_9
  simp only [e_668_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 668 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4050)) := by
  have c42102 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42103 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42104 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42105 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42106 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42107 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_668_10 := arithEq_of_rows h (row := 668) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_10
  simp only [← c42102, ← c42103, ← c42104] at e_668_10
  have e_668_11 := arithEq_of_rows h (row := 668) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_11
  simp only [← c42105, ← c42106, k2, ← c42107] at e_668_11
  have hr := e_668_11
  simp only [e_668_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 668 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4051)) := by
  have c42108 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42109 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42110 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42111 := (publicBatchWrapper89_copies1315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42112 := (publicBatchWrapper89_copies1316 a h).1
  have c42113 := (publicBatchWrapper89_copies1316 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_668_12 := arithEq_of_rows h (row := 668) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_12
  simp only [← c42108, ← c42109, ← c42110] at e_668_12
  have e_668_13 := arithEq_of_rows h (row := 668) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_13
  simp only [← c42111, ← c42112, k2, ← c42113] at e_668_13
  have hr := e_668_13
  simp only [e_668_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 668 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4052)) := by
  have c42114 := (publicBatchWrapper89_copies1316 a h).2.2.1
  have c42115 := (publicBatchWrapper89_copies1316 a h).2.2.2.1
  have c42116 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.1
  have c42117 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.1
  have c42118 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.1
  have c42119 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_668_14 := arithEq_of_rows h (row := 668) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_14
  simp only [← c42114, ← c42115, ← c42116] at e_668_14
  have e_668_15 := arithEq_of_rows h (row := 668) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_15
  simp only [← c42117, ← c42118, k2, ← c42119] at e_668_15
  have hr := e_668_15
  simp only [e_668_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 668 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4053)) := by
  have c42120 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.1
  have c42121 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.1
  have c42122 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42123 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42124 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42125 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_668_16 := arithEq_of_rows h (row := 668) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_16
  simp only [← c42120, ← c42121, ← c42122] at e_668_16
  have e_668_17 := arithEq_of_rows h (row := 668) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_17
  simp only [← c42123, ← c42124, k2, ← c42125] at e_668_17
  have hr := e_668_17
  simp only [e_668_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 668 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4054)) := by
  have c42126 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42127 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42128 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42129 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42130 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42131 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_668_18 := arithEq_of_rows h (row := 668) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_18
  simp only [← c42126, ← c42127, ← c42128] at e_668_18
  have e_668_19 := arithEq_of_rows h (row := 668) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_668_19
  simp only [← c42129, ← c42130, k2, ← c42131] at e_668_19
  have hr := e_668_19
  simp only [e_668_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 669 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4055)) := by
  have c42132 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42133 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42134 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42135 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42136 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42137 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_669_0 := arithEq_of_rows h (row := 669) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_0
  simp only [← c42132, ← c42133, ← c42134] at e_669_0
  have e_669_1 := arithEq_of_rows h (row := 669) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_1
  simp only [← c42135, ← c42136, k2, ← c42137] at e_669_1
  have hr := e_669_1
  simp only [e_669_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 669 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4056)) := by
  have c42138 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42139 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42140 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42141 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42142 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42143 := (publicBatchWrapper89_copies1316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_669_2 := arithEq_of_rows h (row := 669) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_2
  simp only [← c42138, ← c42139, ← c42140] at e_669_2
  have e_669_3 := arithEq_of_rows h (row := 669) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_3
  simp only [← c42141, ← c42142, k2, ← c42143] at e_669_3
  have hr := e_669_3
  simp only [e_669_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 669 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4057)) := by
  have c42144 := (publicBatchWrapper89_copies1317 a h).1
  have c42145 := (publicBatchWrapper89_copies1317 a h).2.1
  have c42146 := (publicBatchWrapper89_copies1317 a h).2.2.1
  have c42147 := (publicBatchWrapper89_copies1317 a h).2.2.2.1
  have c42148 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.1
  have c42149 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_669_4 := arithEq_of_rows h (row := 669) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_4
  simp only [← c42144, ← c42145, ← c42146] at e_669_4
  have e_669_5 := arithEq_of_rows h (row := 669) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_5
  simp only [← c42147, ← c42148, k2, ← c42149] at e_669_5
  have hr := e_669_5
  simp only [e_669_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 669 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4058)) := by
  have c42150 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.1
  have c42151 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.1
  have c42152 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.1
  have c42153 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.1
  have c42154 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42155 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_669_6 := arithEq_of_rows h (row := 669) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_6
  simp only [← c42150, ← c42151, ← c42152] at e_669_6
  have e_669_7 := arithEq_of_rows h (row := 669) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_7
  simp only [← c42153, ← c42154, k2, ← c42155] at e_669_7
  have hr := e_669_7
  simp only [e_669_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 669 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4059)) := by
  have c42156 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42157 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42158 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42159 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42160 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42161 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_669_8 := arithEq_of_rows h (row := 669) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_8
  simp only [← c42156, ← c42157, ← c42158] at e_669_8
  have e_669_9 := arithEq_of_rows h (row := 669) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_9
  simp only [← c42159, ← c42160, k2, ← c42161] at e_669_9
  have hr := e_669_9
  simp only [e_669_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 669 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4060)) := by
  have c42162 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42163 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42164 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42165 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42166 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42167 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_669_10 := arithEq_of_rows h (row := 669) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_10
  simp only [← c42162, ← c42163, ← c42164] at e_669_10
  have e_669_11 := arithEq_of_rows h (row := 669) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_11
  simp only [← c42165, ← c42166, k2, ← c42167] at e_669_11
  have hr := e_669_11
  simp only [e_669_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 669 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4061)) := by
  have c42168 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42169 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42170 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42171 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42172 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42173 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_669_12 := arithEq_of_rows h (row := 669) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_12
  simp only [← c42168, ← c42169, ← c42170] at e_669_12
  have e_669_13 := arithEq_of_rows h (row := 669) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_13
  simp only [← c42171, ← c42172, k2, ← c42173] at e_669_13
  have hr := e_669_13
  simp only [e_669_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 669 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4062)) := by
  have c42174 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42175 := (publicBatchWrapper89_copies1317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42176 := (publicBatchWrapper89_copies1318 a h).1
  have c42177 := (publicBatchWrapper89_copies1318 a h).2.1
  have c42178 := (publicBatchWrapper89_copies1318 a h).2.2.1
  have c42179 := (publicBatchWrapper89_copies1318 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_669_14 := arithEq_of_rows h (row := 669) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_14
  simp only [← c42174, ← c42175, ← c42176] at e_669_14
  have e_669_15 := arithEq_of_rows h (row := 669) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_15
  simp only [← c42177, ← c42178, k2, ← c42179] at e_669_15
  have hr := e_669_15
  simp only [e_669_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 669 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4063)) := by
  have c42180 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.1
  have c42181 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.1
  have c42182 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.1
  have c42183 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.1
  have c42184 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.1
  have c42185 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_669_16 := arithEq_of_rows h (row := 669) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_16
  simp only [← c42180, ← c42181, ← c42182] at e_669_16
  have e_669_17 := arithEq_of_rows h (row := 669) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_17
  simp only [← c42183, ← c42184, k2, ← c42185] at e_669_17
  have hr := e_669_17
  simp only [e_669_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 669 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4064)) := by
  have c42186 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42187 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42188 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42189 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42190 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42191 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_669_18 := arithEq_of_rows h (row := 669) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_18
  simp only [← c42186, ← c42187, ← c42188] at e_669_18
  have e_669_19 := arithEq_of_rows h (row := 669) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_669_19
  simp only [← c42189, ← c42190, k2, ← c42191] at e_669_19
  have hr := e_669_19
  simp only [e_669_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 670 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4065)) := by
  have c42192 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42193 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42194 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42195 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42196 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42197 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_670_0 := arithEq_of_rows h (row := 670) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_0
  simp only [← c42192, ← c42193, ← c42194] at e_670_0
  have e_670_1 := arithEq_of_rows h (row := 670) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_1
  simp only [← c42195, ← c42196, k2, ← c42197] at e_670_1
  have hr := e_670_1
  simp only [e_670_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 670 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4066)) := by
  have c42198 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42199 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42200 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42201 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42202 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42203 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_670_2 := arithEq_of_rows h (row := 670) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_2
  simp only [← c42198, ← c42199, ← c42200] at e_670_2
  have e_670_3 := arithEq_of_rows h (row := 670) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_3
  simp only [← c42201, ← c42202, k2, ← c42203] at e_670_3
  have hr := e_670_3
  simp only [e_670_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 670 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4067)) := by
  have c42204 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42205 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42206 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42207 := (publicBatchWrapper89_copies1318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42208 := (publicBatchWrapper89_copies1319 a h).1
  have c42209 := (publicBatchWrapper89_copies1319 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_670_4 := arithEq_of_rows h (row := 670) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_4
  simp only [← c42204, ← c42205, ← c42206] at e_670_4
  have e_670_5 := arithEq_of_rows h (row := 670) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_5
  simp only [← c42207, ← c42208, k2, ← c42209] at e_670_5
  have hr := e_670_5
  simp only [e_670_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 670 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4068)) := by
  have c42210 := (publicBatchWrapper89_copies1319 a h).2.2.1
  have c42211 := (publicBatchWrapper89_copies1319 a h).2.2.2.1
  have c42212 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.1
  have c42213 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.1
  have c42214 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.1
  have c42215 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_670_6 := arithEq_of_rows h (row := 670) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_6
  simp only [← c42210, ← c42211, ← c42212] at e_670_6
  have e_670_7 := arithEq_of_rows h (row := 670) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_7
  simp only [← c42213, ← c42214, k2, ← c42215] at e_670_7
  have hr := e_670_7
  simp only [e_670_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 670 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4069)) := by
  have c42216 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.1
  have c42217 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.1
  have c42218 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42219 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42220 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42221 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_670_8 := arithEq_of_rows h (row := 670) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_8
  simp only [← c42216, ← c42217, ← c42218] at e_670_8
  have e_670_9 := arithEq_of_rows h (row := 670) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_9
  simp only [← c42219, ← c42220, k2, ← c42221] at e_670_9
  have hr := e_670_9
  simp only [e_670_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 670 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4070)) := by
  have c42222 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42223 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42224 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42225 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42226 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42227 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_670_10 := arithEq_of_rows h (row := 670) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_10
  simp only [← c42222, ← c42223, ← c42224] at e_670_10
  have e_670_11 := arithEq_of_rows h (row := 670) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_11
  simp only [← c42225, ← c42226, k2, ← c42227] at e_670_11
  have hr := e_670_11
  simp only [e_670_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 670 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4071)) := by
  have c42228 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42229 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42230 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42231 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42232 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42233 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_670_12 := arithEq_of_rows h (row := 670) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_12
  simp only [← c42228, ← c42229, ← c42230] at e_670_12
  have e_670_13 := arithEq_of_rows h (row := 670) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_13
  simp only [← c42231, ← c42232, k2, ← c42233] at e_670_13
  have hr := e_670_13
  simp only [e_670_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 670 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4072)) := by
  have c42234 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42235 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42236 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42237 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42238 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42239 := (publicBatchWrapper89_copies1319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_670_14 := arithEq_of_rows h (row := 670) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_14
  simp only [← c42234, ← c42235, ← c42236] at e_670_14
  have e_670_15 := arithEq_of_rows h (row := 670) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_15
  simp only [← c42237, ← c42238, k2, ← c42239] at e_670_15
  have hr := e_670_15
  simp only [e_670_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 670 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4073)) := by
  have c42240 := (publicBatchWrapper89_copies1320 a h).1
  have c42241 := (publicBatchWrapper89_copies1320 a h).2.1
  have c42242 := (publicBatchWrapper89_copies1320 a h).2.2.1
  have c42243 := (publicBatchWrapper89_copies1320 a h).2.2.2.1
  have c42244 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.1
  have c42245 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_670_16 := arithEq_of_rows h (row := 670) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_16
  simp only [← c42240, ← c42241, ← c42242] at e_670_16
  have e_670_17 := arithEq_of_rows h (row := 670) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_17
  simp only [← c42243, ← c42244, k2, ← c42245] at e_670_17
  have hr := e_670_17
  simp only [e_670_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 670 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4074)) := by
  have c42246 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.1
  have c42247 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.1
  have c42248 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.1
  have c42249 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.1
  have c42250 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42251 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_670_18 := arithEq_of_rows h (row := 670) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_18
  simp only [← c42246, ← c42247, ← c42248] at e_670_18
  have e_670_19 := arithEq_of_rows h (row := 670) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_670_19
  simp only [← c42249, ← c42250, k2, ← c42251] at e_670_19
  have hr := e_670_19
  simp only [e_670_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 671 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4075)) := by
  have c42252 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42253 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42254 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42255 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42256 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42257 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_671_0 := arithEq_of_rows h (row := 671) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_0
  simp only [← c42252, ← c42253, ← c42254] at e_671_0
  have e_671_1 := arithEq_of_rows h (row := 671) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_1
  simp only [← c42255, ← c42256, k2, ← c42257] at e_671_1
  have hr := e_671_1
  simp only [e_671_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 671 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4076)) := by
  have c42258 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42259 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42260 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42261 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42262 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42263 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_671_2 := arithEq_of_rows h (row := 671) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_2
  simp only [← c42258, ← c42259, ← c42260] at e_671_2
  have e_671_3 := arithEq_of_rows h (row := 671) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_3
  simp only [← c42261, ← c42262, k2, ← c42263] at e_671_3
  have hr := e_671_3
  simp only [e_671_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 671 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4077)) := by
  have c42264 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42265 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42266 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42267 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42268 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42269 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_671_4 := arithEq_of_rows h (row := 671) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_4
  simp only [← c42264, ← c42265, ← c42266] at e_671_4
  have e_671_5 := arithEq_of_rows h (row := 671) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_5
  simp only [← c42267, ← c42268, k2, ← c42269] at e_671_5
  have hr := e_671_5
  simp only [e_671_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 671 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4078)) := by
  have c42270 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42271 := (publicBatchWrapper89_copies1320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42272 := (publicBatchWrapper89_copies1321 a h).1
  have c42273 := (publicBatchWrapper89_copies1321 a h).2.1
  have c42274 := (publicBatchWrapper89_copies1321 a h).2.2.1
  have c42275 := (publicBatchWrapper89_copies1321 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_671_6 := arithEq_of_rows h (row := 671) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_6
  simp only [← c42270, ← c42271, ← c42272] at e_671_6
  have e_671_7 := arithEq_of_rows h (row := 671) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_7
  simp only [← c42273, ← c42274, k2, ← c42275] at e_671_7
  have hr := e_671_7
  simp only [e_671_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 671 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4079)) := by
  have c42276 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.1
  have c42277 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.1
  have c42278 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.1
  have c42279 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.1
  have c42280 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.1
  have c42281 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_671_8 := arithEq_of_rows h (row := 671) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_8
  simp only [← c42276, ← c42277, ← c42278] at e_671_8
  have e_671_9 := arithEq_of_rows h (row := 671) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_9
  simp only [← c42279, ← c42280, k2, ← c42281] at e_671_9
  have hr := e_671_9
  simp only [e_671_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 671 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4080)) := by
  have c42282 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42283 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42284 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42285 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42286 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42287 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_671_10 := arithEq_of_rows h (row := 671) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_10
  simp only [← c42282, ← c42283, ← c42284] at e_671_10
  have e_671_11 := arithEq_of_rows h (row := 671) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_11
  simp only [← c42285, ← c42286, k2, ← c42287] at e_671_11
  have hr := e_671_11
  simp only [e_671_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 671 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4081)) := by
  have c42288 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42289 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42290 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42291 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42292 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42293 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_671_12 := arithEq_of_rows h (row := 671) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_12
  simp only [← c42288, ← c42289, ← c42290] at e_671_12
  have e_671_13 := arithEq_of_rows h (row := 671) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_13
  simp only [← c42291, ← c42292, k2, ← c42293] at e_671_13
  have hr := e_671_13
  simp only [e_671_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 671 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4082)) := by
  have c42294 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42295 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42296 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42297 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42298 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42299 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_671_14 := arithEq_of_rows h (row := 671) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_14
  simp only [← c42294, ← c42295, ← c42296] at e_671_14
  have e_671_15 := arithEq_of_rows h (row := 671) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_15
  simp only [← c42297, ← c42298, k2, ← c42299] at e_671_15
  have hr := e_671_15
  simp only [e_671_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 671 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4083)) := by
  have c42300 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42301 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42302 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42303 := (publicBatchWrapper89_copies1321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42304 := (publicBatchWrapper89_copies1322 a h).1
  have c42305 := (publicBatchWrapper89_copies1322 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_671_16 := arithEq_of_rows h (row := 671) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_16
  simp only [← c42300, ← c42301, ← c42302] at e_671_16
  have e_671_17 := arithEq_of_rows h (row := 671) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_17
  simp only [← c42303, ← c42304, k2, ← c42305] at e_671_17
  have hr := e_671_17
  simp only [e_671_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 671 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4084)) := by
  have c42306 := (publicBatchWrapper89_copies1322 a h).2.2.1
  have c42307 := (publicBatchWrapper89_copies1322 a h).2.2.2.1
  have c42308 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.1
  have c42309 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.1
  have c42310 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.1
  have c42311 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_671_18 := arithEq_of_rows h (row := 671) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_18
  simp only [← c42306, ← c42307, ← c42308] at e_671_18
  have e_671_19 := arithEq_of_rows h (row := 671) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_671_19
  simp only [← c42309, ← c42310, k2, ← c42311] at e_671_19
  have hr := e_671_19
  simp only [e_671_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 672 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4085)) := by
  have c42312 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.1
  have c42313 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.1
  have c42314 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42315 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42316 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42317 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_672_0 := arithEq_of_rows h (row := 672) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_0
  simp only [← c42312, ← c42313, ← c42314] at e_672_0
  have e_672_1 := arithEq_of_rows h (row := 672) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_1
  simp only [← c42315, ← c42316, k2, ← c42317] at e_672_1
  have hr := e_672_1
  simp only [e_672_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 672 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4086)) := by
  have c42318 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42319 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42320 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42321 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42322 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42323 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_672_2 := arithEq_of_rows h (row := 672) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_2
  simp only [← c42318, ← c42319, ← c42320] at e_672_2
  have e_672_3 := arithEq_of_rows h (row := 672) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_3
  simp only [← c42321, ← c42322, k2, ← c42323] at e_672_3
  have hr := e_672_3
  simp only [e_672_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 672 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4087)) := by
  have c42324 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42325 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42326 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42327 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42328 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42329 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_672_4 := arithEq_of_rows h (row := 672) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_4
  simp only [← c42324, ← c42325, ← c42326] at e_672_4
  have e_672_5 := arithEq_of_rows h (row := 672) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_5
  simp only [← c42327, ← c42328, k2, ← c42329] at e_672_5
  have hr := e_672_5
  simp only [e_672_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 672 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4088)) := by
  have c42330 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42331 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42332 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42333 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42334 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42335 := (publicBatchWrapper89_copies1322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_672_6 := arithEq_of_rows h (row := 672) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_6
  simp only [← c42330, ← c42331, ← c42332] at e_672_6
  have e_672_7 := arithEq_of_rows h (row := 672) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_7
  simp only [← c42333, ← c42334, k2, ← c42335] at e_672_7
  have hr := e_672_7
  simp only [e_672_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 672 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4089)) := by
  have c42336 := (publicBatchWrapper89_copies1323 a h).1
  have c42337 := (publicBatchWrapper89_copies1323 a h).2.1
  have c42338 := (publicBatchWrapper89_copies1323 a h).2.2.1
  have c42339 := (publicBatchWrapper89_copies1323 a h).2.2.2.1
  have c42340 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.1
  have c42341 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_672_8 := arithEq_of_rows h (row := 672) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_8
  simp only [← c42336, ← c42337, ← c42338] at e_672_8
  have e_672_9 := arithEq_of_rows h (row := 672) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_9
  simp only [← c42339, ← c42340, k2, ← c42341] at e_672_9
  have hr := e_672_9
  simp only [e_672_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 672 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4090)) := by
  have c42342 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.1
  have c42343 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.1
  have c42344 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.1
  have c42345 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.1
  have c42346 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42347 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_672_10 := arithEq_of_rows h (row := 672) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_10
  simp only [← c42342, ← c42343, ← c42344] at e_672_10
  have e_672_11 := arithEq_of_rows h (row := 672) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_11
  simp only [← c42345, ← c42346, k2, ← c42347] at e_672_11
  have hr := e_672_11
  simp only [e_672_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 672 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4091)) := by
  have c42348 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42349 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42350 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42351 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42352 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42353 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_672_12 := arithEq_of_rows h (row := 672) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_12
  simp only [← c42348, ← c42349, ← c42350] at e_672_12
  have e_672_13 := arithEq_of_rows h (row := 672) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_13
  simp only [← c42351, ← c42352, k2, ← c42353] at e_672_13
  have hr := e_672_13
  simp only [e_672_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 672 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4092)) := by
  have c42354 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42355 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42356 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42357 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42358 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42359 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_672_14 := arithEq_of_rows h (row := 672) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_14
  simp only [← c42354, ← c42355, ← c42356] at e_672_14
  have e_672_15 := arithEq_of_rows h (row := 672) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_15
  simp only [← c42357, ← c42358, k2, ← c42359] at e_672_15
  have hr := e_672_15
  simp only [e_672_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 672 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4093)) := by
  have c42360 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42361 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42362 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42363 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42364 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42365 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_672_16 := arithEq_of_rows h (row := 672) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_16
  simp only [← c42360, ← c42361, ← c42362] at e_672_16
  have e_672_17 := arithEq_of_rows h (row := 672) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_17
  simp only [← c42363, ← c42364, k2, ← c42365] at e_672_17
  have hr := e_672_17
  simp only [e_672_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 672 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4094)) := by
  have c42366 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42367 := (publicBatchWrapper89_copies1323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42368 := (publicBatchWrapper89_copies1324 a h).1
  have c42369 := (publicBatchWrapper89_copies1324 a h).2.1
  have c42370 := (publicBatchWrapper89_copies1324 a h).2.2.1
  have c42371 := (publicBatchWrapper89_copies1324 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_672_18 := arithEq_of_rows h (row := 672) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_18
  simp only [← c42366, ← c42367, ← c42368] at e_672_18
  have e_672_19 := arithEq_of_rows h (row := 672) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_672_19
  simp only [← c42369, ← c42370, k2, ← c42371] at e_672_19
  have hr := e_672_19
  simp only [e_672_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 673 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4095)) := by
  have c42372 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.1
  have c42373 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.1
  have c42374 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.1
  have c42375 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.1
  have c42376 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.1
  have c42377 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_673_0 := arithEq_of_rows h (row := 673) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_0
  simp only [← c42372, ← c42373, ← c42374] at e_673_0
  have e_673_1 := arithEq_of_rows h (row := 673) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_1
  simp only [← c42375, ← c42376, k2, ← c42377] at e_673_1
  have hr := e_673_1
  simp only [e_673_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 673 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4096)) := by
  have c42378 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42379 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42380 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42381 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42382 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42383 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_673_2 := arithEq_of_rows h (row := 673) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_2
  simp only [← c42378, ← c42379, ← c42380] at e_673_2
  have e_673_3 := arithEq_of_rows h (row := 673) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_3
  simp only [← c42381, ← c42382, k2, ← c42383] at e_673_3
  have hr := e_673_3
  simp only [e_673_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 673 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4097)) := by
  have c42384 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42385 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42386 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42387 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42388 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42389 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_673_4 := arithEq_of_rows h (row := 673) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_4
  simp only [← c42384, ← c42385, ← c42386] at e_673_4
  have e_673_5 := arithEq_of_rows h (row := 673) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_5
  simp only [← c42387, ← c42388, k2, ← c42389] at e_673_5
  have hr := e_673_5
  simp only [e_673_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 673 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4098)) := by
  have c42390 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42391 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42392 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42393 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42394 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42395 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_673_6 := arithEq_of_rows h (row := 673) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_6
  simp only [← c42390, ← c42391, ← c42392] at e_673_6
  have e_673_7 := arithEq_of_rows h (row := 673) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_7
  simp only [← c42393, ← c42394, k2, ← c42395] at e_673_7
  have hr := e_673_7
  simp only [e_673_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 673 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4099)) := by
  have c42396 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42397 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42398 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42399 := (publicBatchWrapper89_copies1324 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42400 := (publicBatchWrapper89_copies1325 a h).1
  have c42401 := (publicBatchWrapper89_copies1325 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_673_8 := arithEq_of_rows h (row := 673) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_8
  simp only [← c42396, ← c42397, ← c42398] at e_673_8
  have e_673_9 := arithEq_of_rows h (row := 673) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_9
  simp only [← c42399, ← c42400, k2, ← c42401] at e_673_9
  have hr := e_673_9
  simp only [e_673_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 673 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4100)) := by
  have c42402 := (publicBatchWrapper89_copies1325 a h).2.2.1
  have c42403 := (publicBatchWrapper89_copies1325 a h).2.2.2.1
  have c42404 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.1
  have c42405 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.1
  have c42406 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.1
  have c42407 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_673_10 := arithEq_of_rows h (row := 673) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_10
  simp only [← c42402, ← c42403, ← c42404] at e_673_10
  have e_673_11 := arithEq_of_rows h (row := 673) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_11
  simp only [← c42405, ← c42406, k2, ← c42407] at e_673_11
  have hr := e_673_11
  simp only [e_673_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 673 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4101)) := by
  have c42408 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.1
  have c42409 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.1
  have c42410 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42411 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42412 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42413 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_673_12 := arithEq_of_rows h (row := 673) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_12
  simp only [← c42408, ← c42409, ← c42410] at e_673_12
  have e_673_13 := arithEq_of_rows h (row := 673) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_13
  simp only [← c42411, ← c42412, k2, ← c42413] at e_673_13
  have hr := e_673_13
  simp only [e_673_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 673 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4102)) := by
  have c42414 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42415 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42416 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42417 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42418 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42419 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_673_14 := arithEq_of_rows h (row := 673) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_14
  simp only [← c42414, ← c42415, ← c42416] at e_673_14
  have e_673_15 := arithEq_of_rows h (row := 673) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_15
  simp only [← c42417, ← c42418, k2, ← c42419] at e_673_15
  have hr := e_673_15
  simp only [e_673_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 673 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4103)) := by
  have c42420 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42421 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42422 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42423 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42424 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42425 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_673_16 := arithEq_of_rows h (row := 673) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_16
  simp only [← c42420, ← c42421, ← c42422] at e_673_16
  have e_673_17 := arithEq_of_rows h (row := 673) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_17
  simp only [← c42423, ← c42424, k2, ← c42425] at e_673_17
  have hr := e_673_17
  simp only [e_673_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 673 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4104)) := by
  have c42426 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42427 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42428 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42429 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42430 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42431 := (publicBatchWrapper89_copies1325 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_673_18 := arithEq_of_rows h (row := 673) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_18
  simp only [← c42426, ← c42427, ← c42428] at e_673_18
  have e_673_19 := arithEq_of_rows h (row := 673) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_673_19
  simp only [← c42429, ← c42430, k2, ← c42431] at e_673_19
  have hr := e_673_19
  simp only [e_673_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 674 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4105)) := by
  have c42432 := (publicBatchWrapper89_copies1326 a h).1
  have c42433 := (publicBatchWrapper89_copies1326 a h).2.1
  have c42434 := (publicBatchWrapper89_copies1326 a h).2.2.1
  have c42435 := (publicBatchWrapper89_copies1326 a h).2.2.2.1
  have c42436 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.1
  have c42437 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_674_0 := arithEq_of_rows h (row := 674) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_0
  simp only [← c42432, ← c42433, ← c42434] at e_674_0
  have e_674_1 := arithEq_of_rows h (row := 674) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_1
  simp only [← c42435, ← c42436, k2, ← c42437] at e_674_1
  have hr := e_674_1
  simp only [e_674_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 674 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4106)) := by
  have c42438 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.1
  have c42439 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.1
  have c42440 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.1
  have c42441 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.1
  have c42442 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42443 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_674_2 := arithEq_of_rows h (row := 674) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_2
  simp only [← c42438, ← c42439, ← c42440] at e_674_2
  have e_674_3 := arithEq_of_rows h (row := 674) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_3
  simp only [← c42441, ← c42442, k2, ← c42443] at e_674_3
  have hr := e_674_3
  simp only [e_674_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 674 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4107)) := by
  have c42444 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42445 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42446 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42447 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42448 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42449 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_674_4 := arithEq_of_rows h (row := 674) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_4
  simp only [← c42444, ← c42445, ← c42446] at e_674_4
  have e_674_5 := arithEq_of_rows h (row := 674) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_5
  simp only [← c42447, ← c42448, k2, ← c42449] at e_674_5
  have hr := e_674_5
  simp only [e_674_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 674 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4108)) := by
  have c42450 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42451 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42452 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42453 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42454 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42455 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_674_6 := arithEq_of_rows h (row := 674) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_6
  simp only [← c42450, ← c42451, ← c42452] at e_674_6
  have e_674_7 := arithEq_of_rows h (row := 674) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_7
  simp only [← c42453, ← c42454, k2, ← c42455] at e_674_7
  have hr := e_674_7
  simp only [e_674_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 674 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4109)) := by
  have c42456 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42457 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42458 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42459 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42460 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42461 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_674_8 := arithEq_of_rows h (row := 674) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_8
  simp only [← c42456, ← c42457, ← c42458] at e_674_8
  have e_674_9 := arithEq_of_rows h (row := 674) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_9
  simp only [← c42459, ← c42460, k2, ← c42461] at e_674_9
  have hr := e_674_9
  simp only [e_674_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 674 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4110)) := by
  have c42462 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42463 := (publicBatchWrapper89_copies1326 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42464 := (publicBatchWrapper89_copies1327 a h).1
  have c42465 := (publicBatchWrapper89_copies1327 a h).2.1
  have c42466 := (publicBatchWrapper89_copies1327 a h).2.2.1
  have c42467 := (publicBatchWrapper89_copies1327 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_674_10 := arithEq_of_rows h (row := 674) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_10
  simp only [← c42462, ← c42463, ← c42464] at e_674_10
  have e_674_11 := arithEq_of_rows h (row := 674) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_11
  simp only [← c42465, ← c42466, k2, ← c42467] at e_674_11
  have hr := e_674_11
  simp only [e_674_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 674 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4111)) := by
  have c42468 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.1
  have c42469 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.1
  have c42470 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.1
  have c42471 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.1
  have c42472 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.1
  have c42473 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_674_12 := arithEq_of_rows h (row := 674) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_12
  simp only [← c42468, ← c42469, ← c42470] at e_674_12
  have e_674_13 := arithEq_of_rows h (row := 674) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_13
  simp only [← c42471, ← c42472, k2, ← c42473] at e_674_13
  have hr := e_674_13
  simp only [e_674_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 674 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4112)) := by
  have c42474 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42475 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42476 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42477 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42478 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42479 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_674_14 := arithEq_of_rows h (row := 674) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_14
  simp only [← c42474, ← c42475, ← c42476] at e_674_14
  have e_674_15 := arithEq_of_rows h (row := 674) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_15
  simp only [← c42477, ← c42478, k2, ← c42479] at e_674_15
  have hr := e_674_15
  simp only [e_674_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 674 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4113)) := by
  have c42480 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42481 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42482 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42483 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42484 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42485 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_674_16 := arithEq_of_rows h (row := 674) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_16
  simp only [← c42480, ← c42481, ← c42482] at e_674_16
  have e_674_17 := arithEq_of_rows h (row := 674) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_17
  simp only [← c42483, ← c42484, k2, ← c42485] at e_674_17
  have hr := e_674_17
  simp only [e_674_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 674 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4114)) := by
  have c42486 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42487 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42488 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42489 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42490 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42491 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_674_18 := arithEq_of_rows h (row := 674) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_18
  simp only [← c42486, ← c42487, ← c42488] at e_674_18
  have e_674_19 := arithEq_of_rows h (row := 674) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_674_19
  simp only [← c42489, ← c42490, k2, ← c42491] at e_674_19
  have hr := e_674_19
  simp only [e_674_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 675 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4115)) := by
  have c42492 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42493 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42494 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42495 := (publicBatchWrapper89_copies1327 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42496 := (publicBatchWrapper89_copies1328 a h).1
  have c42497 := (publicBatchWrapper89_copies1328 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_675_0 := arithEq_of_rows h (row := 675) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_0
  simp only [← c42492, ← c42493, ← c42494] at e_675_0
  have e_675_1 := arithEq_of_rows h (row := 675) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_1
  simp only [← c42495, ← c42496, k2, ← c42497] at e_675_1
  have hr := e_675_1
  simp only [e_675_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 675 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4116)) := by
  have c42498 := (publicBatchWrapper89_copies1328 a h).2.2.1
  have c42499 := (publicBatchWrapper89_copies1328 a h).2.2.2.1
  have c42500 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.1
  have c42501 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.1
  have c42502 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.1
  have c42503 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_675_2 := arithEq_of_rows h (row := 675) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_2
  simp only [← c42498, ← c42499, ← c42500] at e_675_2
  have e_675_3 := arithEq_of_rows h (row := 675) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_3
  simp only [← c42501, ← c42502, k2, ← c42503] at e_675_3
  have hr := e_675_3
  simp only [e_675_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 675 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4117)) := by
  have c42504 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.1
  have c42505 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.1
  have c42506 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42507 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42508 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42509 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_675_4 := arithEq_of_rows h (row := 675) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_4
  simp only [← c42504, ← c42505, ← c42506] at e_675_4
  have e_675_5 := arithEq_of_rows h (row := 675) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_5
  simp only [← c42507, ← c42508, k2, ← c42509] at e_675_5
  have hr := e_675_5
  simp only [e_675_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 675 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4118)) := by
  have c42510 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42511 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42512 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42513 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42514 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42515 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_675_6 := arithEq_of_rows h (row := 675) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_6
  simp only [← c42510, ← c42511, ← c42512] at e_675_6
  have e_675_7 := arithEq_of_rows h (row := 675) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_7
  simp only [← c42513, ← c42514, k2, ← c42515] at e_675_7
  have hr := e_675_7
  simp only [e_675_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 675 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4119)) := by
  have c42516 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42517 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42518 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42519 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42520 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42521 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_675_8 := arithEq_of_rows h (row := 675) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_8
  simp only [← c42516, ← c42517, ← c42518] at e_675_8
  have e_675_9 := arithEq_of_rows h (row := 675) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_9
  simp only [← c42519, ← c42520, k2, ← c42521] at e_675_9
  have hr := e_675_9
  simp only [e_675_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 675 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4120)) := by
  have c42522 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42523 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42524 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42525 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42526 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42527 := (publicBatchWrapper89_copies1328 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_675_10 := arithEq_of_rows h (row := 675) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_10
  simp only [← c42522, ← c42523, ← c42524] at e_675_10
  have e_675_11 := arithEq_of_rows h (row := 675) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_11
  simp only [← c42525, ← c42526, k2, ← c42527] at e_675_11
  have hr := e_675_11
  simp only [e_675_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 675 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4121)) := by
  have c42528 := (publicBatchWrapper89_copies1329 a h).1
  have c42529 := (publicBatchWrapper89_copies1329 a h).2.1
  have c42530 := (publicBatchWrapper89_copies1329 a h).2.2.1
  have c42531 := (publicBatchWrapper89_copies1329 a h).2.2.2.1
  have c42532 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.1
  have c42533 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_675_12 := arithEq_of_rows h (row := 675) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_12
  simp only [← c42528, ← c42529, ← c42530] at e_675_12
  have e_675_13 := arithEq_of_rows h (row := 675) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_13
  simp only [← c42531, ← c42532, k2, ← c42533] at e_675_13
  have hr := e_675_13
  simp only [e_675_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 675 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4122)) := by
  have c42534 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.1
  have c42535 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.1
  have c42536 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.1
  have c42537 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.1
  have c42538 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42539 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_675_14 := arithEq_of_rows h (row := 675) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_14
  simp only [← c42534, ← c42535, ← c42536] at e_675_14
  have e_675_15 := arithEq_of_rows h (row := 675) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_15
  simp only [← c42537, ← c42538, k2, ← c42539] at e_675_15
  have hr := e_675_15
  simp only [e_675_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 675 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4123)) := by
  have c42540 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42541 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42542 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42543 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42544 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42545 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_675_16 := arithEq_of_rows h (row := 675) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_16
  simp only [← c42540, ← c42541, ← c42542] at e_675_16
  have e_675_17 := arithEq_of_rows h (row := 675) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_17
  simp only [← c42543, ← c42544, k2, ← c42545] at e_675_17
  have hr := e_675_17
  simp only [e_675_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 675 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4124)) := by
  have c42546 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42547 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42548 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42549 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42550 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42551 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_675_18 := arithEq_of_rows h (row := 675) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_18
  simp only [← c42546, ← c42547, ← c42548] at e_675_18
  have e_675_19 := arithEq_of_rows h (row := 675) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_675_19
  simp only [← c42549, ← c42550, k2, ← c42551] at e_675_19
  have hr := e_675_19
  simp only [e_675_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 676 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4125)) := by
  have c42552 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42553 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42554 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42555 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42556 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42557 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_676_0 := arithEq_of_rows h (row := 676) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_0
  simp only [← c42552, ← c42553, ← c42554] at e_676_0
  have e_676_1 := arithEq_of_rows h (row := 676) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_1
  simp only [← c42555, ← c42556, k2, ← c42557] at e_676_1
  have hr := e_676_1
  simp only [e_676_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 676 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4126)) := by
  have c42558 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42559 := (publicBatchWrapper89_copies1329 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42560 := (publicBatchWrapper89_copies1330 a h).1
  have c42561 := (publicBatchWrapper89_copies1330 a h).2.1
  have c42562 := (publicBatchWrapper89_copies1330 a h).2.2.1
  have c42563 := (publicBatchWrapper89_copies1330 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_676_2 := arithEq_of_rows h (row := 676) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_2
  simp only [← c42558, ← c42559, ← c42560] at e_676_2
  have e_676_3 := arithEq_of_rows h (row := 676) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_3
  simp only [← c42561, ← c42562, k2, ← c42563] at e_676_3
  have hr := e_676_3
  simp only [e_676_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 676 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4127)) := by
  have c42564 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.1
  have c42565 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.1
  have c42566 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.1
  have c42567 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.1
  have c42568 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.1
  have c42569 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_676_4 := arithEq_of_rows h (row := 676) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_4
  simp only [← c42564, ← c42565, ← c42566] at e_676_4
  have e_676_5 := arithEq_of_rows h (row := 676) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_5
  simp only [← c42567, ← c42568, k2, ← c42569] at e_676_5
  have hr := e_676_5
  simp only [e_676_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 676 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4128)) := by
  have c42570 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42571 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42572 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42573 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42574 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42575 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_676_6 := arithEq_of_rows h (row := 676) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_6
  simp only [← c42570, ← c42571, ← c42572] at e_676_6
  have e_676_7 := arithEq_of_rows h (row := 676) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_7
  simp only [← c42573, ← c42574, k2, ← c42575] at e_676_7
  have hr := e_676_7
  simp only [e_676_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 676 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4129)) := by
  have c42576 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42577 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42578 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42579 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42580 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42581 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_676_8 := arithEq_of_rows h (row := 676) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_8
  simp only [← c42576, ← c42577, ← c42578] at e_676_8
  have e_676_9 := arithEq_of_rows h (row := 676) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_9
  simp only [← c42579, ← c42580, k2, ← c42581] at e_676_9
  have hr := e_676_9
  simp only [e_676_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 676 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4130)) := by
  have c42582 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42583 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42584 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42585 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42586 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42587 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_676_10 := arithEq_of_rows h (row := 676) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_10
  simp only [← c42582, ← c42583, ← c42584] at e_676_10
  have e_676_11 := arithEq_of_rows h (row := 676) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_11
  simp only [← c42585, ← c42586, k2, ← c42587] at e_676_11
  have hr := e_676_11
  simp only [e_676_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 676 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4131)) := by
  have c42588 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42589 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42590 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42591 := (publicBatchWrapper89_copies1330 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42592 := (publicBatchWrapper89_copies1331 a h).1
  have c42593 := (publicBatchWrapper89_copies1331 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_676_12 := arithEq_of_rows h (row := 676) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_12
  simp only [← c42588, ← c42589, ← c42590] at e_676_12
  have e_676_13 := arithEq_of_rows h (row := 676) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_13
  simp only [← c42591, ← c42592, k2, ← c42593] at e_676_13
  have hr := e_676_13
  simp only [e_676_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 676 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4132)) := by
  have c42594 := (publicBatchWrapper89_copies1331 a h).2.2.1
  have c42595 := (publicBatchWrapper89_copies1331 a h).2.2.2.1
  have c42596 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.1
  have c42597 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.1
  have c42598 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.1
  have c42599 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_676_14 := arithEq_of_rows h (row := 676) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_14
  simp only [← c42594, ← c42595, ← c42596] at e_676_14
  have e_676_15 := arithEq_of_rows h (row := 676) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_15
  simp only [← c42597, ← c42598, k2, ← c42599] at e_676_15
  have hr := e_676_15
  simp only [e_676_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 676 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4133)) := by
  have c42600 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.1
  have c42601 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.1
  have c42602 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42603 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42604 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42605 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_676_16 := arithEq_of_rows h (row := 676) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_16
  simp only [← c42600, ← c42601, ← c42602] at e_676_16
  have e_676_17 := arithEq_of_rows h (row := 676) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_17
  simp only [← c42603, ← c42604, k2, ← c42605] at e_676_17
  have hr := e_676_17
  simp only [e_676_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 676 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4134)) := by
  have c42606 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42607 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42608 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42609 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42610 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42611 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_676_18 := arithEq_of_rows h (row := 676) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_18
  simp only [← c42606, ← c42607, ← c42608] at e_676_18
  have e_676_19 := arithEq_of_rows h (row := 676) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_676_19
  simp only [← c42609, ← c42610, k2, ← c42611] at e_676_19
  have hr := e_676_19
  simp only [e_676_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 677 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4135)) := by
  have c42612 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42613 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42614 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42615 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42616 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42617 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_677_0 := arithEq_of_rows h (row := 677) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_0
  simp only [← c42612, ← c42613, ← c42614] at e_677_0
  have e_677_1 := arithEq_of_rows h (row := 677) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_1
  simp only [← c42615, ← c42616, k2, ← c42617] at e_677_1
  have hr := e_677_1
  simp only [e_677_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 677 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4136)) := by
  have c42618 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42619 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42620 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42621 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42622 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42623 := (publicBatchWrapper89_copies1331 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_677_2 := arithEq_of_rows h (row := 677) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_2
  simp only [← c42618, ← c42619, ← c42620] at e_677_2
  have e_677_3 := arithEq_of_rows h (row := 677) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_3
  simp only [← c42621, ← c42622, k2, ← c42623] at e_677_3
  have hr := e_677_3
  simp only [e_677_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 677 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4137)) := by
  have c42624 := (publicBatchWrapper89_copies1332 a h).1
  have c42625 := (publicBatchWrapper89_copies1332 a h).2.1
  have c42626 := (publicBatchWrapper89_copies1332 a h).2.2.1
  have c42627 := (publicBatchWrapper89_copies1332 a h).2.2.2.1
  have c42628 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.1
  have c42629 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_677_4 := arithEq_of_rows h (row := 677) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_4
  simp only [← c42624, ← c42625, ← c42626] at e_677_4
  have e_677_5 := arithEq_of_rows h (row := 677) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_5
  simp only [← c42627, ← c42628, k2, ← c42629] at e_677_5
  have hr := e_677_5
  simp only [e_677_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 677 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4138)) := by
  have c42630 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.1
  have c42631 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.1
  have c42632 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.1
  have c42633 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.1
  have c42634 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42635 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_677_6 := arithEq_of_rows h (row := 677) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_6
  simp only [← c42630, ← c42631, ← c42632] at e_677_6
  have e_677_7 := arithEq_of_rows h (row := 677) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_7
  simp only [← c42633, ← c42634, k2, ← c42635] at e_677_7
  have hr := e_677_7
  simp only [e_677_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 677 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4139)) := by
  have c42636 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42637 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42638 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42639 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42640 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42641 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_677_8 := arithEq_of_rows h (row := 677) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_8
  simp only [← c42636, ← c42637, ← c42638] at e_677_8
  have e_677_9 := arithEq_of_rows h (row := 677) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_9
  simp only [← c42639, ← c42640, k2, ← c42641] at e_677_9
  have hr := e_677_9
  simp only [e_677_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 677 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4140)) := by
  have c42642 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42643 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42644 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42645 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42646 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42647 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_677_10 := arithEq_of_rows h (row := 677) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_10
  simp only [← c42642, ← c42643, ← c42644] at e_677_10
  have e_677_11 := arithEq_of_rows h (row := 677) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_11
  simp only [← c42645, ← c42646, k2, ← c42647] at e_677_11
  have hr := e_677_11
  simp only [e_677_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 677 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4141)) := by
  have c42648 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42649 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42650 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42651 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42652 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42653 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_677_12 := arithEq_of_rows h (row := 677) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_12
  simp only [← c42648, ← c42649, ← c42650] at e_677_12
  have e_677_13 := arithEq_of_rows h (row := 677) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_13
  simp only [← c42651, ← c42652, k2, ← c42653] at e_677_13
  have hr := e_677_13
  simp only [e_677_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 677 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4142)) := by
  have c42654 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42655 := (publicBatchWrapper89_copies1332 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42656 := (publicBatchWrapper89_copies1333 a h).1
  have c42657 := (publicBatchWrapper89_copies1333 a h).2.1
  have c42658 := (publicBatchWrapper89_copies1333 a h).2.2.1
  have c42659 := (publicBatchWrapper89_copies1333 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_677_14 := arithEq_of_rows h (row := 677) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_14
  simp only [← c42654, ← c42655, ← c42656] at e_677_14
  have e_677_15 := arithEq_of_rows h (row := 677) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_15
  simp only [← c42657, ← c42658, k2, ← c42659] at e_677_15
  have hr := e_677_15
  simp only [e_677_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 677 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4143)) := by
  have c42660 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.1
  have c42661 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.1
  have c42662 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.1
  have c42663 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.1
  have c42664 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.1
  have c42665 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_677_16 := arithEq_of_rows h (row := 677) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_16
  simp only [← c42660, ← c42661, ← c42662] at e_677_16
  have e_677_17 := arithEq_of_rows h (row := 677) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_17
  simp only [← c42663, ← c42664, k2, ← c42665] at e_677_17
  have hr := e_677_17
  simp only [e_677_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 677 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4144)) := by
  have c42666 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42667 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42668 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42669 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42670 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42671 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_677_18 := arithEq_of_rows h (row := 677) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_18
  simp only [← c42666, ← c42667, ← c42668] at e_677_18
  have e_677_19 := arithEq_of_rows h (row := 677) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_677_19
  simp only [← c42669, ← c42670, k2, ← c42671] at e_677_19
  have hr := e_677_19
  simp only [e_677_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 678 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4145)) := by
  have c42672 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42673 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42674 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42675 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42676 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42677 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_678_0 := arithEq_of_rows h (row := 678) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_0
  simp only [← c42672, ← c42673, ← c42674] at e_678_0
  have e_678_1 := arithEq_of_rows h (row := 678) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_1
  simp only [← c42675, ← c42676, k2, ← c42677] at e_678_1
  have hr := e_678_1
  simp only [e_678_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 678 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4146)) := by
  have c42678 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42679 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42680 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42681 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42682 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42683 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_678_2 := arithEq_of_rows h (row := 678) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_2
  simp only [← c42678, ← c42679, ← c42680] at e_678_2
  have e_678_3 := arithEq_of_rows h (row := 678) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_3
  simp only [← c42681, ← c42682, k2, ← c42683] at e_678_3
  have hr := e_678_3
  simp only [e_678_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 678 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4147)) := by
  have c42684 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42685 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42686 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42687 := (publicBatchWrapper89_copies1333 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42688 := (publicBatchWrapper89_copies1334 a h).1
  have c42689 := (publicBatchWrapper89_copies1334 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_678_4 := arithEq_of_rows h (row := 678) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_4
  simp only [← c42684, ← c42685, ← c42686] at e_678_4
  have e_678_5 := arithEq_of_rows h (row := 678) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_5
  simp only [← c42687, ← c42688, k2, ← c42689] at e_678_5
  have hr := e_678_5
  simp only [e_678_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 678 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4148)) := by
  have c42690 := (publicBatchWrapper89_copies1334 a h).2.2.1
  have c42691 := (publicBatchWrapper89_copies1334 a h).2.2.2.1
  have c42692 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.1
  have c42693 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.1
  have c42694 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.1
  have c42695 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_678_6 := arithEq_of_rows h (row := 678) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_6
  simp only [← c42690, ← c42691, ← c42692] at e_678_6
  have e_678_7 := arithEq_of_rows h (row := 678) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_7
  simp only [← c42693, ← c42694, k2, ← c42695] at e_678_7
  have hr := e_678_7
  simp only [e_678_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 678 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4149)) := by
  have c42696 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.1
  have c42697 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.1
  have c42698 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42699 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42700 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42701 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_678_8 := arithEq_of_rows h (row := 678) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_8
  simp only [← c42696, ← c42697, ← c42698] at e_678_8
  have e_678_9 := arithEq_of_rows h (row := 678) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_9
  simp only [← c42699, ← c42700, k2, ← c42701] at e_678_9
  have hr := e_678_9
  simp only [e_678_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 678 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4150)) := by
  have c42702 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42703 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42704 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42705 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42706 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42707 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_678_10 := arithEq_of_rows h (row := 678) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_10
  simp only [← c42702, ← c42703, ← c42704] at e_678_10
  have e_678_11 := arithEq_of_rows h (row := 678) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_11
  simp only [← c42705, ← c42706, k2, ← c42707] at e_678_11
  have hr := e_678_11
  simp only [e_678_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 678 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4151)) := by
  have c42708 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42709 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42710 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42711 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42712 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42713 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_678_12 := arithEq_of_rows h (row := 678) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_12
  simp only [← c42708, ← c42709, ← c42710] at e_678_12
  have e_678_13 := arithEq_of_rows h (row := 678) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_13
  simp only [← c42711, ← c42712, k2, ← c42713] at e_678_13
  have hr := e_678_13
  simp only [e_678_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 678 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4152)) := by
  have c42714 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42715 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42716 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42717 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42718 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42719 := (publicBatchWrapper89_copies1334 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_678_14 := arithEq_of_rows h (row := 678) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_14
  simp only [← c42714, ← c42715, ← c42716] at e_678_14
  have e_678_15 := arithEq_of_rows h (row := 678) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_15
  simp only [← c42717, ← c42718, k2, ← c42719] at e_678_15
  have hr := e_678_15
  simp only [e_678_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 678 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4153)) := by
  have c42720 := (publicBatchWrapper89_copies1335 a h).1
  have c42721 := (publicBatchWrapper89_copies1335 a h).2.1
  have c42722 := (publicBatchWrapper89_copies1335 a h).2.2.1
  have c42723 := (publicBatchWrapper89_copies1335 a h).2.2.2.1
  have c42724 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.1
  have c42725 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_678_16 := arithEq_of_rows h (row := 678) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_16
  simp only [← c42720, ← c42721, ← c42722] at e_678_16
  have e_678_17 := arithEq_of_rows h (row := 678) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_17
  simp only [← c42723, ← c42724, k2, ← c42725] at e_678_17
  have hr := e_678_17
  simp only [e_678_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 678 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4154)) := by
  have c42726 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.1
  have c42727 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.1
  have c42728 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.1
  have c42729 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.1
  have c42730 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42731 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_678_18 := arithEq_of_rows h (row := 678) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_18
  simp only [← c42726, ← c42727, ← c42728] at e_678_18
  have e_678_19 := arithEq_of_rows h (row := 678) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_678_19
  simp only [← c42729, ← c42730, k2, ← c42731] at e_678_19
  have hr := e_678_19
  simp only [e_678_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 679 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4155)) := by
  have c42732 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42733 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42734 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42735 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42736 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42737 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_679_0 := arithEq_of_rows h (row := 679) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_0
  simp only [← c42732, ← c42733, ← c42734] at e_679_0
  have e_679_1 := arithEq_of_rows h (row := 679) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_1
  simp only [← c42735, ← c42736, k2, ← c42737] at e_679_1
  have hr := e_679_1
  simp only [e_679_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 679 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4156)) := by
  have c42738 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42739 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42740 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42741 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42742 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42743 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_679_2 := arithEq_of_rows h (row := 679) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_2
  simp only [← c42738, ← c42739, ← c42740] at e_679_2
  have e_679_3 := arithEq_of_rows h (row := 679) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_3
  simp only [← c42741, ← c42742, k2, ← c42743] at e_679_3
  have hr := e_679_3
  simp only [e_679_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 679 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4157)) := by
  have c42744 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42745 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42746 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42747 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42748 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42749 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_679_4 := arithEq_of_rows h (row := 679) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_4
  simp only [← c42744, ← c42745, ← c42746] at e_679_4
  have e_679_5 := arithEq_of_rows h (row := 679) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_5
  simp only [← c42747, ← c42748, k2, ← c42749] at e_679_5
  have hr := e_679_5
  simp only [e_679_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 679 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4158)) := by
  have c42750 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42751 := (publicBatchWrapper89_copies1335 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42752 := (publicBatchWrapper89_copies1336 a h).1
  have c42753 := (publicBatchWrapper89_copies1336 a h).2.1
  have c42754 := (publicBatchWrapper89_copies1336 a h).2.2.1
  have c42755 := (publicBatchWrapper89_copies1336 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_679_6 := arithEq_of_rows h (row := 679) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_6
  simp only [← c42750, ← c42751, ← c42752] at e_679_6
  have e_679_7 := arithEq_of_rows h (row := 679) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_7
  simp only [← c42753, ← c42754, k2, ← c42755] at e_679_7
  have hr := e_679_7
  simp only [e_679_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 679 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4159)) := by
  have c42756 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.1
  have c42757 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.1
  have c42758 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.1
  have c42759 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.1
  have c42760 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.1
  have c42761 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_679_8 := arithEq_of_rows h (row := 679) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_8
  simp only [← c42756, ← c42757, ← c42758] at e_679_8
  have e_679_9 := arithEq_of_rows h (row := 679) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_9
  simp only [← c42759, ← c42760, k2, ← c42761] at e_679_9
  have hr := e_679_9
  simp only [e_679_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 679 47) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4160)) := by
  have c42762 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42763 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42764 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42765 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42766 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42767 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_679_10 := arithEq_of_rows h (row := 679) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_10
  simp only [← c42762, ← c42763, ← c42764] at e_679_10
  have e_679_11 := arithEq_of_rows h (row := 679) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_11
  simp only [← c42765, ← c42766, k2, ← c42767] at e_679_11
  have hr := e_679_11
  simp only [e_679_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 679 55) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4161)) := by
  have c42768 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42769 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42770 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42771 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42772 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42773 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_679_12 := arithEq_of_rows h (row := 679) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_12
  simp only [← c42768, ← c42769, ← c42770] at e_679_12
  have e_679_13 := arithEq_of_rows h (row := 679) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_13
  simp only [← c42771, ← c42772, k2, ← c42773] at e_679_13
  have hr := e_679_13
  simp only [e_679_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 679 63) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4162)) := by
  have c42774 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42775 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42776 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42777 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42778 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42779 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_679_14 := arithEq_of_rows h (row := 679) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_14
  simp only [← c42774, ← c42775, ← c42776] at e_679_14
  have e_679_15 := arithEq_of_rows h (row := 679) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_15
  simp only [← c42777, ← c42778, k2, ← c42779] at e_679_15
  have hr := e_679_15
  simp only [e_679_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 679 71) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4163)) := by
  have c42780 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42781 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42782 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42783 := (publicBatchWrapper89_copies1336 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42784 := (publicBatchWrapper89_copies1337 a h).1
  have c42785 := (publicBatchWrapper89_copies1337 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_679_16 := arithEq_of_rows h (row := 679) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_16
  simp only [← c42780, ← c42781, ← c42782] at e_679_16
  have e_679_17 := arithEq_of_rows h (row := 679) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_17
  simp only [← c42783, ← c42784, k2, ← c42785] at e_679_17
  have hr := e_679_17
  simp only [e_679_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 679 79) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4164)) := by
  have c42786 := (publicBatchWrapper89_copies1337 a h).2.2.1
  have c42787 := (publicBatchWrapper89_copies1337 a h).2.2.2.1
  have c42788 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.1
  have c42789 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.1
  have c42790 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.1
  have c42791 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_679_18 := arithEq_of_rows h (row := 679) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_18
  simp only [← c42786, ← c42787, ← c42788] at e_679_18
  have e_679_19 := arithEq_of_rows h (row := 679) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_679_19
  simp only [← c42789, ← c42790, k2, ← c42791] at e_679_19
  have hr := e_679_19
  simp only [e_679_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 680 7) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4165)) := by
  have c42792 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.1
  have c42793 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.1
  have c42794 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42795 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42796 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42797 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_680_0 := arithEq_of_rows h (row := 680) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_0
  simp only [← c42792, ← c42793, ← c42794] at e_680_0
  have e_680_1 := arithEq_of_rows h (row := 680) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_1
  simp only [← c42795, ← c42796, k2, ← c42797] at e_680_1
  have hr := e_680_1
  simp only [e_680_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 680 15) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4166)) := by
  have c42798 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42799 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42800 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42801 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42802 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42803 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_680_2 := arithEq_of_rows h (row := 680) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_2
  simp only [← c42798, ← c42799, ← c42800] at e_680_2
  have e_680_3 := arithEq_of_rows h (row := 680) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_3
  simp only [← c42801, ← c42802, k2, ← c42803] at e_680_3
  have hr := e_680_3
  simp only [e_680_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 680 23) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4167)) := by
  have c42804 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42805 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42806 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42807 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42808 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42809 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_680_4 := arithEq_of_rows h (row := 680) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_4
  simp only [← c42804, ← c42805, ← c42806] at e_680_4
  have e_680_5 := arithEq_of_rows h (row := 680) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_5
  simp only [← c42807, ← c42808, k2, ← c42809] at e_680_5
  have hr := e_680_5
  simp only [e_680_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 680 31) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4168)) := by
  have c42810 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42811 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42812 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42813 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42814 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42815 := (publicBatchWrapper89_copies1337 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_680_6 := arithEq_of_rows h (row := 680) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_6
  simp only [← c42810, ← c42811, ← c42812] at e_680_6
  have e_680_7 := arithEq_of_rows h (row := 680) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_7
  simp only [← c42813, ← c42814, k2, ← c42815] at e_680_7
  have hr := e_680_7
  simp only [e_680_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 680 39) = bselect (a (.wire 20 59)) (a (.virt 200)) (a (.virt 4169)) := by
  have c42816 := (publicBatchWrapper89_copies1338 a h).1
  have c42817 := (publicBatchWrapper89_copies1338 a h).2.1
  have c42818 := (publicBatchWrapper89_copies1338 a h).2.2.1
  have c42819 := (publicBatchWrapper89_copies1338 a h).2.2.2.1
  have c42820 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.1
  have c42821 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_680_8 := arithEq_of_rows h (row := 680) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_8
  simp only [← c42816, ← c42817, ← c42818] at e_680_8
  have e_680_9 := arithEq_of_rows h (row := 680) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_9
  simp only [← c42819, ← c42820, k2, ← c42821] at e_680_9
  have hr := e_680_9
  simp only [e_680_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 680 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4210)) := by
  have c42822 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.1
  have c42823 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.1
  have c42824 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.1
  have c42825 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.1
  have c42826 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42827 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_680_10 := arithEq_of_rows h (row := 680) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_10
  simp only [← c42822, ← c42823, ← c42824] at e_680_10
  have e_680_11 := arithEq_of_rows h (row := 680) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_11
  simp only [← c42825, ← c42826, k2, ← c42827] at e_680_11
  have hr := e_680_11
  simp only [e_680_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 680 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4211)) := by
  have c42828 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42829 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42830 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42831 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42832 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42833 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_680_12 := arithEq_of_rows h (row := 680) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_12
  simp only [← c42828, ← c42829, ← c42830] at e_680_12
  have e_680_13 := arithEq_of_rows h (row := 680) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_13
  simp only [← c42831, ← c42832, k2, ← c42833] at e_680_13
  have hr := e_680_13
  simp only [e_680_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 680 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4212)) := by
  have c42834 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42835 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42836 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42837 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42838 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42839 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_680_14 := arithEq_of_rows h (row := 680) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_14
  simp only [← c42834, ← c42835, ← c42836] at e_680_14
  have e_680_15 := arithEq_of_rows h (row := 680) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_15
  simp only [← c42837, ← c42838, k2, ← c42839] at e_680_15
  have hr := e_680_15
  simp only [e_680_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 680 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4213)) := by
  have c42840 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42841 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42842 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42843 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42844 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42845 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_680_16 := arithEq_of_rows h (row := 680) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_16
  simp only [← c42840, ← c42841, ← c42842] at e_680_16
  have e_680_17 := arithEq_of_rows h (row := 680) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_17
  simp only [← c42843, ← c42844, k2, ← c42845] at e_680_17
  have hr := e_680_17
  simp only [e_680_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 680 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4214)) := by
  have c42846 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42847 := (publicBatchWrapper89_copies1338 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42848 := (publicBatchWrapper89_copies1339 a h).1
  have c42849 := (publicBatchWrapper89_copies1339 a h).2.1
  have c42850 := (publicBatchWrapper89_copies1339 a h).2.2.1
  have c42851 := (publicBatchWrapper89_copies1339 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_680_18 := arithEq_of_rows h (row := 680) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_18
  simp only [← c42846, ← c42847, ← c42848] at e_680_18
  have e_680_19 := arithEq_of_rows h (row := 680) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_680_19
  simp only [← c42849, ← c42850, k2, ← c42851] at e_680_19
  have hr := e_680_19
  simp only [e_680_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 681 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4215)) := by
  have c42852 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.1
  have c42853 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.1
  have c42854 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.1
  have c42855 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.1
  have c42856 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.1
  have c42857 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_681_0 := arithEq_of_rows h (row := 681) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_0
  simp only [← c42852, ← c42853, ← c42854] at e_681_0
  have e_681_1 := arithEq_of_rows h (row := 681) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_1
  simp only [← c42855, ← c42856, k2, ← c42857] at e_681_1
  have hr := e_681_1
  simp only [e_681_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 681 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4216)) := by
  have c42858 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42859 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42860 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42861 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42862 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42863 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_681_2 := arithEq_of_rows h (row := 681) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_2
  simp only [← c42858, ← c42859, ← c42860] at e_681_2
  have e_681_3 := arithEq_of_rows h (row := 681) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_3
  simp only [← c42861, ← c42862, k2, ← c42863] at e_681_3
  have hr := e_681_3
  simp only [e_681_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 681 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4217)) := by
  have c42864 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42865 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42866 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42867 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42868 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42869 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_681_4 := arithEq_of_rows h (row := 681) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_4
  simp only [← c42864, ← c42865, ← c42866] at e_681_4
  have e_681_5 := arithEq_of_rows h (row := 681) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_5
  simp only [← c42867, ← c42868, k2, ← c42869] at e_681_5
  have hr := e_681_5
  simp only [e_681_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 681 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4218)) := by
  have c42870 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42871 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42872 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42873 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42874 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42875 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_681_6 := arithEq_of_rows h (row := 681) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_6
  simp only [← c42870, ← c42871, ← c42872] at e_681_6
  have e_681_7 := arithEq_of_rows h (row := 681) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_7
  simp only [← c42873, ← c42874, k2, ← c42875] at e_681_7
  have hr := e_681_7
  simp only [e_681_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 681 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4219)) := by
  have c42876 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42877 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42878 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42879 := (publicBatchWrapper89_copies1339 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42880 := (publicBatchWrapper89_copies1340 a h).1
  have c42881 := (publicBatchWrapper89_copies1340 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_681_8 := arithEq_of_rows h (row := 681) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_8
  simp only [← c42876, ← c42877, ← c42878] at e_681_8
  have e_681_9 := arithEq_of_rows h (row := 681) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_9
  simp only [← c42879, ← c42880, k2, ← c42881] at e_681_9
  have hr := e_681_9
  simp only [e_681_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 681 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4220)) := by
  have c42882 := (publicBatchWrapper89_copies1340 a h).2.2.1
  have c42883 := (publicBatchWrapper89_copies1340 a h).2.2.2.1
  have c42884 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.1
  have c42885 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.1
  have c42886 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.1
  have c42887 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_681_10 := arithEq_of_rows h (row := 681) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_10
  simp only [← c42882, ← c42883, ← c42884] at e_681_10
  have e_681_11 := arithEq_of_rows h (row := 681) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_11
  simp only [← c42885, ← c42886, k2, ← c42887] at e_681_11
  have hr := e_681_11
  simp only [e_681_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 681 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4221)) := by
  have c42888 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.1
  have c42889 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.1
  have c42890 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42891 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42892 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42893 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_681_12 := arithEq_of_rows h (row := 681) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_12
  simp only [← c42888, ← c42889, ← c42890] at e_681_12
  have e_681_13 := arithEq_of_rows h (row := 681) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_13
  simp only [← c42891, ← c42892, k2, ← c42893] at e_681_13
  have hr := e_681_13
  simp only [e_681_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 681 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4222)) := by
  have c42894 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42895 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42896 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42897 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42898 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42899 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_681_14 := arithEq_of_rows h (row := 681) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_14
  simp only [← c42894, ← c42895, ← c42896] at e_681_14
  have e_681_15 := arithEq_of_rows h (row := 681) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_15
  simp only [← c42897, ← c42898, k2, ← c42899] at e_681_15
  have hr := e_681_15
  simp only [e_681_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 681 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4223)) := by
  have c42900 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42901 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42902 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42903 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42904 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42905 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_681_16 := arithEq_of_rows h (row := 681) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_16
  simp only [← c42900, ← c42901, ← c42902] at e_681_16
  have e_681_17 := arithEq_of_rows h (row := 681) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_17
  simp only [← c42903, ← c42904, k2, ← c42905] at e_681_17
  have hr := e_681_17
  simp only [e_681_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 681 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4224)) := by
  have c42906 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42907 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42908 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42909 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42910 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42911 := (publicBatchWrapper89_copies1340 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_681_18 := arithEq_of_rows h (row := 681) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_18
  simp only [← c42906, ← c42907, ← c42908] at e_681_18
  have e_681_19 := arithEq_of_rows h (row := 681) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_681_19
  simp only [← c42909, ← c42910, k2, ← c42911] at e_681_19
  have hr := e_681_19
  simp only [e_681_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 682 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4225)) := by
  have c42912 := (publicBatchWrapper89_copies1341 a h).1
  have c42913 := (publicBatchWrapper89_copies1341 a h).2.1
  have c42914 := (publicBatchWrapper89_copies1341 a h).2.2.1
  have c42915 := (publicBatchWrapper89_copies1341 a h).2.2.2.1
  have c42916 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.1
  have c42917 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_682_0 := arithEq_of_rows h (row := 682) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_0
  simp only [← c42912, ← c42913, ← c42914] at e_682_0
  have e_682_1 := arithEq_of_rows h (row := 682) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_1
  simp only [← c42915, ← c42916, k2, ← c42917] at e_682_1
  have hr := e_682_1
  simp only [e_682_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 682 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4226)) := by
  have c42918 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.1
  have c42919 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.1
  have c42920 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.1
  have c42921 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.1
  have c42922 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42923 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_682_2 := arithEq_of_rows h (row := 682) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_2
  simp only [← c42918, ← c42919, ← c42920] at e_682_2
  have e_682_3 := arithEq_of_rows h (row := 682) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_3
  simp only [← c42921, ← c42922, k2, ← c42923] at e_682_3
  have hr := e_682_3
  simp only [e_682_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 682 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4227)) := by
  have c42924 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42925 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42926 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42927 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42928 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42929 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_682_4 := arithEq_of_rows h (row := 682) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_4
  simp only [← c42924, ← c42925, ← c42926] at e_682_4
  have e_682_5 := arithEq_of_rows h (row := 682) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_5
  simp only [← c42927, ← c42928, k2, ← c42929] at e_682_5
  have hr := e_682_5
  simp only [e_682_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 682 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4228)) := by
  have c42930 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42931 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42932 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42933 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42934 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42935 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_682_6 := arithEq_of_rows h (row := 682) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_6
  simp only [← c42930, ← c42931, ← c42932] at e_682_6
  have e_682_7 := arithEq_of_rows h (row := 682) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_7
  simp only [← c42933, ← c42934, k2, ← c42935] at e_682_7
  have hr := e_682_7
  simp only [e_682_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 682 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4229)) := by
  have c42936 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42937 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42938 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42939 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42940 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42941 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_682_8 := arithEq_of_rows h (row := 682) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_8
  simp only [← c42936, ← c42937, ← c42938] at e_682_8
  have e_682_9 := arithEq_of_rows h (row := 682) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_9
  simp only [← c42939, ← c42940, k2, ← c42941] at e_682_9
  have hr := e_682_9
  simp only [e_682_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 682 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4230)) := by
  have c42942 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42943 := (publicBatchWrapper89_copies1341 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42944 := (publicBatchWrapper89_copies1342 a h).1
  have c42945 := (publicBatchWrapper89_copies1342 a h).2.1
  have c42946 := (publicBatchWrapper89_copies1342 a h).2.2.1
  have c42947 := (publicBatchWrapper89_copies1342 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_682_10 := arithEq_of_rows h (row := 682) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_10
  simp only [← c42942, ← c42943, ← c42944] at e_682_10
  have e_682_11 := arithEq_of_rows h (row := 682) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_11
  simp only [← c42945, ← c42946, k2, ← c42947] at e_682_11
  have hr := e_682_11
  simp only [e_682_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 682 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4231)) := by
  have c42948 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.1
  have c42949 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.1
  have c42950 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.1
  have c42951 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.1
  have c42952 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.1
  have c42953 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_682_12 := arithEq_of_rows h (row := 682) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_12
  simp only [← c42948, ← c42949, ← c42950] at e_682_12
  have e_682_13 := arithEq_of_rows h (row := 682) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_13
  simp only [← c42951, ← c42952, k2, ← c42953] at e_682_13
  have hr := e_682_13
  simp only [e_682_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 682 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4232)) := by
  have c42954 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42955 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42956 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42957 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42958 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42959 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_682_14 := arithEq_of_rows h (row := 682) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_14
  simp only [← c42954, ← c42955, ← c42956] at e_682_14
  have e_682_15 := arithEq_of_rows h (row := 682) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_15
  simp only [← c42957, ← c42958, k2, ← c42959] at e_682_15
  have hr := e_682_15
  simp only [e_682_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 682 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4233)) := by
  have c42960 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42961 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42962 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42963 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42964 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42965 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_682_16 := arithEq_of_rows h (row := 682) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_16
  simp only [← c42960, ← c42961, ← c42962] at e_682_16
  have e_682_17 := arithEq_of_rows h (row := 682) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_17
  simp only [← c42963, ← c42964, k2, ← c42965] at e_682_17
  have hr := e_682_17
  simp only [e_682_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 682 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4234)) := by
  have c42966 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42967 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42968 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42969 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42970 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42971 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_682_18 := arithEq_of_rows h (row := 682) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_18
  simp only [← c42966, ← c42967, ← c42968] at e_682_18
  have e_682_19 := arithEq_of_rows h (row := 682) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_682_19
  simp only [← c42969, ← c42970, k2, ← c42971] at e_682_19
  have hr := e_682_19
  simp only [e_682_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 683 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4235)) := by
  have c42972 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42973 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42974 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42975 := (publicBatchWrapper89_copies1342 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c42976 := (publicBatchWrapper89_copies1343 a h).1
  have c42977 := (publicBatchWrapper89_copies1343 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_683_0 := arithEq_of_rows h (row := 683) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_0
  simp only [← c42972, ← c42973, ← c42974] at e_683_0
  have e_683_1 := arithEq_of_rows h (row := 683) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_1
  simp only [← c42975, ← c42976, k2, ← c42977] at e_683_1
  have hr := e_683_1
  simp only [e_683_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 683 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4236)) := by
  have c42978 := (publicBatchWrapper89_copies1343 a h).2.2.1
  have c42979 := (publicBatchWrapper89_copies1343 a h).2.2.2.1
  have c42980 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.1
  have c42981 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.1
  have c42982 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.1
  have c42983 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_683_2 := arithEq_of_rows h (row := 683) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_2
  simp only [← c42978, ← c42979, ← c42980] at e_683_2
  have e_683_3 := arithEq_of_rows h (row := 683) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_3
  simp only [← c42981, ← c42982, k2, ← c42983] at e_683_3
  have hr := e_683_3
  simp only [e_683_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 683 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4237)) := by
  have c42984 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.1
  have c42985 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.1
  have c42986 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.1
  have c42987 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c42988 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42989 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_683_4 := arithEq_of_rows h (row := 683) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_4
  simp only [← c42984, ← c42985, ← c42986] at e_683_4
  have e_683_5 := arithEq_of_rows h (row := 683) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_5
  simp only [← c42987, ← c42988, k2, ← c42989] at e_683_5
  have hr := e_683_5
  simp only [e_683_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 683 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4238)) := by
  have c42990 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42991 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42992 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42993 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42994 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42995 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_683_6 := arithEq_of_rows h (row := 683) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_6
  simp only [← c42990, ← c42991, ← c42992] at e_683_6
  have e_683_7 := arithEq_of_rows h (row := 683) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_7
  simp only [← c42993, ← c42994, k2, ← c42995] at e_683_7
  have hr := e_683_7
  simp only [e_683_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 683 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4239)) := by
  have c42996 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42997 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42998 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c42999 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43000 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43001 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_683_8 := arithEq_of_rows h (row := 683) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_8
  simp only [← c42996, ← c42997, ← c42998] at e_683_8
  have e_683_9 := arithEq_of_rows h (row := 683) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_9
  simp only [← c42999, ← c43000, k2, ← c43001] at e_683_9
  have hr := e_683_9
  simp only [e_683_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 683 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4240)) := by
  have c43002 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43003 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43004 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43005 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43006 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43007 := (publicBatchWrapper89_copies1343 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_683_10 := arithEq_of_rows h (row := 683) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_10
  simp only [← c43002, ← c43003, ← c43004] at e_683_10
  have e_683_11 := arithEq_of_rows h (row := 683) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_11
  simp only [← c43005, ← c43006, k2, ← c43007] at e_683_11
  have hr := e_683_11
  simp only [e_683_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 683 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4241)) := by
  have c43008 := (publicBatchWrapper89_copies1344 a h).1
  have c43009 := (publicBatchWrapper89_copies1344 a h).2.1
  have c43010 := (publicBatchWrapper89_copies1344 a h).2.2.1
  have c43011 := (publicBatchWrapper89_copies1344 a h).2.2.2.1
  have c43012 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.1
  have c43013 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_683_12 := arithEq_of_rows h (row := 683) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_12
  simp only [← c43008, ← c43009, ← c43010] at e_683_12
  have e_683_13 := arithEq_of_rows h (row := 683) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_13
  simp only [← c43011, ← c43012, k2, ← c43013] at e_683_13
  have hr := e_683_13
  simp only [e_683_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 683 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4242)) := by
  have c43014 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.1
  have c43015 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.1
  have c43016 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.1
  have c43017 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.1
  have c43018 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43019 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_683_14 := arithEq_of_rows h (row := 683) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_14
  simp only [← c43014, ← c43015, ← c43016] at e_683_14
  have e_683_15 := arithEq_of_rows h (row := 683) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_15
  simp only [← c43017, ← c43018, k2, ← c43019] at e_683_15
  have hr := e_683_15
  simp only [e_683_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 683 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4243)) := by
  have c43020 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43021 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43022 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43023 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43024 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43025 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_683_16 := arithEq_of_rows h (row := 683) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_16
  simp only [← c43020, ← c43021, ← c43022] at e_683_16
  have e_683_17 := arithEq_of_rows h (row := 683) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_17
  simp only [← c43023, ← c43024, k2, ← c43025] at e_683_17
  have hr := e_683_17
  simp only [e_683_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 683 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4244)) := by
  have c43026 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43027 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43028 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43029 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43030 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43031 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_683_18 := arithEq_of_rows h (row := 683) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_18
  simp only [← c43026, ← c43027, ← c43028] at e_683_18
  have e_683_19 := arithEq_of_rows h (row := 683) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_683_19
  simp only [← c43029, ← c43030, k2, ← c43031] at e_683_19
  have hr := e_683_19
  simp only [e_683_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 684 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4245)) := by
  have c43032 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43033 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43034 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43035 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43036 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43037 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_684_0 := arithEq_of_rows h (row := 684) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_0
  simp only [← c43032, ← c43033, ← c43034] at e_684_0
  have e_684_1 := arithEq_of_rows h (row := 684) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_1
  simp only [← c43035, ← c43036, k2, ← c43037] at e_684_1
  have hr := e_684_1
  simp only [e_684_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 684 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4246)) := by
  have c43038 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43039 := (publicBatchWrapper89_copies1344 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43040 := (publicBatchWrapper89_copies1345 a h).1
  have c43041 := (publicBatchWrapper89_copies1345 a h).2.1
  have c43042 := (publicBatchWrapper89_copies1345 a h).2.2.1
  have c43043 := (publicBatchWrapper89_copies1345 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_684_2 := arithEq_of_rows h (row := 684) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_2
  simp only [← c43038, ← c43039, ← c43040] at e_684_2
  have e_684_3 := arithEq_of_rows h (row := 684) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_3
  simp only [← c43041, ← c43042, k2, ← c43043] at e_684_3
  have hr := e_684_3
  simp only [e_684_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 684 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4247)) := by
  have c43044 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.1
  have c43045 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.1
  have c43046 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.1
  have c43047 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.1
  have c43048 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.1
  have c43049 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_684_4 := arithEq_of_rows h (row := 684) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_4
  simp only [← c43044, ← c43045, ← c43046] at e_684_4
  have e_684_5 := arithEq_of_rows h (row := 684) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_5
  simp only [← c43047, ← c43048, k2, ← c43049] at e_684_5
  have hr := e_684_5
  simp only [e_684_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 684 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4248)) := by
  have c43050 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43051 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43052 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43053 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43054 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43055 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_684_6 := arithEq_of_rows h (row := 684) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_6
  simp only [← c43050, ← c43051, ← c43052] at e_684_6
  have e_684_7 := arithEq_of_rows h (row := 684) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_7
  simp only [← c43053, ← c43054, k2, ← c43055] at e_684_7
  have hr := e_684_7
  simp only [e_684_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 684 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4249)) := by
  have c43056 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43057 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43058 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43059 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43060 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43061 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_684_8 := arithEq_of_rows h (row := 684) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_8
  simp only [← c43056, ← c43057, ← c43058] at e_684_8
  have e_684_9 := arithEq_of_rows h (row := 684) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_9
  simp only [← c43059, ← c43060, k2, ← c43061] at e_684_9
  have hr := e_684_9
  simp only [e_684_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 684 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4250)) := by
  have c43062 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43063 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43064 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43065 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43066 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43067 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_684_10 := arithEq_of_rows h (row := 684) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_10
  simp only [← c43062, ← c43063, ← c43064] at e_684_10
  have e_684_11 := arithEq_of_rows h (row := 684) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_11
  simp only [← c43065, ← c43066, k2, ← c43067] at e_684_11
  have hr := e_684_11
  simp only [e_684_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 684 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4251)) := by
  have c43068 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43069 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43070 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43071 := (publicBatchWrapper89_copies1345 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43072 := (publicBatchWrapper89_copies1346 a h).1
  have c43073 := (publicBatchWrapper89_copies1346 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_684_12 := arithEq_of_rows h (row := 684) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_12
  simp only [← c43068, ← c43069, ← c43070] at e_684_12
  have e_684_13 := arithEq_of_rows h (row := 684) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_13
  simp only [← c43071, ← c43072, k2, ← c43073] at e_684_13
  have hr := e_684_13
  simp only [e_684_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 684 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4252)) := by
  have c43074 := (publicBatchWrapper89_copies1346 a h).2.2.1
  have c43075 := (publicBatchWrapper89_copies1346 a h).2.2.2.1
  have c43076 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.1
  have c43077 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.1
  have c43078 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.1
  have c43079 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_684_14 := arithEq_of_rows h (row := 684) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_14
  simp only [← c43074, ← c43075, ← c43076] at e_684_14
  have e_684_15 := arithEq_of_rows h (row := 684) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_15
  simp only [← c43077, ← c43078, k2, ← c43079] at e_684_15
  have hr := e_684_15
  simp only [e_684_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 684 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4253)) := by
  have c43080 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.1
  have c43081 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.1
  have c43082 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43083 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43084 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43085 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_684_16 := arithEq_of_rows h (row := 684) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_16
  simp only [← c43080, ← c43081, ← c43082] at e_684_16
  have e_684_17 := arithEq_of_rows h (row := 684) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_17
  simp only [← c43083, ← c43084, k2, ← c43085] at e_684_17
  have hr := e_684_17
  simp only [e_684_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 684 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4254)) := by
  have c43086 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43087 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43088 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43089 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43090 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43091 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_684_18 := arithEq_of_rows h (row := 684) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_18
  simp only [← c43086, ← c43087, ← c43088] at e_684_18
  have e_684_19 := arithEq_of_rows h (row := 684) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_684_19
  simp only [← c43089, ← c43090, k2, ← c43091] at e_684_19
  have hr := e_684_19
  simp only [e_684_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 685 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4255)) := by
  have c43092 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43093 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43094 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43095 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43096 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43097 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_685_0 := arithEq_of_rows h (row := 685) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_0
  simp only [← c43092, ← c43093, ← c43094] at e_685_0
  have e_685_1 := arithEq_of_rows h (row := 685) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_1
  simp only [← c43095, ← c43096, k2, ← c43097] at e_685_1
  have hr := e_685_1
  simp only [e_685_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 685 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4256)) := by
  have c43098 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43099 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43100 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43101 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43102 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43103 := (publicBatchWrapper89_copies1346 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_685_2 := arithEq_of_rows h (row := 685) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_2
  simp only [← c43098, ← c43099, ← c43100] at e_685_2
  have e_685_3 := arithEq_of_rows h (row := 685) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_3
  simp only [← c43101, ← c43102, k2, ← c43103] at e_685_3
  have hr := e_685_3
  simp only [e_685_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 685 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4257)) := by
  have c43104 := (publicBatchWrapper89_copies1347 a h).1
  have c43105 := (publicBatchWrapper89_copies1347 a h).2.1
  have c43106 := (publicBatchWrapper89_copies1347 a h).2.2.1
  have c43107 := (publicBatchWrapper89_copies1347 a h).2.2.2.1
  have c43108 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.1
  have c43109 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_685_4 := arithEq_of_rows h (row := 685) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_4
  simp only [← c43104, ← c43105, ← c43106] at e_685_4
  have e_685_5 := arithEq_of_rows h (row := 685) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_5
  simp only [← c43107, ← c43108, k2, ← c43109] at e_685_5
  have hr := e_685_5
  simp only [e_685_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 685 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4258)) := by
  have c43110 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.1
  have c43111 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.1
  have c43112 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.1
  have c43113 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.1
  have c43114 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43115 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_685_6 := arithEq_of_rows h (row := 685) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_6
  simp only [← c43110, ← c43111, ← c43112] at e_685_6
  have e_685_7 := arithEq_of_rows h (row := 685) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_7
  simp only [← c43113, ← c43114, k2, ← c43115] at e_685_7
  have hr := e_685_7
  simp only [e_685_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 685 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4259)) := by
  have c43116 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43117 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43118 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43119 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43120 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43121 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_685_8 := arithEq_of_rows h (row := 685) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_8
  simp only [← c43116, ← c43117, ← c43118] at e_685_8
  have e_685_9 := arithEq_of_rows h (row := 685) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_9
  simp only [← c43119, ← c43120, k2, ← c43121] at e_685_9
  have hr := e_685_9
  simp only [e_685_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 685 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4260)) := by
  have c43122 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43123 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43124 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43125 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43126 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43127 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_685_10 := arithEq_of_rows h (row := 685) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_10
  simp only [← c43122, ← c43123, ← c43124] at e_685_10
  have e_685_11 := arithEq_of_rows h (row := 685) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_11
  simp only [← c43125, ← c43126, k2, ← c43127] at e_685_11
  have hr := e_685_11
  simp only [e_685_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 685 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4261)) := by
  have c43128 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43129 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43130 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43131 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43132 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43133 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_685_12 := arithEq_of_rows h (row := 685) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_12
  simp only [← c43128, ← c43129, ← c43130] at e_685_12
  have e_685_13 := arithEq_of_rows h (row := 685) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_13
  simp only [← c43131, ← c43132, k2, ← c43133] at e_685_13
  have hr := e_685_13
  simp only [e_685_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 685 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4262)) := by
  have c43134 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43135 := (publicBatchWrapper89_copies1347 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43136 := (publicBatchWrapper89_copies1348 a h).1
  have c43137 := (publicBatchWrapper89_copies1348 a h).2.1
  have c43138 := (publicBatchWrapper89_copies1348 a h).2.2.1
  have c43139 := (publicBatchWrapper89_copies1348 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_685_14 := arithEq_of_rows h (row := 685) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_14
  simp only [← c43134, ← c43135, ← c43136] at e_685_14
  have e_685_15 := arithEq_of_rows h (row := 685) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_15
  simp only [← c43137, ← c43138, k2, ← c43139] at e_685_15
  have hr := e_685_15
  simp only [e_685_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 685 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4263)) := by
  have c43140 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.1
  have c43141 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.1
  have c43142 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.1
  have c43143 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.1
  have c43144 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.1
  have c43145 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_685_16 := arithEq_of_rows h (row := 685) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_16
  simp only [← c43140, ← c43141, ← c43142] at e_685_16
  have e_685_17 := arithEq_of_rows h (row := 685) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_17
  simp only [← c43143, ← c43144, k2, ← c43145] at e_685_17
  have hr := e_685_17
  simp only [e_685_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 685 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4264)) := by
  have c43146 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43147 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43148 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43149 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43150 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43151 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_685_18 := arithEq_of_rows h (row := 685) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_18
  simp only [← c43146, ← c43147, ← c43148] at e_685_18
  have e_685_19 := arithEq_of_rows h (row := 685) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_685_19
  simp only [← c43149, ← c43150, k2, ← c43151] at e_685_19
  have hr := e_685_19
  simp only [e_685_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 686 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4265)) := by
  have c43152 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43153 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43154 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43155 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43156 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43157 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_686_0 := arithEq_of_rows h (row := 686) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_0
  simp only [← c43152, ← c43153, ← c43154] at e_686_0
  have e_686_1 := arithEq_of_rows h (row := 686) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_1
  simp only [← c43155, ← c43156, k2, ← c43157] at e_686_1
  have hr := e_686_1
  simp only [e_686_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 686 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4266)) := by
  have c43158 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43159 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43160 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43161 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43162 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43163 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_686_2 := arithEq_of_rows h (row := 686) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_2
  simp only [← c43158, ← c43159, ← c43160] at e_686_2
  have e_686_3 := arithEq_of_rows h (row := 686) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_3
  simp only [← c43161, ← c43162, k2, ← c43163] at e_686_3
  have hr := e_686_3
  simp only [e_686_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 686 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4267)) := by
  have c43164 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43165 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43166 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43167 := (publicBatchWrapper89_copies1348 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43168 := (publicBatchWrapper89_copies1349 a h).1
  have c43169 := (publicBatchWrapper89_copies1349 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_686_4 := arithEq_of_rows h (row := 686) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_4
  simp only [← c43164, ← c43165, ← c43166] at e_686_4
  have e_686_5 := arithEq_of_rows h (row := 686) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_5
  simp only [← c43167, ← c43168, k2, ← c43169] at e_686_5
  have hr := e_686_5
  simp only [e_686_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 686 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4268)) := by
  have c43170 := (publicBatchWrapper89_copies1349 a h).2.2.1
  have c43171 := (publicBatchWrapper89_copies1349 a h).2.2.2.1
  have c43172 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.1
  have c43173 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.1
  have c43174 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.1
  have c43175 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_686_6 := arithEq_of_rows h (row := 686) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_6
  simp only [← c43170, ← c43171, ← c43172] at e_686_6
  have e_686_7 := arithEq_of_rows h (row := 686) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_7
  simp only [← c43173, ← c43174, k2, ← c43175] at e_686_7
  have hr := e_686_7
  simp only [e_686_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 686 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4269)) := by
  have c43176 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.1
  have c43177 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.1
  have c43178 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43179 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43180 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43181 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_686_8 := arithEq_of_rows h (row := 686) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_8
  simp only [← c43176, ← c43177, ← c43178] at e_686_8
  have e_686_9 := arithEq_of_rows h (row := 686) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_9
  simp only [← c43179, ← c43180, k2, ← c43181] at e_686_9
  have hr := e_686_9
  simp only [e_686_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 686 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4270)) := by
  have c43182 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43183 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43184 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43185 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43186 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43187 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_686_10 := arithEq_of_rows h (row := 686) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_10
  simp only [← c43182, ← c43183, ← c43184] at e_686_10
  have e_686_11 := arithEq_of_rows h (row := 686) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_11
  simp only [← c43185, ← c43186, k2, ← c43187] at e_686_11
  have hr := e_686_11
  simp only [e_686_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 686 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4271)) := by
  have c43188 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43189 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43190 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43191 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43192 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43193 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_686_12 := arithEq_of_rows h (row := 686) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_12
  simp only [← c43188, ← c43189, ← c43190] at e_686_12
  have e_686_13 := arithEq_of_rows h (row := 686) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_13
  simp only [← c43191, ← c43192, k2, ← c43193] at e_686_13
  have hr := e_686_13
  simp only [e_686_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 686 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4272)) := by
  have c43194 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43195 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43196 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43197 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43198 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43199 := (publicBatchWrapper89_copies1349 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_686_14 := arithEq_of_rows h (row := 686) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_14
  simp only [← c43194, ← c43195, ← c43196] at e_686_14
  have e_686_15 := arithEq_of_rows h (row := 686) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_15
  simp only [← c43197, ← c43198, k2, ← c43199] at e_686_15
  have hr := e_686_15
  simp only [e_686_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 686 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4273)) := by
  have c43200 := (publicBatchWrapper89_copies1350 a h).1
  have c43201 := (publicBatchWrapper89_copies1350 a h).2.1
  have c43202 := (publicBatchWrapper89_copies1350 a h).2.2.1
  have c43203 := (publicBatchWrapper89_copies1350 a h).2.2.2.1
  have c43204 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.1
  have c43205 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_686_16 := arithEq_of_rows h (row := 686) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_16
  simp only [← c43200, ← c43201, ← c43202] at e_686_16
  have e_686_17 := arithEq_of_rows h (row := 686) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_17
  simp only [← c43203, ← c43204, k2, ← c43205] at e_686_17
  have hr := e_686_17
  simp only [e_686_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 686 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4274)) := by
  have c43206 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.1
  have c43207 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.1
  have c43208 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.1
  have c43209 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.1
  have c43210 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43211 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_686_18 := arithEq_of_rows h (row := 686) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_18
  simp only [← c43206, ← c43207, ← c43208] at e_686_18
  have e_686_19 := arithEq_of_rows h (row := 686) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_686_19
  simp only [← c43209, ← c43210, k2, ← c43211] at e_686_19
  have hr := e_686_19
  simp only [e_686_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 687 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4275)) := by
  have c43212 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43213 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43214 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43215 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43216 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43217 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_687_0 := arithEq_of_rows h (row := 687) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_0
  simp only [← c43212, ← c43213, ← c43214] at e_687_0
  have e_687_1 := arithEq_of_rows h (row := 687) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_1
  simp only [← c43215, ← c43216, k2, ← c43217] at e_687_1
  have hr := e_687_1
  simp only [e_687_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 687 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4276)) := by
  have c43218 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43219 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43220 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43221 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43222 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43223 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_687_2 := arithEq_of_rows h (row := 687) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_2
  simp only [← c43218, ← c43219, ← c43220] at e_687_2
  have e_687_3 := arithEq_of_rows h (row := 687) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_3
  simp only [← c43221, ← c43222, k2, ← c43223] at e_687_3
  have hr := e_687_3
  simp only [e_687_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 687 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4277)) := by
  have c43224 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43225 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43226 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43227 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43228 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43229 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_687_4 := arithEq_of_rows h (row := 687) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_4
  simp only [← c43224, ← c43225, ← c43226] at e_687_4
  have e_687_5 := arithEq_of_rows h (row := 687) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_5
  simp only [← c43227, ← c43228, k2, ← c43229] at e_687_5
  have hr := e_687_5
  simp only [e_687_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 687 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4278)) := by
  have c43230 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43231 := (publicBatchWrapper89_copies1350 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43232 := (publicBatchWrapper89_copies1351 a h).1
  have c43233 := (publicBatchWrapper89_copies1351 a h).2.1
  have c43234 := (publicBatchWrapper89_copies1351 a h).2.2.1
  have c43235 := (publicBatchWrapper89_copies1351 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_687_6 := arithEq_of_rows h (row := 687) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_6
  simp only [← c43230, ← c43231, ← c43232] at e_687_6
  have e_687_7 := arithEq_of_rows h (row := 687) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_7
  simp only [← c43233, ← c43234, k2, ← c43235] at e_687_7
  have hr := e_687_7
  simp only [e_687_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 687 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4279)) := by
  have c43236 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.1
  have c43237 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.1
  have c43238 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.1
  have c43239 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.1
  have c43240 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.1
  have c43241 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_687_8 := arithEq_of_rows h (row := 687) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_8
  simp only [← c43236, ← c43237, ← c43238] at e_687_8
  have e_687_9 := arithEq_of_rows h (row := 687) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_9
  simp only [← c43239, ← c43240, k2, ← c43241] at e_687_9
  have hr := e_687_9
  simp only [e_687_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 687 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4280)) := by
  have c43242 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43243 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43244 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43245 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43246 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43247 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_687_10 := arithEq_of_rows h (row := 687) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_10
  simp only [← c43242, ← c43243, ← c43244] at e_687_10
  have e_687_11 := arithEq_of_rows h (row := 687) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_11
  simp only [← c43245, ← c43246, k2, ← c43247] at e_687_11
  have hr := e_687_11
  simp only [e_687_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 687 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4281)) := by
  have c43248 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43249 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43250 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43251 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43252 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43253 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_687_12 := arithEq_of_rows h (row := 687) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_12
  simp only [← c43248, ← c43249, ← c43250] at e_687_12
  have e_687_13 := arithEq_of_rows h (row := 687) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_13
  simp only [← c43251, ← c43252, k2, ← c43253] at e_687_13
  have hr := e_687_13
  simp only [e_687_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 687 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4282)) := by
  have c43254 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43255 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43256 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43257 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43258 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43259 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_687_14 := arithEq_of_rows h (row := 687) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_14
  simp only [← c43254, ← c43255, ← c43256] at e_687_14
  have e_687_15 := arithEq_of_rows h (row := 687) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_15
  simp only [← c43257, ← c43258, k2, ← c43259] at e_687_15
  have hr := e_687_15
  simp only [e_687_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 687 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4283)) := by
  have c43260 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43261 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43262 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43263 := (publicBatchWrapper89_copies1351 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43264 := (publicBatchWrapper89_copies1352 a h).1
  have c43265 := (publicBatchWrapper89_copies1352 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_687_16 := arithEq_of_rows h (row := 687) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_16
  simp only [← c43260, ← c43261, ← c43262] at e_687_16
  have e_687_17 := arithEq_of_rows h (row := 687) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_17
  simp only [← c43263, ← c43264, k2, ← c43265] at e_687_17
  have hr := e_687_17
  simp only [e_687_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 687 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4284)) := by
  have c43266 := (publicBatchWrapper89_copies1352 a h).2.2.1
  have c43267 := (publicBatchWrapper89_copies1352 a h).2.2.2.1
  have c43268 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.1
  have c43269 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.1
  have c43270 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.1
  have c43271 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_687_18 := arithEq_of_rows h (row := 687) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_18
  simp only [← c43266, ← c43267, ← c43268] at e_687_18
  have e_687_19 := arithEq_of_rows h (row := 687) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_687_19
  simp only [← c43269, ← c43270, k2, ← c43271] at e_687_19
  have hr := e_687_19
  simp only [e_687_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 688 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4285)) := by
  have c43272 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.1
  have c43273 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.1
  have c43274 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43275 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43276 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43277 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_688_0 := arithEq_of_rows h (row := 688) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_0
  simp only [← c43272, ← c43273, ← c43274] at e_688_0
  have e_688_1 := arithEq_of_rows h (row := 688) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_1
  simp only [← c43275, ← c43276, k2, ← c43277] at e_688_1
  have hr := e_688_1
  simp only [e_688_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 688 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4286)) := by
  have c43278 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43279 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43280 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43281 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43282 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43283 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_688_2 := arithEq_of_rows h (row := 688) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_2
  simp only [← c43278, ← c43279, ← c43280] at e_688_2
  have e_688_3 := arithEq_of_rows h (row := 688) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_3
  simp only [← c43281, ← c43282, k2, ← c43283] at e_688_3
  have hr := e_688_3
  simp only [e_688_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 688 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4287)) := by
  have c43284 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43285 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43286 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43287 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43288 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43289 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_688_4 := arithEq_of_rows h (row := 688) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_4
  simp only [← c43284, ← c43285, ← c43286] at e_688_4
  have e_688_5 := arithEq_of_rows h (row := 688) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_5
  simp only [← c43287, ← c43288, k2, ← c43289] at e_688_5
  have hr := e_688_5
  simp only [e_688_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 688 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4288)) := by
  have c43290 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43291 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43292 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43293 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43294 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43295 := (publicBatchWrapper89_copies1352 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_688_6 := arithEq_of_rows h (row := 688) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_6
  simp only [← c43290, ← c43291, ← c43292] at e_688_6
  have e_688_7 := arithEq_of_rows h (row := 688) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_7
  simp only [← c43293, ← c43294, k2, ← c43295] at e_688_7
  have hr := e_688_7
  simp only [e_688_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 688 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4289)) := by
  have c43296 := (publicBatchWrapper89_copies1353 a h).1
  have c43297 := (publicBatchWrapper89_copies1353 a h).2.1
  have c43298 := (publicBatchWrapper89_copies1353 a h).2.2.1
  have c43299 := (publicBatchWrapper89_copies1353 a h).2.2.2.1
  have c43300 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.1
  have c43301 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_688_8 := arithEq_of_rows h (row := 688) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_8
  simp only [← c43296, ← c43297, ← c43298] at e_688_8
  have e_688_9 := arithEq_of_rows h (row := 688) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_9
  simp only [← c43299, ← c43300, k2, ← c43301] at e_688_9
  have hr := e_688_9
  simp only [e_688_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 688 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4290)) := by
  have c43302 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.1
  have c43303 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.1
  have c43304 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.1
  have c43305 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.1
  have c43306 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43307 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_688_10 := arithEq_of_rows h (row := 688) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_10
  simp only [← c43302, ← c43303, ← c43304] at e_688_10
  have e_688_11 := arithEq_of_rows h (row := 688) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_11
  simp only [← c43305, ← c43306, k2, ← c43307] at e_688_11
  have hr := e_688_11
  simp only [e_688_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 688 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4291)) := by
  have c43308 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43309 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43310 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43311 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43312 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43313 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_688_12 := arithEq_of_rows h (row := 688) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_12
  simp only [← c43308, ← c43309, ← c43310] at e_688_12
  have e_688_13 := arithEq_of_rows h (row := 688) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_13
  simp only [← c43311, ← c43312, k2, ← c43313] at e_688_13
  have hr := e_688_13
  simp only [e_688_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 688 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4292)) := by
  have c43314 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43315 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43316 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43317 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43318 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43319 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_688_14 := arithEq_of_rows h (row := 688) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_14
  simp only [← c43314, ← c43315, ← c43316] at e_688_14
  have e_688_15 := arithEq_of_rows h (row := 688) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_15
  simp only [← c43317, ← c43318, k2, ← c43319] at e_688_15
  have hr := e_688_15
  simp only [e_688_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 688 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4293)) := by
  have c43320 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43321 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43322 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43323 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43324 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43325 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_688_16 := arithEq_of_rows h (row := 688) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_16
  simp only [← c43320, ← c43321, ← c43322] at e_688_16
  have e_688_17 := arithEq_of_rows h (row := 688) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_17
  simp only [← c43323, ← c43324, k2, ← c43325] at e_688_17
  have hr := e_688_17
  simp only [e_688_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 688 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4294)) := by
  have c43326 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43327 := (publicBatchWrapper89_copies1353 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43328 := (publicBatchWrapper89_copies1354 a h).1
  have c43329 := (publicBatchWrapper89_copies1354 a h).2.1
  have c43330 := (publicBatchWrapper89_copies1354 a h).2.2.1
  have c43331 := (publicBatchWrapper89_copies1354 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_688_18 := arithEq_of_rows h (row := 688) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_18
  simp only [← c43326, ← c43327, ← c43328] at e_688_18
  have e_688_19 := arithEq_of_rows h (row := 688) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_688_19
  simp only [← c43329, ← c43330, k2, ← c43331] at e_688_19
  have hr := e_688_19
  simp only [e_688_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 689 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4295)) := by
  have c43332 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.1
  have c43333 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.1
  have c43334 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.1
  have c43335 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.1
  have c43336 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.1
  have c43337 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_689_0 := arithEq_of_rows h (row := 689) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_0
  simp only [← c43332, ← c43333, ← c43334] at e_689_0
  have e_689_1 := arithEq_of_rows h (row := 689) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_1
  simp only [← c43335, ← c43336, k2, ← c43337] at e_689_1
  have hr := e_689_1
  simp only [e_689_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 689 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4296)) := by
  have c43338 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43339 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43340 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43341 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43342 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43343 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_689_2 := arithEq_of_rows h (row := 689) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_2
  simp only [← c43338, ← c43339, ← c43340] at e_689_2
  have e_689_3 := arithEq_of_rows h (row := 689) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_3
  simp only [← c43341, ← c43342, k2, ← c43343] at e_689_3
  have hr := e_689_3
  simp only [e_689_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 689 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4297)) := by
  have c43344 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43345 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43346 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43347 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43348 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43349 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_689_4 := arithEq_of_rows h (row := 689) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_4
  simp only [← c43344, ← c43345, ← c43346] at e_689_4
  have e_689_5 := arithEq_of_rows h (row := 689) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_5
  simp only [← c43347, ← c43348, k2, ← c43349] at e_689_5
  have hr := e_689_5
  simp only [e_689_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 689 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4298)) := by
  have c43350 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43351 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43352 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43353 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43354 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43355 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_689_6 := arithEq_of_rows h (row := 689) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_6
  simp only [← c43350, ← c43351, ← c43352] at e_689_6
  have e_689_7 := arithEq_of_rows h (row := 689) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_7
  simp only [← c43353, ← c43354, k2, ← c43355] at e_689_7
  have hr := e_689_7
  simp only [e_689_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 689 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4299)) := by
  have c43356 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43357 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43358 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43359 := (publicBatchWrapper89_copies1354 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43360 := (publicBatchWrapper89_copies1355 a h).1
  have c43361 := (publicBatchWrapper89_copies1355 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_689_8 := arithEq_of_rows h (row := 689) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_8
  simp only [← c43356, ← c43357, ← c43358] at e_689_8
  have e_689_9 := arithEq_of_rows h (row := 689) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_9
  simp only [← c43359, ← c43360, k2, ← c43361] at e_689_9
  have hr := e_689_9
  simp only [e_689_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 689 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4300)) := by
  have c43362 := (publicBatchWrapper89_copies1355 a h).2.2.1
  have c43363 := (publicBatchWrapper89_copies1355 a h).2.2.2.1
  have c43364 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.1
  have c43365 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.1
  have c43366 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.1
  have c43367 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_689_10 := arithEq_of_rows h (row := 689) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_10
  simp only [← c43362, ← c43363, ← c43364] at e_689_10
  have e_689_11 := arithEq_of_rows h (row := 689) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_11
  simp only [← c43365, ← c43366, k2, ← c43367] at e_689_11
  have hr := e_689_11
  simp only [e_689_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 689 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4301)) := by
  have c43368 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.1
  have c43369 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.1
  have c43370 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43371 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43372 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43373 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_689_12 := arithEq_of_rows h (row := 689) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_12
  simp only [← c43368, ← c43369, ← c43370] at e_689_12
  have e_689_13 := arithEq_of_rows h (row := 689) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_13
  simp only [← c43371, ← c43372, k2, ← c43373] at e_689_13
  have hr := e_689_13
  simp only [e_689_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 689 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4302)) := by
  have c43374 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43375 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43376 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43377 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43378 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43379 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_689_14 := arithEq_of_rows h (row := 689) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_14
  simp only [← c43374, ← c43375, ← c43376] at e_689_14
  have e_689_15 := arithEq_of_rows h (row := 689) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_15
  simp only [← c43377, ← c43378, k2, ← c43379] at e_689_15
  have hr := e_689_15
  simp only [e_689_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 689 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4303)) := by
  have c43380 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43381 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43382 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43383 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43384 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43385 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_689_16 := arithEq_of_rows h (row := 689) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_16
  simp only [← c43380, ← c43381, ← c43382] at e_689_16
  have e_689_17 := arithEq_of_rows h (row := 689) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_17
  simp only [← c43383, ← c43384, k2, ← c43385] at e_689_17
  have hr := e_689_17
  simp only [e_689_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 689 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4304)) := by
  have c43386 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43387 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43388 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43389 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43390 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43391 := (publicBatchWrapper89_copies1355 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_689_18 := arithEq_of_rows h (row := 689) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_18
  simp only [← c43386, ← c43387, ← c43388] at e_689_18
  have e_689_19 := arithEq_of_rows h (row := 689) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_689_19
  simp only [← c43389, ← c43390, k2, ← c43391] at e_689_19
  have hr := e_689_19
  simp only [e_689_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 690 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4305)) := by
  have c43392 := (publicBatchWrapper89_copies1356 a h).1
  have c43393 := (publicBatchWrapper89_copies1356 a h).2.1
  have c43394 := (publicBatchWrapper89_copies1356 a h).2.2.1
  have c43395 := (publicBatchWrapper89_copies1356 a h).2.2.2.1
  have c43396 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.1
  have c43397 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_690_0 := arithEq_of_rows h (row := 690) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_0
  simp only [← c43392, ← c43393, ← c43394] at e_690_0
  have e_690_1 := arithEq_of_rows h (row := 690) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_1
  simp only [← c43395, ← c43396, k2, ← c43397] at e_690_1
  have hr := e_690_1
  simp only [e_690_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 690 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4306)) := by
  have c43398 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.1
  have c43399 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.1
  have c43400 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.1
  have c43401 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.1
  have c43402 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43403 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_690_2 := arithEq_of_rows h (row := 690) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_2
  simp only [← c43398, ← c43399, ← c43400] at e_690_2
  have e_690_3 := arithEq_of_rows h (row := 690) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_3
  simp only [← c43401, ← c43402, k2, ← c43403] at e_690_3
  have hr := e_690_3
  simp only [e_690_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 690 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4307)) := by
  have c43404 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43405 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43406 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43407 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43408 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43409 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_690_4 := arithEq_of_rows h (row := 690) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_4
  simp only [← c43404, ← c43405, ← c43406] at e_690_4
  have e_690_5 := arithEq_of_rows h (row := 690) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_5
  simp only [← c43407, ← c43408, k2, ← c43409] at e_690_5
  have hr := e_690_5
  simp only [e_690_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 690 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4308)) := by
  have c43410 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43411 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43412 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43413 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43414 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43415 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_690_6 := arithEq_of_rows h (row := 690) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_6
  simp only [← c43410, ← c43411, ← c43412] at e_690_6
  have e_690_7 := arithEq_of_rows h (row := 690) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_7
  simp only [← c43413, ← c43414, k2, ← c43415] at e_690_7
  have hr := e_690_7
  simp only [e_690_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 690 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4309)) := by
  have c43416 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43417 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43418 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43419 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43420 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43421 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_690_8 := arithEq_of_rows h (row := 690) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_8
  simp only [← c43416, ← c43417, ← c43418] at e_690_8
  have e_690_9 := arithEq_of_rows h (row := 690) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_9
  simp only [← c43419, ← c43420, k2, ← c43421] at e_690_9
  have hr := e_690_9
  simp only [e_690_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 690 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4310)) := by
  have c43422 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43423 := (publicBatchWrapper89_copies1356 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43424 := (publicBatchWrapper89_copies1357 a h).1
  have c43425 := (publicBatchWrapper89_copies1357 a h).2.1
  have c43426 := (publicBatchWrapper89_copies1357 a h).2.2.1
  have c43427 := (publicBatchWrapper89_copies1357 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_690_10 := arithEq_of_rows h (row := 690) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_10
  simp only [← c43422, ← c43423, ← c43424] at e_690_10
  have e_690_11 := arithEq_of_rows h (row := 690) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_11
  simp only [← c43425, ← c43426, k2, ← c43427] at e_690_11
  have hr := e_690_11
  simp only [e_690_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 690 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4311)) := by
  have c43428 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.1
  have c43429 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.1
  have c43430 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.1
  have c43431 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.1
  have c43432 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.1
  have c43433 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_690_12 := arithEq_of_rows h (row := 690) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_12
  simp only [← c43428, ← c43429, ← c43430] at e_690_12
  have e_690_13 := arithEq_of_rows h (row := 690) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_13
  simp only [← c43431, ← c43432, k2, ← c43433] at e_690_13
  have hr := e_690_13
  simp only [e_690_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 690 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4312)) := by
  have c43434 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43435 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43436 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43437 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43438 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43439 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_690_14 := arithEq_of_rows h (row := 690) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_14
  simp only [← c43434, ← c43435, ← c43436] at e_690_14
  have e_690_15 := arithEq_of_rows h (row := 690) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_15
  simp only [← c43437, ← c43438, k2, ← c43439] at e_690_15
  have hr := e_690_15
  simp only [e_690_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 690 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4313)) := by
  have c43440 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43441 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43442 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43443 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43444 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43445 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_690_16 := arithEq_of_rows h (row := 690) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_16
  simp only [← c43440, ← c43441, ← c43442] at e_690_16
  have e_690_17 := arithEq_of_rows h (row := 690) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_17
  simp only [← c43443, ← c43444, k2, ← c43445] at e_690_17
  have hr := e_690_17
  simp only [e_690_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 690 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4314)) := by
  have c43446 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43447 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43448 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43449 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43450 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43451 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_690_18 := arithEq_of_rows h (row := 690) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_18
  simp only [← c43446, ← c43447, ← c43448] at e_690_18
  have e_690_19 := arithEq_of_rows h (row := 690) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_690_19
  simp only [← c43449, ← c43450, k2, ← c43451] at e_690_19
  have hr := e_690_19
  simp only [e_690_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 691 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4315)) := by
  have c43452 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43453 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43454 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43455 := (publicBatchWrapper89_copies1357 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43456 := (publicBatchWrapper89_copies1358 a h).1
  have c43457 := (publicBatchWrapper89_copies1358 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_691_0 := arithEq_of_rows h (row := 691) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_0
  simp only [← c43452, ← c43453, ← c43454] at e_691_0
  have e_691_1 := arithEq_of_rows h (row := 691) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_1
  simp only [← c43455, ← c43456, k2, ← c43457] at e_691_1
  have hr := e_691_1
  simp only [e_691_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 691 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4316)) := by
  have c43458 := (publicBatchWrapper89_copies1358 a h).2.2.1
  have c43459 := (publicBatchWrapper89_copies1358 a h).2.2.2.1
  have c43460 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.1
  have c43461 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.1
  have c43462 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.1
  have c43463 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_691_2 := arithEq_of_rows h (row := 691) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_2
  simp only [← c43458, ← c43459, ← c43460] at e_691_2
  have e_691_3 := arithEq_of_rows h (row := 691) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_3
  simp only [← c43461, ← c43462, k2, ← c43463] at e_691_3
  have hr := e_691_3
  simp only [e_691_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 691 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4317)) := by
  have c43464 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.1
  have c43465 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.1
  have c43466 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43467 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43468 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43469 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_691_4 := arithEq_of_rows h (row := 691) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_4
  simp only [← c43464, ← c43465, ← c43466] at e_691_4
  have e_691_5 := arithEq_of_rows h (row := 691) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_5
  simp only [← c43467, ← c43468, k2, ← c43469] at e_691_5
  have hr := e_691_5
  simp only [e_691_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 691 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4318)) := by
  have c43470 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43471 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43472 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43473 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43474 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43475 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_691_6 := arithEq_of_rows h (row := 691) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_6
  simp only [← c43470, ← c43471, ← c43472] at e_691_6
  have e_691_7 := arithEq_of_rows h (row := 691) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_7
  simp only [← c43473, ← c43474, k2, ← c43475] at e_691_7
  have hr := e_691_7
  simp only [e_691_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 691 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4319)) := by
  have c43476 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43477 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43478 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43479 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43480 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43481 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_691_8 := arithEq_of_rows h (row := 691) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_8
  simp only [← c43476, ← c43477, ← c43478] at e_691_8
  have e_691_9 := arithEq_of_rows h (row := 691) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_9
  simp only [← c43479, ← c43480, k2, ← c43481] at e_691_9
  have hr := e_691_9
  simp only [e_691_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 691 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4320)) := by
  have c43482 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43483 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43484 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43485 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43486 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43487 := (publicBatchWrapper89_copies1358 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_691_10 := arithEq_of_rows h (row := 691) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_10
  simp only [← c43482, ← c43483, ← c43484] at e_691_10
  have e_691_11 := arithEq_of_rows h (row := 691) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_11
  simp only [← c43485, ← c43486, k2, ← c43487] at e_691_11
  have hr := e_691_11
  simp only [e_691_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 691 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4321)) := by
  have c43488 := (publicBatchWrapper89_copies1359 a h).1
  have c43489 := (publicBatchWrapper89_copies1359 a h).2.1
  have c43490 := (publicBatchWrapper89_copies1359 a h).2.2.1
  have c43491 := (publicBatchWrapper89_copies1359 a h).2.2.2.1
  have c43492 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.1
  have c43493 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_691_12 := arithEq_of_rows h (row := 691) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_12
  simp only [← c43488, ← c43489, ← c43490] at e_691_12
  have e_691_13 := arithEq_of_rows h (row := 691) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_13
  simp only [← c43491, ← c43492, k2, ← c43493] at e_691_13
  have hr := e_691_13
  simp only [e_691_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 691 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4322)) := by
  have c43494 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.1
  have c43495 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.1
  have c43496 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.1
  have c43497 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.1
  have c43498 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43499 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_691_14 := arithEq_of_rows h (row := 691) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_14
  simp only [← c43494, ← c43495, ← c43496] at e_691_14
  have e_691_15 := arithEq_of_rows h (row := 691) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_15
  simp only [← c43497, ← c43498, k2, ← c43499] at e_691_15
  have hr := e_691_15
  simp only [e_691_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 691 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4323)) := by
  have c43500 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43501 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43502 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43503 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43504 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43505 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_691_16 := arithEq_of_rows h (row := 691) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_16
  simp only [← c43500, ← c43501, ← c43502] at e_691_16
  have e_691_17 := arithEq_of_rows h (row := 691) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_17
  simp only [← c43503, ← c43504, k2, ← c43505] at e_691_17
  have hr := e_691_17
  simp only [e_691_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 691 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4324)) := by
  have c43506 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43507 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43508 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43509 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43510 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43511 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_691_18 := arithEq_of_rows h (row := 691) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_18
  simp only [← c43506, ← c43507, ← c43508] at e_691_18
  have e_691_19 := arithEq_of_rows h (row := 691) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_691_19
  simp only [← c43509, ← c43510, k2, ← c43511] at e_691_19
  have hr := e_691_19
  simp only [e_691_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 692 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4325)) := by
  have c43512 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43513 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43514 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43515 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43516 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43517 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_692_0 := arithEq_of_rows h (row := 692) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_0
  simp only [← c43512, ← c43513, ← c43514] at e_692_0
  have e_692_1 := arithEq_of_rows h (row := 692) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_1
  simp only [← c43515, ← c43516, k2, ← c43517] at e_692_1
  have hr := e_692_1
  simp only [e_692_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 692 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4326)) := by
  have c43518 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43519 := (publicBatchWrapper89_copies1359 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43520 := (publicBatchWrapper89_copies1360 a h).1
  have c43521 := (publicBatchWrapper89_copies1360 a h).2.1
  have c43522 := (publicBatchWrapper89_copies1360 a h).2.2.1
  have c43523 := (publicBatchWrapper89_copies1360 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_692_2 := arithEq_of_rows h (row := 692) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_2
  simp only [← c43518, ← c43519, ← c43520] at e_692_2
  have e_692_3 := arithEq_of_rows h (row := 692) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_3
  simp only [← c43521, ← c43522, k2, ← c43523] at e_692_3
  have hr := e_692_3
  simp only [e_692_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 692 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4327)) := by
  have c43524 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.1
  have c43525 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.1
  have c43526 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.1
  have c43527 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.1
  have c43528 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.1
  have c43529 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_692_4 := arithEq_of_rows h (row := 692) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_4
  simp only [← c43524, ← c43525, ← c43526] at e_692_4
  have e_692_5 := arithEq_of_rows h (row := 692) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_5
  simp only [← c43527, ← c43528, k2, ← c43529] at e_692_5
  have hr := e_692_5
  simp only [e_692_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 692 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4328)) := by
  have c43530 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43531 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43532 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43533 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43534 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43535 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_692_6 := arithEq_of_rows h (row := 692) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_6
  simp only [← c43530, ← c43531, ← c43532] at e_692_6
  have e_692_7 := arithEq_of_rows h (row := 692) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_7
  simp only [← c43533, ← c43534, k2, ← c43535] at e_692_7
  have hr := e_692_7
  simp only [e_692_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 692 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4329)) := by
  have c43536 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43537 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43538 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43539 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43540 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43541 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_692_8 := arithEq_of_rows h (row := 692) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_8
  simp only [← c43536, ← c43537, ← c43538] at e_692_8
  have e_692_9 := arithEq_of_rows h (row := 692) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_9
  simp only [← c43539, ← c43540, k2, ← c43541] at e_692_9
  have hr := e_692_9
  simp only [e_692_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 692 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4330)) := by
  have c43542 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43543 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43544 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43545 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43546 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43547 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_692_10 := arithEq_of_rows h (row := 692) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_10
  simp only [← c43542, ← c43543, ← c43544] at e_692_10
  have e_692_11 := arithEq_of_rows h (row := 692) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_11
  simp only [← c43545, ← c43546, k2, ← c43547] at e_692_11
  have hr := e_692_11
  simp only [e_692_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 692 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4331)) := by
  have c43548 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43549 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43550 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43551 := (publicBatchWrapper89_copies1360 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43552 := (publicBatchWrapper89_copies1361 a h).1
  have c43553 := (publicBatchWrapper89_copies1361 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_692_12 := arithEq_of_rows h (row := 692) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_12
  simp only [← c43548, ← c43549, ← c43550] at e_692_12
  have e_692_13 := arithEq_of_rows h (row := 692) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_13
  simp only [← c43551, ← c43552, k2, ← c43553] at e_692_13
  have hr := e_692_13
  simp only [e_692_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 692 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4332)) := by
  have c43554 := (publicBatchWrapper89_copies1361 a h).2.2.1
  have c43555 := (publicBatchWrapper89_copies1361 a h).2.2.2.1
  have c43556 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.1
  have c43557 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.1
  have c43558 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.1
  have c43559 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_692_14 := arithEq_of_rows h (row := 692) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_14
  simp only [← c43554, ← c43555, ← c43556] at e_692_14
  have e_692_15 := arithEq_of_rows h (row := 692) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_15
  simp only [← c43557, ← c43558, k2, ← c43559] at e_692_15
  have hr := e_692_15
  simp only [e_692_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 692 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4333)) := by
  have c43560 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.1
  have c43561 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.1
  have c43562 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43563 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43564 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43565 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_692_16 := arithEq_of_rows h (row := 692) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_16
  simp only [← c43560, ← c43561, ← c43562] at e_692_16
  have e_692_17 := arithEq_of_rows h (row := 692) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_17
  simp only [← c43563, ← c43564, k2, ← c43565] at e_692_17
  have hr := e_692_17
  simp only [e_692_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 692 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4334)) := by
  have c43566 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43567 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43568 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43569 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43570 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43571 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_692_18 := arithEq_of_rows h (row := 692) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_18
  simp only [← c43566, ← c43567, ← c43568] at e_692_18
  have e_692_19 := arithEq_of_rows h (row := 692) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_692_19
  simp only [← c43569, ← c43570, k2, ← c43571] at e_692_19
  have hr := e_692_19
  simp only [e_692_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 693 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4335)) := by
  have c43572 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43573 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43574 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43575 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43576 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43577 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_693_0 := arithEq_of_rows h (row := 693) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_0
  simp only [← c43572, ← c43573, ← c43574] at e_693_0
  have e_693_1 := arithEq_of_rows h (row := 693) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_1
  simp only [← c43575, ← c43576, k2, ← c43577] at e_693_1
  have hr := e_693_1
  simp only [e_693_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 693 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4336)) := by
  have c43578 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43579 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43580 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43581 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43582 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43583 := (publicBatchWrapper89_copies1361 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_693_2 := arithEq_of_rows h (row := 693) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_2
  simp only [← c43578, ← c43579, ← c43580] at e_693_2
  have e_693_3 := arithEq_of_rows h (row := 693) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_3
  simp only [← c43581, ← c43582, k2, ← c43583] at e_693_3
  have hr := e_693_3
  simp only [e_693_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 693 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4337)) := by
  have c43584 := (publicBatchWrapper89_copies1362 a h).1
  have c43585 := (publicBatchWrapper89_copies1362 a h).2.1
  have c43586 := (publicBatchWrapper89_copies1362 a h).2.2.1
  have c43587 := (publicBatchWrapper89_copies1362 a h).2.2.2.1
  have c43588 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.1
  have c43589 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_693_4 := arithEq_of_rows h (row := 693) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_4
  simp only [← c43584, ← c43585, ← c43586] at e_693_4
  have e_693_5 := arithEq_of_rows h (row := 693) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_5
  simp only [← c43587, ← c43588, k2, ← c43589] at e_693_5
  have hr := e_693_5
  simp only [e_693_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 693 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4338)) := by
  have c43590 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.1
  have c43591 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.1
  have c43592 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.1
  have c43593 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.1
  have c43594 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43595 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_693_6 := arithEq_of_rows h (row := 693) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_6
  simp only [← c43590, ← c43591, ← c43592] at e_693_6
  have e_693_7 := arithEq_of_rows h (row := 693) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_7
  simp only [← c43593, ← c43594, k2, ← c43595] at e_693_7
  have hr := e_693_7
  simp only [e_693_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 693 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4339)) := by
  have c43596 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43597 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43598 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43599 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43600 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43601 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_693_8 := arithEq_of_rows h (row := 693) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_8
  simp only [← c43596, ← c43597, ← c43598] at e_693_8
  have e_693_9 := arithEq_of_rows h (row := 693) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_9
  simp only [← c43599, ← c43600, k2, ← c43601] at e_693_9
  have hr := e_693_9
  simp only [e_693_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 693 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4340)) := by
  have c43602 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43603 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43604 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43605 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43606 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43607 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_693_10 := arithEq_of_rows h (row := 693) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_10
  simp only [← c43602, ← c43603, ← c43604] at e_693_10
  have e_693_11 := arithEq_of_rows h (row := 693) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_11
  simp only [← c43605, ← c43606, k2, ← c43607] at e_693_11
  have hr := e_693_11
  simp only [e_693_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 693 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4341)) := by
  have c43608 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43609 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43610 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43611 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43612 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43613 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_693_12 := arithEq_of_rows h (row := 693) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_12
  simp only [← c43608, ← c43609, ← c43610] at e_693_12
  have e_693_13 := arithEq_of_rows h (row := 693) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_13
  simp only [← c43611, ← c43612, k2, ← c43613] at e_693_13
  have hr := e_693_13
  simp only [e_693_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 693 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4342)) := by
  have c43614 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43615 := (publicBatchWrapper89_copies1362 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43616 := (publicBatchWrapper89_copies1363 a h).1
  have c43617 := (publicBatchWrapper89_copies1363 a h).2.1
  have c43618 := (publicBatchWrapper89_copies1363 a h).2.2.1
  have c43619 := (publicBatchWrapper89_copies1363 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_693_14 := arithEq_of_rows h (row := 693) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_14
  simp only [← c43614, ← c43615, ← c43616] at e_693_14
  have e_693_15 := arithEq_of_rows h (row := 693) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_15
  simp only [← c43617, ← c43618, k2, ← c43619] at e_693_15
  have hr := e_693_15
  simp only [e_693_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 693 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4343)) := by
  have c43620 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.1
  have c43621 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.1
  have c43622 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.1
  have c43623 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.1
  have c43624 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.1
  have c43625 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_693_16 := arithEq_of_rows h (row := 693) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_16
  simp only [← c43620, ← c43621, ← c43622] at e_693_16
  have e_693_17 := arithEq_of_rows h (row := 693) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_17
  simp only [← c43623, ← c43624, k2, ← c43625] at e_693_17
  have hr := e_693_17
  simp only [e_693_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 693 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4344)) := by
  have c43626 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43627 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43628 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43629 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43630 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43631 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_693_18 := arithEq_of_rows h (row := 693) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_18
  simp only [← c43626, ← c43627, ← c43628] at e_693_18
  have e_693_19 := arithEq_of_rows h (row := 693) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_693_19
  simp only [← c43629, ← c43630, k2, ← c43631] at e_693_19
  have hr := e_693_19
  simp only [e_693_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 694 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4345)) := by
  have c43632 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43633 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43634 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43635 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43636 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43637 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_694_0 := arithEq_of_rows h (row := 694) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_0
  simp only [← c43632, ← c43633, ← c43634] at e_694_0
  have e_694_1 := arithEq_of_rows h (row := 694) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_1
  simp only [← c43635, ← c43636, k2, ← c43637] at e_694_1
  have hr := e_694_1
  simp only [e_694_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 694 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4346)) := by
  have c43638 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43639 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43640 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43641 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43642 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43643 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_694_2 := arithEq_of_rows h (row := 694) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_2
  simp only [← c43638, ← c43639, ← c43640] at e_694_2
  have e_694_3 := arithEq_of_rows h (row := 694) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_3
  simp only [← c43641, ← c43642, k2, ← c43643] at e_694_3
  have hr := e_694_3
  simp only [e_694_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 694 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4347)) := by
  have c43644 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43645 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43646 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43647 := (publicBatchWrapper89_copies1363 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43648 := (publicBatchWrapper89_copies1364 a h).1
  have c43649 := (publicBatchWrapper89_copies1364 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_694_4 := arithEq_of_rows h (row := 694) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_4
  simp only [← c43644, ← c43645, ← c43646] at e_694_4
  have e_694_5 := arithEq_of_rows h (row := 694) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_5
  simp only [← c43647, ← c43648, k2, ← c43649] at e_694_5
  have hr := e_694_5
  simp only [e_694_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 694 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4348)) := by
  have c43650 := (publicBatchWrapper89_copies1364 a h).2.2.1
  have c43651 := (publicBatchWrapper89_copies1364 a h).2.2.2.1
  have c43652 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.1
  have c43653 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.1
  have c43654 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.1
  have c43655 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_694_6 := arithEq_of_rows h (row := 694) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_6
  simp only [← c43650, ← c43651, ← c43652] at e_694_6
  have e_694_7 := arithEq_of_rows h (row := 694) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_7
  simp only [← c43653, ← c43654, k2, ← c43655] at e_694_7
  have hr := e_694_7
  simp only [e_694_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 694 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4349)) := by
  have c43656 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.1
  have c43657 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.1
  have c43658 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43659 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43660 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43661 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_694_8 := arithEq_of_rows h (row := 694) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_8
  simp only [← c43656, ← c43657, ← c43658] at e_694_8
  have e_694_9 := arithEq_of_rows h (row := 694) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_9
  simp only [← c43659, ← c43660, k2, ← c43661] at e_694_9
  have hr := e_694_9
  simp only [e_694_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 694 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4350)) := by
  have c43662 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43663 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43664 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43665 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43666 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43667 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_694_10 := arithEq_of_rows h (row := 694) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_10
  simp only [← c43662, ← c43663, ← c43664] at e_694_10
  have e_694_11 := arithEq_of_rows h (row := 694) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_11
  simp only [← c43665, ← c43666, k2, ← c43667] at e_694_11
  have hr := e_694_11
  simp only [e_694_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 694 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4351)) := by
  have c43668 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43669 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43670 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43671 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43672 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43673 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_694_12 := arithEq_of_rows h (row := 694) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_12
  simp only [← c43668, ← c43669, ← c43670] at e_694_12
  have e_694_13 := arithEq_of_rows h (row := 694) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_13
  simp only [← c43671, ← c43672, k2, ← c43673] at e_694_13
  have hr := e_694_13
  simp only [e_694_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 694 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4352)) := by
  have c43674 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43675 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43676 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43677 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43678 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43679 := (publicBatchWrapper89_copies1364 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_694_14 := arithEq_of_rows h (row := 694) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_14
  simp only [← c43674, ← c43675, ← c43676] at e_694_14
  have e_694_15 := arithEq_of_rows h (row := 694) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_15
  simp only [← c43677, ← c43678, k2, ← c43679] at e_694_15
  have hr := e_694_15
  simp only [e_694_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 694 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4353)) := by
  have c43680 := (publicBatchWrapper89_copies1365 a h).1
  have c43681 := (publicBatchWrapper89_copies1365 a h).2.1
  have c43682 := (publicBatchWrapper89_copies1365 a h).2.2.1
  have c43683 := (publicBatchWrapper89_copies1365 a h).2.2.2.1
  have c43684 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.1
  have c43685 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_694_16 := arithEq_of_rows h (row := 694) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_16
  simp only [← c43680, ← c43681, ← c43682] at e_694_16
  have e_694_17 := arithEq_of_rows h (row := 694) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_17
  simp only [← c43683, ← c43684, k2, ← c43685] at e_694_17
  have hr := e_694_17
  simp only [e_694_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 694 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4354)) := by
  have c43686 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.1
  have c43687 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.1
  have c43688 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.1
  have c43689 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.1
  have c43690 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43691 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_694_18 := arithEq_of_rows h (row := 694) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_18
  simp only [← c43686, ← c43687, ← c43688] at e_694_18
  have e_694_19 := arithEq_of_rows h (row := 694) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_694_19
  simp only [← c43689, ← c43690, k2, ← c43691] at e_694_19
  have hr := e_694_19
  simp only [e_694_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 695 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4355)) := by
  have c43692 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43693 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43694 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43695 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43696 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43697 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_695_0 := arithEq_of_rows h (row := 695) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_0
  simp only [← c43692, ← c43693, ← c43694] at e_695_0
  have e_695_1 := arithEq_of_rows h (row := 695) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_1
  simp only [← c43695, ← c43696, k2, ← c43697] at e_695_1
  have hr := e_695_1
  simp only [e_695_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 695 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4356)) := by
  have c43698 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43699 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43700 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43701 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43702 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43703 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_695_2 := arithEq_of_rows h (row := 695) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_2
  simp only [← c43698, ← c43699, ← c43700] at e_695_2
  have e_695_3 := arithEq_of_rows h (row := 695) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_3
  simp only [← c43701, ← c43702, k2, ← c43703] at e_695_3
  have hr := e_695_3
  simp only [e_695_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 695 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4357)) := by
  have c43704 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43705 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43706 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43707 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43708 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43709 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_695_4 := arithEq_of_rows h (row := 695) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_4
  simp only [← c43704, ← c43705, ← c43706] at e_695_4
  have e_695_5 := arithEq_of_rows h (row := 695) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_5
  simp only [← c43707, ← c43708, k2, ← c43709] at e_695_5
  have hr := e_695_5
  simp only [e_695_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 695 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4358)) := by
  have c43710 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43711 := (publicBatchWrapper89_copies1365 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43712 := (publicBatchWrapper89_copies1366 a h).1
  have c43713 := (publicBatchWrapper89_copies1366 a h).2.1
  have c43714 := (publicBatchWrapper89_copies1366 a h).2.2.1
  have c43715 := (publicBatchWrapper89_copies1366 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_695_6 := arithEq_of_rows h (row := 695) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_6
  simp only [← c43710, ← c43711, ← c43712] at e_695_6
  have e_695_7 := arithEq_of_rows h (row := 695) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_7
  simp only [← c43713, ← c43714, k2, ← c43715] at e_695_7
  have hr := e_695_7
  simp only [e_695_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 695 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4359)) := by
  have c43716 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.1
  have c43717 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.1
  have c43718 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.1
  have c43719 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.1
  have c43720 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.1
  have c43721 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_695_8 := arithEq_of_rows h (row := 695) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_8
  simp only [← c43716, ← c43717, ← c43718] at e_695_8
  have e_695_9 := arithEq_of_rows h (row := 695) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_9
  simp only [← c43719, ← c43720, k2, ← c43721] at e_695_9
  have hr := e_695_9
  simp only [e_695_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 695 47) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4360)) := by
  have c43722 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43723 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43724 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43725 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43726 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43727 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_695_10 := arithEq_of_rows h (row := 695) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_10
  simp only [← c43722, ← c43723, ← c43724] at e_695_10
  have e_695_11 := arithEq_of_rows h (row := 695) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_11
  simp only [← c43725, ← c43726, k2, ← c43727] at e_695_11
  have hr := e_695_11
  simp only [e_695_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 695 55) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4361)) := by
  have c43728 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43729 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43730 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43731 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43732 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43733 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_695_12 := arithEq_of_rows h (row := 695) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_12
  simp only [← c43728, ← c43729, ← c43730] at e_695_12
  have e_695_13 := arithEq_of_rows h (row := 695) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_13
  simp only [← c43731, ← c43732, k2, ← c43733] at e_695_13
  have hr := e_695_13
  simp only [e_695_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 695 63) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4362)) := by
  have c43734 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43735 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43736 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43737 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43738 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43739 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_695_14 := arithEq_of_rows h (row := 695) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_14
  simp only [← c43734, ← c43735, ← c43736] at e_695_14
  have e_695_15 := arithEq_of_rows h (row := 695) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_15
  simp only [← c43737, ← c43738, k2, ← c43739] at e_695_15
  have hr := e_695_15
  simp only [e_695_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 695 71) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4363)) := by
  have c43740 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43741 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43742 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43743 := (publicBatchWrapper89_copies1366 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43744 := (publicBatchWrapper89_copies1367 a h).1
  have c43745 := (publicBatchWrapper89_copies1367 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_695_16 := arithEq_of_rows h (row := 695) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_16
  simp only [← c43740, ← c43741, ← c43742] at e_695_16
  have e_695_17 := arithEq_of_rows h (row := 695) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_17
  simp only [← c43743, ← c43744, k2, ← c43745] at e_695_17
  have hr := e_695_17
  simp only [e_695_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 695 79) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4364)) := by
  have c43746 := (publicBatchWrapper89_copies1367 a h).2.2.1
  have c43747 := (publicBatchWrapper89_copies1367 a h).2.2.2.1
  have c43748 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.1
  have c43749 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.1
  have c43750 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.1
  have c43751 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_695_18 := arithEq_of_rows h (row := 695) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_18
  simp only [← c43746, ← c43747, ← c43748] at e_695_18
  have e_695_19 := arithEq_of_rows h (row := 695) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_695_19
  simp only [← c43749, ← c43750, k2, ← c43751] at e_695_19
  have hr := e_695_19
  simp only [e_695_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 696 7) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4365)) := by
  have c43752 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.1
  have c43753 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.1
  have c43754 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43755 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43756 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43757 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_696_0 := arithEq_of_rows h (row := 696) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_0
  simp only [← c43752, ← c43753, ← c43754] at e_696_0
  have e_696_1 := arithEq_of_rows h (row := 696) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_1
  simp only [← c43755, ← c43756, k2, ← c43757] at e_696_1
  have hr := e_696_1
  simp only [e_696_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 696 15) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4366)) := by
  have c43758 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43759 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43760 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43761 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43762 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43763 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_696_2 := arithEq_of_rows h (row := 696) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_2
  simp only [← c43758, ← c43759, ← c43760] at e_696_2
  have e_696_3 := arithEq_of_rows h (row := 696) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_3
  simp only [← c43761, ← c43762, k2, ← c43763] at e_696_3
  have hr := e_696_3
  simp only [e_696_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 696 23) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4367)) := by
  have c43764 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43765 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43766 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43767 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43768 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43769 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_696_4 := arithEq_of_rows h (row := 696) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_4
  simp only [← c43764, ← c43765, ← c43766] at e_696_4
  have e_696_5 := arithEq_of_rows h (row := 696) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_5
  simp only [← c43767, ← c43768, k2, ← c43769] at e_696_5
  have hr := e_696_5
  simp only [e_696_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 696 31) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4368)) := by
  have c43770 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43771 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43772 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43773 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43774 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43775 := (publicBatchWrapper89_copies1367 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_696_6 := arithEq_of_rows h (row := 696) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_6
  simp only [← c43770, ← c43771, ← c43772] at e_696_6
  have e_696_7 := arithEq_of_rows h (row := 696) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_7
  simp only [← c43773, ← c43774, k2, ← c43775] at e_696_7
  have hr := e_696_7
  simp only [e_696_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 696 39) = bselect (a (.wire 22 23)) (a (.virt 200)) (a (.virt 4369)) := by
  have c43776 := (publicBatchWrapper89_copies1368 a h).1
  have c43777 := (publicBatchWrapper89_copies1368 a h).2.1
  have c43778 := (publicBatchWrapper89_copies1368 a h).2.2.1
  have c43779 := (publicBatchWrapper89_copies1368 a h).2.2.2.1
  have c43780 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.1
  have c43781 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_696_8 := arithEq_of_rows h (row := 696) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_8
  simp only [← c43776, ← c43777, ← c43778] at e_696_8
  have e_696_9 := arithEq_of_rows h (row := 696) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_9
  simp only [← c43779, ← c43780, k2, ← c43781] at e_696_9
  have hr := e_696_9
  simp only [e_696_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 696 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4410)) := by
  have c43782 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.1
  have c43783 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.1
  have c43784 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.1
  have c43785 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.1
  have c43786 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43787 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_696_10 := arithEq_of_rows h (row := 696) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_10
  simp only [← c43782, ← c43783, ← c43784] at e_696_10
  have e_696_11 := arithEq_of_rows h (row := 696) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_11
  simp only [← c43785, ← c43786, k2, ← c43787] at e_696_11
  have hr := e_696_11
  simp only [e_696_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 696 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4411)) := by
  have c43788 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43789 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43790 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43791 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43792 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43793 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_696_12 := arithEq_of_rows h (row := 696) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_12
  simp only [← c43788, ← c43789, ← c43790] at e_696_12
  have e_696_13 := arithEq_of_rows h (row := 696) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_13
  simp only [← c43791, ← c43792, k2, ← c43793] at e_696_13
  have hr := e_696_13
  simp only [e_696_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 696 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4412)) := by
  have c43794 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43795 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43796 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43797 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43798 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43799 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_696_14 := arithEq_of_rows h (row := 696) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_14
  simp only [← c43794, ← c43795, ← c43796] at e_696_14
  have e_696_15 := arithEq_of_rows h (row := 696) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_15
  simp only [← c43797, ← c43798, k2, ← c43799] at e_696_15
  have hr := e_696_15
  simp only [e_696_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 696 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4413)) := by
  have c43800 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43801 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43802 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43803 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43804 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43805 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_696_16 := arithEq_of_rows h (row := 696) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_16
  simp only [← c43800, ← c43801, ← c43802] at e_696_16
  have e_696_17 := arithEq_of_rows h (row := 696) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_17
  simp only [← c43803, ← c43804, k2, ← c43805] at e_696_17
  have hr := e_696_17
  simp only [e_696_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 696 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4414)) := by
  have c43806 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43807 := (publicBatchWrapper89_copies1368 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43808 := (publicBatchWrapper89_copies1369 a h).1
  have c43809 := (publicBatchWrapper89_copies1369 a h).2.1
  have c43810 := (publicBatchWrapper89_copies1369 a h).2.2.1
  have c43811 := (publicBatchWrapper89_copies1369 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_696_18 := arithEq_of_rows h (row := 696) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_18
  simp only [← c43806, ← c43807, ← c43808] at e_696_18
  have e_696_19 := arithEq_of_rows h (row := 696) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_696_19
  simp only [← c43809, ← c43810, k2, ← c43811] at e_696_19
  have hr := e_696_19
  simp only [e_696_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 697 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4415)) := by
  have c43812 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.1
  have c43813 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.1
  have c43814 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.1
  have c43815 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.1
  have c43816 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.1
  have c43817 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_697_0 := arithEq_of_rows h (row := 697) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_0
  simp only [← c43812, ← c43813, ← c43814] at e_697_0
  have e_697_1 := arithEq_of_rows h (row := 697) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_1
  simp only [← c43815, ← c43816, k2, ← c43817] at e_697_1
  have hr := e_697_1
  simp only [e_697_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 697 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4416)) := by
  have c43818 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43819 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43820 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43821 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43822 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43823 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_697_2 := arithEq_of_rows h (row := 697) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_2
  simp only [← c43818, ← c43819, ← c43820] at e_697_2
  have e_697_3 := arithEq_of_rows h (row := 697) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_3
  simp only [← c43821, ← c43822, k2, ← c43823] at e_697_3
  have hr := e_697_3
  simp only [e_697_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 697 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4417)) := by
  have c43824 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43825 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43826 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43827 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43828 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43829 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_697_4 := arithEq_of_rows h (row := 697) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_4
  simp only [← c43824, ← c43825, ← c43826] at e_697_4
  have e_697_5 := arithEq_of_rows h (row := 697) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_5
  simp only [← c43827, ← c43828, k2, ← c43829] at e_697_5
  have hr := e_697_5
  simp only [e_697_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 697 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4418)) := by
  have c43830 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43831 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43832 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43833 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43834 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43835 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_697_6 := arithEq_of_rows h (row := 697) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_6
  simp only [← c43830, ← c43831, ← c43832] at e_697_6
  have e_697_7 := arithEq_of_rows h (row := 697) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_7
  simp only [← c43833, ← c43834, k2, ← c43835] at e_697_7
  have hr := e_697_7
  simp only [e_697_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 697 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4419)) := by
  have c43836 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43837 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43838 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43839 := (publicBatchWrapper89_copies1369 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43840 := (publicBatchWrapper89_copies1370 a h).1
  have c43841 := (publicBatchWrapper89_copies1370 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_697_8 := arithEq_of_rows h (row := 697) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_8
  simp only [← c43836, ← c43837, ← c43838] at e_697_8
  have e_697_9 := arithEq_of_rows h (row := 697) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_9
  simp only [← c43839, ← c43840, k2, ← c43841] at e_697_9
  have hr := e_697_9
  simp only [e_697_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 697 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4420)) := by
  have c43842 := (publicBatchWrapper89_copies1370 a h).2.2.1
  have c43843 := (publicBatchWrapper89_copies1370 a h).2.2.2.1
  have c43844 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.1
  have c43845 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.1
  have c43846 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.1
  have c43847 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_697_10 := arithEq_of_rows h (row := 697) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_10
  simp only [← c43842, ← c43843, ← c43844] at e_697_10
  have e_697_11 := arithEq_of_rows h (row := 697) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_11
  simp only [← c43845, ← c43846, k2, ← c43847] at e_697_11
  have hr := e_697_11
  simp only [e_697_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 697 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4421)) := by
  have c43848 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.1
  have c43849 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.1
  have c43850 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43851 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43852 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43853 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_697_12 := arithEq_of_rows h (row := 697) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_12
  simp only [← c43848, ← c43849, ← c43850] at e_697_12
  have e_697_13 := arithEq_of_rows h (row := 697) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_13
  simp only [← c43851, ← c43852, k2, ← c43853] at e_697_13
  have hr := e_697_13
  simp only [e_697_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 697 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4422)) := by
  have c43854 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43855 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43856 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43857 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43858 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43859 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_697_14 := arithEq_of_rows h (row := 697) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_14
  simp only [← c43854, ← c43855, ← c43856] at e_697_14
  have e_697_15 := arithEq_of_rows h (row := 697) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_15
  simp only [← c43857, ← c43858, k2, ← c43859] at e_697_15
  have hr := e_697_15
  simp only [e_697_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 697 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4423)) := by
  have c43860 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43861 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43862 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43863 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43864 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43865 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_697_16 := arithEq_of_rows h (row := 697) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_16
  simp only [← c43860, ← c43861, ← c43862] at e_697_16
  have e_697_17 := arithEq_of_rows h (row := 697) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_17
  simp only [← c43863, ← c43864, k2, ← c43865] at e_697_17
  have hr := e_697_17
  simp only [e_697_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 697 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4424)) := by
  have c43866 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43867 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43868 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43869 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43870 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43871 := (publicBatchWrapper89_copies1370 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_697_18 := arithEq_of_rows h (row := 697) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_18
  simp only [← c43866, ← c43867, ← c43868] at e_697_18
  have e_697_19 := arithEq_of_rows h (row := 697) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_697_19
  simp only [← c43869, ← c43870, k2, ← c43871] at e_697_19
  have hr := e_697_19
  simp only [e_697_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 698 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4425)) := by
  have c43872 := (publicBatchWrapper89_copies1371 a h).1
  have c43873 := (publicBatchWrapper89_copies1371 a h).2.1
  have c43874 := (publicBatchWrapper89_copies1371 a h).2.2.1
  have c43875 := (publicBatchWrapper89_copies1371 a h).2.2.2.1
  have c43876 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.1
  have c43877 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_698_0 := arithEq_of_rows h (row := 698) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_0
  simp only [← c43872, ← c43873, ← c43874] at e_698_0
  have e_698_1 := arithEq_of_rows h (row := 698) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_1
  simp only [← c43875, ← c43876, k2, ← c43877] at e_698_1
  have hr := e_698_1
  simp only [e_698_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 698 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4426)) := by
  have c43878 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.1
  have c43879 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.1
  have c43880 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.1
  have c43881 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.1
  have c43882 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43883 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_698_2 := arithEq_of_rows h (row := 698) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_2
  simp only [← c43878, ← c43879, ← c43880] at e_698_2
  have e_698_3 := arithEq_of_rows h (row := 698) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_3
  simp only [← c43881, ← c43882, k2, ← c43883] at e_698_3
  have hr := e_698_3
  simp only [e_698_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 698 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4427)) := by
  have c43884 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43885 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43886 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43887 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43888 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43889 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_698_4 := arithEq_of_rows h (row := 698) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_4
  simp only [← c43884, ← c43885, ← c43886] at e_698_4
  have e_698_5 := arithEq_of_rows h (row := 698) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_5
  simp only [← c43887, ← c43888, k2, ← c43889] at e_698_5
  have hr := e_698_5
  simp only [e_698_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 698 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4428)) := by
  have c43890 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43891 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43892 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43893 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43894 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43895 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_698_6 := arithEq_of_rows h (row := 698) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_6
  simp only [← c43890, ← c43891, ← c43892] at e_698_6
  have e_698_7 := arithEq_of_rows h (row := 698) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_7
  simp only [← c43893, ← c43894, k2, ← c43895] at e_698_7
  have hr := e_698_7
  simp only [e_698_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 698 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4429)) := by
  have c43896 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43897 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43898 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43899 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43900 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43901 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_698_8 := arithEq_of_rows h (row := 698) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_8
  simp only [← c43896, ← c43897, ← c43898] at e_698_8
  have e_698_9 := arithEq_of_rows h (row := 698) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_9
  simp only [← c43899, ← c43900, k2, ← c43901] at e_698_9
  have hr := e_698_9
  simp only [e_698_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 698 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4430)) := by
  have c43902 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43903 := (publicBatchWrapper89_copies1371 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43904 := (publicBatchWrapper89_copies1372 a h).1
  have c43905 := (publicBatchWrapper89_copies1372 a h).2.1
  have c43906 := (publicBatchWrapper89_copies1372 a h).2.2.1
  have c43907 := (publicBatchWrapper89_copies1372 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_698_10 := arithEq_of_rows h (row := 698) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_10
  simp only [← c43902, ← c43903, ← c43904] at e_698_10
  have e_698_11 := arithEq_of_rows h (row := 698) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_11
  simp only [← c43905, ← c43906, k2, ← c43907] at e_698_11
  have hr := e_698_11
  simp only [e_698_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 698 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4431)) := by
  have c43908 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.1
  have c43909 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.1
  have c43910 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.1
  have c43911 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.1
  have c43912 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.1
  have c43913 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_698_12 := arithEq_of_rows h (row := 698) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_12
  simp only [← c43908, ← c43909, ← c43910] at e_698_12
  have e_698_13 := arithEq_of_rows h (row := 698) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_13
  simp only [← c43911, ← c43912, k2, ← c43913] at e_698_13
  have hr := e_698_13
  simp only [e_698_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 698 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4432)) := by
  have c43914 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43915 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43916 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43917 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43918 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43919 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_698_14 := arithEq_of_rows h (row := 698) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_14
  simp only [← c43914, ← c43915, ← c43916] at e_698_14
  have e_698_15 := arithEq_of_rows h (row := 698) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_15
  simp only [← c43917, ← c43918, k2, ← c43919] at e_698_15
  have hr := e_698_15
  simp only [e_698_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 698 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4433)) := by
  have c43920 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43921 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43922 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43923 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43924 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43925 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_698_16 := arithEq_of_rows h (row := 698) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_16
  simp only [← c43920, ← c43921, ← c43922] at e_698_16
  have e_698_17 := arithEq_of_rows h (row := 698) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_17
  simp only [← c43923, ← c43924, k2, ← c43925] at e_698_17
  have hr := e_698_17
  simp only [e_698_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 698 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4434)) := by
  have c43926 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43927 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43928 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43929 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43930 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43931 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_698_18 := arithEq_of_rows h (row := 698) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_18
  simp only [← c43926, ← c43927, ← c43928] at e_698_18
  have e_698_19 := arithEq_of_rows h (row := 698) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_698_19
  simp only [← c43929, ← c43930, k2, ← c43931] at e_698_19
  have hr := e_698_19
  simp only [e_698_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 699 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4435)) := by
  have c43932 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43933 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43934 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43935 := (publicBatchWrapper89_copies1372 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c43936 := (publicBatchWrapper89_copies1373 a h).1
  have c43937 := (publicBatchWrapper89_copies1373 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_699_0 := arithEq_of_rows h (row := 699) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_0
  simp only [← c43932, ← c43933, ← c43934] at e_699_0
  have e_699_1 := arithEq_of_rows h (row := 699) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_1
  simp only [← c43935, ← c43936, k2, ← c43937] at e_699_1
  have hr := e_699_1
  simp only [e_699_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 699 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4436)) := by
  have c43938 := (publicBatchWrapper89_copies1373 a h).2.2.1
  have c43939 := (publicBatchWrapper89_copies1373 a h).2.2.2.1
  have c43940 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.1
  have c43941 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.1
  have c43942 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.1
  have c43943 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_699_2 := arithEq_of_rows h (row := 699) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_2
  simp only [← c43938, ← c43939, ← c43940] at e_699_2
  have e_699_3 := arithEq_of_rows h (row := 699) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_3
  simp only [← c43941, ← c43942, k2, ← c43943] at e_699_3
  have hr := e_699_3
  simp only [e_699_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 699 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4437)) := by
  have c43944 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.1
  have c43945 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.1
  have c43946 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43947 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c43948 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43949 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_699_4 := arithEq_of_rows h (row := 699) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_4
  simp only [← c43944, ← c43945, ← c43946] at e_699_4
  have e_699_5 := arithEq_of_rows h (row := 699) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_5
  simp only [← c43947, ← c43948, k2, ← c43949] at e_699_5
  have hr := e_699_5
  simp only [e_699_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 699 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4438)) := by
  have c43950 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43951 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43952 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43953 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43954 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43955 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_699_6 := arithEq_of_rows h (row := 699) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_6
  simp only [← c43950, ← c43951, ← c43952] at e_699_6
  have e_699_7 := arithEq_of_rows h (row := 699) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_7
  simp only [← c43953, ← c43954, k2, ← c43955] at e_699_7
  have hr := e_699_7
  simp only [e_699_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 699 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4439)) := by
  have c43956 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43957 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43958 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43959 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43960 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43961 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_699_8 := arithEq_of_rows h (row := 699) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_8
  simp only [← c43956, ← c43957, ← c43958] at e_699_8
  have e_699_9 := arithEq_of_rows h (row := 699) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_9
  simp only [← c43959, ← c43960, k2, ← c43961] at e_699_9
  have hr := e_699_9
  simp only [e_699_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 699 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4440)) := by
  have c43962 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43963 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43964 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43965 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43966 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43967 := (publicBatchWrapper89_copies1373 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_699_10 := arithEq_of_rows h (row := 699) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_10
  simp only [← c43962, ← c43963, ← c43964] at e_699_10
  have e_699_11 := arithEq_of_rows h (row := 699) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_11
  simp only [← c43965, ← c43966, k2, ← c43967] at e_699_11
  have hr := e_699_11
  simp only [e_699_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 699 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4441)) := by
  have c43968 := (publicBatchWrapper89_copies1374 a h).1
  have c43969 := (publicBatchWrapper89_copies1374 a h).2.1
  have c43970 := (publicBatchWrapper89_copies1374 a h).2.2.1
  have c43971 := (publicBatchWrapper89_copies1374 a h).2.2.2.1
  have c43972 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.1
  have c43973 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_699_12 := arithEq_of_rows h (row := 699) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_12
  simp only [← c43968, ← c43969, ← c43970] at e_699_12
  have e_699_13 := arithEq_of_rows h (row := 699) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_13
  simp only [← c43971, ← c43972, k2, ← c43973] at e_699_13
  have hr := e_699_13
  simp only [e_699_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 699 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4442)) := by
  have c43974 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.1
  have c43975 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.1
  have c43976 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.1
  have c43977 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.1
  have c43978 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.1
  have c43979 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_699_14 := arithEq_of_rows h (row := 699) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_14
  simp only [← c43974, ← c43975, ← c43976] at e_699_14
  have e_699_15 := arithEq_of_rows h (row := 699) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_15
  simp only [← c43977, ← c43978, k2, ← c43979] at e_699_15
  have hr := e_699_15
  simp only [e_699_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 699 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4443)) := by
  have c43980 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43981 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43982 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43983 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43984 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43985 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_699_16 := arithEq_of_rows h (row := 699) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_16
  simp only [← c43980, ← c43981, ← c43982] at e_699_16
  have e_699_17 := arithEq_of_rows h (row := 699) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_17
  simp only [← c43983, ← c43984, k2, ← c43985] at e_699_17
  have hr := e_699_17
  simp only [e_699_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 699 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4444)) := by
  have c43986 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43987 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43988 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43989 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43990 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43991 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_699_18 := arithEq_of_rows h (row := 699) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_18
  simp only [← c43986, ← c43987, ← c43988] at e_699_18
  have e_699_19 := arithEq_of_rows h (row := 699) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_699_19
  simp only [← c43989, ← c43990, k2, ← c43991] at e_699_19
  have hr := e_699_19
  simp only [e_699_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 700 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4445)) := by
  have c43992 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43993 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43994 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43995 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43996 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43997 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_700_0 := arithEq_of_rows h (row := 700) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_0
  simp only [← c43992, ← c43993, ← c43994] at e_700_0
  have e_700_1 := arithEq_of_rows h (row := 700) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_1
  simp only [← c43995, ← c43996, k2, ← c43997] at e_700_1
  have hr := e_700_1
  simp only [e_700_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 700 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4446)) := by
  have c43998 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c43999 := (publicBatchWrapper89_copies1374 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44000 := (publicBatchWrapper89_copies1375 a h).1
  have c44001 := (publicBatchWrapper89_copies1375 a h).2.1
  have c44002 := (publicBatchWrapper89_copies1375 a h).2.2.1
  have c44003 := (publicBatchWrapper89_copies1375 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_700_2 := arithEq_of_rows h (row := 700) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_2
  simp only [← c43998, ← c43999, ← c44000] at e_700_2
  have e_700_3 := arithEq_of_rows h (row := 700) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_3
  simp only [← c44001, ← c44002, k2, ← c44003] at e_700_3
  have hr := e_700_3
  simp only [e_700_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 700 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4447)) := by
  have c44004 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.1
  have c44005 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.1
  have c44006 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.1
  have c44007 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.1
  have c44008 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.1
  have c44009 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_700_4 := arithEq_of_rows h (row := 700) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_4
  simp only [← c44004, ← c44005, ← c44006] at e_700_4
  have e_700_5 := arithEq_of_rows h (row := 700) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_5
  simp only [← c44007, ← c44008, k2, ← c44009] at e_700_5
  have hr := e_700_5
  simp only [e_700_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 700 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4448)) := by
  have c44010 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44011 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44012 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44013 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44014 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44015 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_700_6 := arithEq_of_rows h (row := 700) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_6
  simp only [← c44010, ← c44011, ← c44012] at e_700_6
  have e_700_7 := arithEq_of_rows h (row := 700) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_7
  simp only [← c44013, ← c44014, k2, ← c44015] at e_700_7
  have hr := e_700_7
  simp only [e_700_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 700 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4449)) := by
  have c44016 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44017 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44018 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44019 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44020 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44021 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_700_8 := arithEq_of_rows h (row := 700) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_8
  simp only [← c44016, ← c44017, ← c44018] at e_700_8
  have e_700_9 := arithEq_of_rows h (row := 700) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_9
  simp only [← c44019, ← c44020, k2, ← c44021] at e_700_9
  have hr := e_700_9
  simp only [e_700_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 700 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4450)) := by
  have c44022 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44023 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44024 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44025 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44026 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44027 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_700_10 := arithEq_of_rows h (row := 700) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_10
  simp only [← c44022, ← c44023, ← c44024] at e_700_10
  have e_700_11 := arithEq_of_rows h (row := 700) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_11
  simp only [← c44025, ← c44026, k2, ← c44027] at e_700_11
  have hr := e_700_11
  simp only [e_700_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 700 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4451)) := by
  have c44028 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44029 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44030 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44031 := (publicBatchWrapper89_copies1375 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44032 := (publicBatchWrapper89_copies1376 a h).1
  have c44033 := (publicBatchWrapper89_copies1376 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_700_12 := arithEq_of_rows h (row := 700) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_12
  simp only [← c44028, ← c44029, ← c44030] at e_700_12
  have e_700_13 := arithEq_of_rows h (row := 700) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_13
  simp only [← c44031, ← c44032, k2, ← c44033] at e_700_13
  have hr := e_700_13
  simp only [e_700_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 700 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4452)) := by
  have c44034 := (publicBatchWrapper89_copies1376 a h).2.2.1
  have c44035 := (publicBatchWrapper89_copies1376 a h).2.2.2.1
  have c44036 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.1
  have c44037 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.1
  have c44038 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.1
  have c44039 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_700_14 := arithEq_of_rows h (row := 700) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_14
  simp only [← c44034, ← c44035, ← c44036] at e_700_14
  have e_700_15 := arithEq_of_rows h (row := 700) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_15
  simp only [← c44037, ← c44038, k2, ← c44039] at e_700_15
  have hr := e_700_15
  simp only [e_700_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 700 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4453)) := by
  have c44040 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.1
  have c44041 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.1
  have c44042 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44043 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44044 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44045 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_700_16 := arithEq_of_rows h (row := 700) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_16
  simp only [← c44040, ← c44041, ← c44042] at e_700_16
  have e_700_17 := arithEq_of_rows h (row := 700) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_17
  simp only [← c44043, ← c44044, k2, ← c44045] at e_700_17
  have hr := e_700_17
  simp only [e_700_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 700 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4454)) := by
  have c44046 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44047 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44048 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44049 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44050 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44051 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_700_18 := arithEq_of_rows h (row := 700) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_18
  simp only [← c44046, ← c44047, ← c44048] at e_700_18
  have e_700_19 := arithEq_of_rows h (row := 700) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_700_19
  simp only [← c44049, ← c44050, k2, ← c44051] at e_700_19
  have hr := e_700_19
  simp only [e_700_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 701 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4455)) := by
  have c44052 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44053 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44054 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44055 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44056 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44057 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_701_0 := arithEq_of_rows h (row := 701) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_0
  simp only [← c44052, ← c44053, ← c44054] at e_701_0
  have e_701_1 := arithEq_of_rows h (row := 701) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_1
  simp only [← c44055, ← c44056, k2, ← c44057] at e_701_1
  have hr := e_701_1
  simp only [e_701_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 701 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4456)) := by
  have c44058 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44059 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44060 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44061 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44062 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44063 := (publicBatchWrapper89_copies1376 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_701_2 := arithEq_of_rows h (row := 701) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_2
  simp only [← c44058, ← c44059, ← c44060] at e_701_2
  have e_701_3 := arithEq_of_rows h (row := 701) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_3
  simp only [← c44061, ← c44062, k2, ← c44063] at e_701_3
  have hr := e_701_3
  simp only [e_701_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 701 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4457)) := by
  have c44064 := (publicBatchWrapper89_copies1377 a h).1
  have c44065 := (publicBatchWrapper89_copies1377 a h).2.1
  have c44066 := (publicBatchWrapper89_copies1377 a h).2.2.1
  have c44067 := (publicBatchWrapper89_copies1377 a h).2.2.2.1
  have c44068 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.1
  have c44069 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_701_4 := arithEq_of_rows h (row := 701) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_4
  simp only [← c44064, ← c44065, ← c44066] at e_701_4
  have e_701_5 := arithEq_of_rows h (row := 701) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_5
  simp only [← c44067, ← c44068, k2, ← c44069] at e_701_5
  have hr := e_701_5
  simp only [e_701_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 701 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4458)) := by
  have c44070 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.1
  have c44071 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.1
  have c44072 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.1
  have c44073 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.1
  have c44074 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44075 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_701_6 := arithEq_of_rows h (row := 701) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_6
  simp only [← c44070, ← c44071, ← c44072] at e_701_6
  have e_701_7 := arithEq_of_rows h (row := 701) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_7
  simp only [← c44073, ← c44074, k2, ← c44075] at e_701_7
  have hr := e_701_7
  simp only [e_701_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 701 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4459)) := by
  have c44076 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44077 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44078 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44079 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44080 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44081 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_701_8 := arithEq_of_rows h (row := 701) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_8
  simp only [← c44076, ← c44077, ← c44078] at e_701_8
  have e_701_9 := arithEq_of_rows h (row := 701) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_9
  simp only [← c44079, ← c44080, k2, ← c44081] at e_701_9
  have hr := e_701_9
  simp only [e_701_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 701 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4460)) := by
  have c44082 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44083 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44084 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44085 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44086 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44087 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_701_10 := arithEq_of_rows h (row := 701) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_10
  simp only [← c44082, ← c44083, ← c44084] at e_701_10
  have e_701_11 := arithEq_of_rows h (row := 701) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_11
  simp only [← c44085, ← c44086, k2, ← c44087] at e_701_11
  have hr := e_701_11
  simp only [e_701_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 701 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4461)) := by
  have c44088 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44089 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44090 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44091 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44092 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44093 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_701_12 := arithEq_of_rows h (row := 701) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_12
  simp only [← c44088, ← c44089, ← c44090] at e_701_12
  have e_701_13 := arithEq_of_rows h (row := 701) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_13
  simp only [← c44091, ← c44092, k2, ← c44093] at e_701_13
  have hr := e_701_13
  simp only [e_701_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 701 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4462)) := by
  have c44094 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44095 := (publicBatchWrapper89_copies1377 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44096 := (publicBatchWrapper89_copies1378 a h).1
  have c44097 := (publicBatchWrapper89_copies1378 a h).2.1
  have c44098 := (publicBatchWrapper89_copies1378 a h).2.2.1
  have c44099 := (publicBatchWrapper89_copies1378 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_701_14 := arithEq_of_rows h (row := 701) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_14
  simp only [← c44094, ← c44095, ← c44096] at e_701_14
  have e_701_15 := arithEq_of_rows h (row := 701) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_15
  simp only [← c44097, ← c44098, k2, ← c44099] at e_701_15
  have hr := e_701_15
  simp only [e_701_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 701 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4463)) := by
  have c44100 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.1
  have c44101 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.1
  have c44102 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.1
  have c44103 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.1
  have c44104 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.1
  have c44105 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_701_16 := arithEq_of_rows h (row := 701) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_16
  simp only [← c44100, ← c44101, ← c44102] at e_701_16
  have e_701_17 := arithEq_of_rows h (row := 701) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_17
  simp only [← c44103, ← c44104, k2, ← c44105] at e_701_17
  have hr := e_701_17
  simp only [e_701_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 701 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4464)) := by
  have c44106 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44107 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44108 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44109 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44110 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44111 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_701_18 := arithEq_of_rows h (row := 701) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_18
  simp only [← c44106, ← c44107, ← c44108] at e_701_18
  have e_701_19 := arithEq_of_rows h (row := 701) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_701_19
  simp only [← c44109, ← c44110, k2, ← c44111] at e_701_19
  have hr := e_701_19
  simp only [e_701_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 702 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4465)) := by
  have c44112 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44113 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44114 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44115 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44116 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44117 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_702_0 := arithEq_of_rows h (row := 702) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_0
  simp only [← c44112, ← c44113, ← c44114] at e_702_0
  have e_702_1 := arithEq_of_rows h (row := 702) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_1
  simp only [← c44115, ← c44116, k2, ← c44117] at e_702_1
  have hr := e_702_1
  simp only [e_702_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 702 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4466)) := by
  have c44118 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44119 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44120 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44121 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44122 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44123 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_702_2 := arithEq_of_rows h (row := 702) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_2
  simp only [← c44118, ← c44119, ← c44120] at e_702_2
  have e_702_3 := arithEq_of_rows h (row := 702) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_3
  simp only [← c44121, ← c44122, k2, ← c44123] at e_702_3
  have hr := e_702_3
  simp only [e_702_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 702 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4467)) := by
  have c44124 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44125 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44126 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44127 := (publicBatchWrapper89_copies1378 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44128 := (publicBatchWrapper89_copies1379 a h).1
  have c44129 := (publicBatchWrapper89_copies1379 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_702_4 := arithEq_of_rows h (row := 702) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_4
  simp only [← c44124, ← c44125, ← c44126] at e_702_4
  have e_702_5 := arithEq_of_rows h (row := 702) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_5
  simp only [← c44127, ← c44128, k2, ← c44129] at e_702_5
  have hr := e_702_5
  simp only [e_702_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 702 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4468)) := by
  have c44130 := (publicBatchWrapper89_copies1379 a h).2.2.1
  have c44131 := (publicBatchWrapper89_copies1379 a h).2.2.2.1
  have c44132 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.1
  have c44133 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.1
  have c44134 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.1
  have c44135 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_702_6 := arithEq_of_rows h (row := 702) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_6
  simp only [← c44130, ← c44131, ← c44132] at e_702_6
  have e_702_7 := arithEq_of_rows h (row := 702) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_7
  simp only [← c44133, ← c44134, k2, ← c44135] at e_702_7
  have hr := e_702_7
  simp only [e_702_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 702 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4469)) := by
  have c44136 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.1
  have c44137 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.1
  have c44138 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44139 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44140 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44141 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_702_8 := arithEq_of_rows h (row := 702) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_8
  simp only [← c44136, ← c44137, ← c44138] at e_702_8
  have e_702_9 := arithEq_of_rows h (row := 702) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_9
  simp only [← c44139, ← c44140, k2, ← c44141] at e_702_9
  have hr := e_702_9
  simp only [e_702_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 702 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4470)) := by
  have c44142 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44143 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44144 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44145 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44146 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44147 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_702_10 := arithEq_of_rows h (row := 702) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_10
  simp only [← c44142, ← c44143, ← c44144] at e_702_10
  have e_702_11 := arithEq_of_rows h (row := 702) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_11
  simp only [← c44145, ← c44146, k2, ← c44147] at e_702_11
  have hr := e_702_11
  simp only [e_702_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 702 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4471)) := by
  have c44148 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44149 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44150 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44151 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44152 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44153 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_702_12 := arithEq_of_rows h (row := 702) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_12
  simp only [← c44148, ← c44149, ← c44150] at e_702_12
  have e_702_13 := arithEq_of_rows h (row := 702) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_13
  simp only [← c44151, ← c44152, k2, ← c44153] at e_702_13
  have hr := e_702_13
  simp only [e_702_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 702 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4472)) := by
  have c44154 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44155 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44156 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44157 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44158 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44159 := (publicBatchWrapper89_copies1379 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_702_14 := arithEq_of_rows h (row := 702) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_14
  simp only [← c44154, ← c44155, ← c44156] at e_702_14
  have e_702_15 := arithEq_of_rows h (row := 702) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_15
  simp only [← c44157, ← c44158, k2, ← c44159] at e_702_15
  have hr := e_702_15
  simp only [e_702_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 702 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4473)) := by
  have c44160 := (publicBatchWrapper89_copies1380 a h).1
  have c44161 := (publicBatchWrapper89_copies1380 a h).2.1
  have c44162 := (publicBatchWrapper89_copies1380 a h).2.2.1
  have c44163 := (publicBatchWrapper89_copies1380 a h).2.2.2.1
  have c44164 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.1
  have c44165 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_702_16 := arithEq_of_rows h (row := 702) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_16
  simp only [← c44160, ← c44161, ← c44162] at e_702_16
  have e_702_17 := arithEq_of_rows h (row := 702) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_17
  simp only [← c44163, ← c44164, k2, ← c44165] at e_702_17
  have hr := e_702_17
  simp only [e_702_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 702 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4474)) := by
  have c44166 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.1
  have c44167 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.1
  have c44168 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.1
  have c44169 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.1
  have c44170 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44171 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_702_18 := arithEq_of_rows h (row := 702) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_18
  simp only [← c44166, ← c44167, ← c44168] at e_702_18
  have e_702_19 := arithEq_of_rows h (row := 702) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_702_19
  simp only [← c44169, ← c44170, k2, ← c44171] at e_702_19
  have hr := e_702_19
  simp only [e_702_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 703 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4475)) := by
  have c44172 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44173 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44174 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44175 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44176 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44177 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_703_0 := arithEq_of_rows h (row := 703) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_0
  simp only [← c44172, ← c44173, ← c44174] at e_703_0
  have e_703_1 := arithEq_of_rows h (row := 703) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_1
  simp only [← c44175, ← c44176, k2, ← c44177] at e_703_1
  have hr := e_703_1
  simp only [e_703_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 703 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4476)) := by
  have c44178 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44179 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44180 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44181 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44182 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44183 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_703_2 := arithEq_of_rows h (row := 703) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_2
  simp only [← c44178, ← c44179, ← c44180] at e_703_2
  have e_703_3 := arithEq_of_rows h (row := 703) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_3
  simp only [← c44181, ← c44182, k2, ← c44183] at e_703_3
  have hr := e_703_3
  simp only [e_703_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 703 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4477)) := by
  have c44184 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44185 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44186 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44187 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44188 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44189 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_703_4 := arithEq_of_rows h (row := 703) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_4
  simp only [← c44184, ← c44185, ← c44186] at e_703_4
  have e_703_5 := arithEq_of_rows h (row := 703) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_5
  simp only [← c44187, ← c44188, k2, ← c44189] at e_703_5
  have hr := e_703_5
  simp only [e_703_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 703 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4478)) := by
  have c44190 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44191 := (publicBatchWrapper89_copies1380 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44192 := (publicBatchWrapper89_copies1381 a h).1
  have c44193 := (publicBatchWrapper89_copies1381 a h).2.1
  have c44194 := (publicBatchWrapper89_copies1381 a h).2.2.1
  have c44195 := (publicBatchWrapper89_copies1381 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_703_6 := arithEq_of_rows h (row := 703) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_6
  simp only [← c44190, ← c44191, ← c44192] at e_703_6
  have e_703_7 := arithEq_of_rows h (row := 703) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_7
  simp only [← c44193, ← c44194, k2, ← c44195] at e_703_7
  have hr := e_703_7
  simp only [e_703_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 703 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4479)) := by
  have c44196 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.1
  have c44197 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.1
  have c44198 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.1
  have c44199 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.1
  have c44200 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.1
  have c44201 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_703_8 := arithEq_of_rows h (row := 703) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_8
  simp only [← c44196, ← c44197, ← c44198] at e_703_8
  have e_703_9 := arithEq_of_rows h (row := 703) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_9
  simp only [← c44199, ← c44200, k2, ← c44201] at e_703_9
  have hr := e_703_9
  simp only [e_703_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 703 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4480)) := by
  have c44202 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44203 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44204 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44205 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44206 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44207 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_703_10 := arithEq_of_rows h (row := 703) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_10
  simp only [← c44202, ← c44203, ← c44204] at e_703_10
  have e_703_11 := arithEq_of_rows h (row := 703) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_11
  simp only [← c44205, ← c44206, k2, ← c44207] at e_703_11
  have hr := e_703_11
  simp only [e_703_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 703 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4481)) := by
  have c44208 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44209 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44210 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44211 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44212 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44213 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_703_12 := arithEq_of_rows h (row := 703) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_12
  simp only [← c44208, ← c44209, ← c44210] at e_703_12
  have e_703_13 := arithEq_of_rows h (row := 703) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_13
  simp only [← c44211, ← c44212, k2, ← c44213] at e_703_13
  have hr := e_703_13
  simp only [e_703_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 703 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4482)) := by
  have c44214 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44215 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44216 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44217 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44218 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44219 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_703_14 := arithEq_of_rows h (row := 703) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_14
  simp only [← c44214, ← c44215, ← c44216] at e_703_14
  have e_703_15 := arithEq_of_rows h (row := 703) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_15
  simp only [← c44217, ← c44218, k2, ← c44219] at e_703_15
  have hr := e_703_15
  simp only [e_703_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 703 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4483)) := by
  have c44220 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44221 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44222 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44223 := (publicBatchWrapper89_copies1381 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44224 := (publicBatchWrapper89_copies1382 a h).1
  have c44225 := (publicBatchWrapper89_copies1382 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_703_16 := arithEq_of_rows h (row := 703) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_16
  simp only [← c44220, ← c44221, ← c44222] at e_703_16
  have e_703_17 := arithEq_of_rows h (row := 703) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_17
  simp only [← c44223, ← c44224, k2, ← c44225] at e_703_17
  have hr := e_703_17
  simp only [e_703_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 703 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4484)) := by
  have c44226 := (publicBatchWrapper89_copies1382 a h).2.2.1
  have c44227 := (publicBatchWrapper89_copies1382 a h).2.2.2.1
  have c44228 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.1
  have c44229 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.1
  have c44230 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.1
  have c44231 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_703_18 := arithEq_of_rows h (row := 703) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_18
  simp only [← c44226, ← c44227, ← c44228] at e_703_18
  have e_703_19 := arithEq_of_rows h (row := 703) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_703_19
  simp only [← c44229, ← c44230, k2, ← c44231] at e_703_19
  have hr := e_703_19
  simp only [e_703_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 704 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4485)) := by
  have c44232 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.1
  have c44233 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.1
  have c44234 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44235 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44236 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44237 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_704_0 := arithEq_of_rows h (row := 704) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_0
  simp only [← c44232, ← c44233, ← c44234] at e_704_0
  have e_704_1 := arithEq_of_rows h (row := 704) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_1
  simp only [← c44235, ← c44236, k2, ← c44237] at e_704_1
  have hr := e_704_1
  simp only [e_704_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 704 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4486)) := by
  have c44238 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44239 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44240 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44241 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44242 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44243 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_704_2 := arithEq_of_rows h (row := 704) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_2
  simp only [← c44238, ← c44239, ← c44240] at e_704_2
  have e_704_3 := arithEq_of_rows h (row := 704) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_3
  simp only [← c44241, ← c44242, k2, ← c44243] at e_704_3
  have hr := e_704_3
  simp only [e_704_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 704 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4487)) := by
  have c44244 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44245 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44246 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44247 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44248 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44249 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_704_4 := arithEq_of_rows h (row := 704) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_4
  simp only [← c44244, ← c44245, ← c44246] at e_704_4
  have e_704_5 := arithEq_of_rows h (row := 704) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_5
  simp only [← c44247, ← c44248, k2, ← c44249] at e_704_5
  have hr := e_704_5
  simp only [e_704_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 704 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4488)) := by
  have c44250 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44251 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44252 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44253 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44254 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44255 := (publicBatchWrapper89_copies1382 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_704_6 := arithEq_of_rows h (row := 704) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_6
  simp only [← c44250, ← c44251, ← c44252] at e_704_6
  have e_704_7 := arithEq_of_rows h (row := 704) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_7
  simp only [← c44253, ← c44254, k2, ← c44255] at e_704_7
  have hr := e_704_7
  simp only [e_704_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 704 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4489)) := by
  have c44256 := (publicBatchWrapper89_copies1383 a h).1
  have c44257 := (publicBatchWrapper89_copies1383 a h).2.1
  have c44258 := (publicBatchWrapper89_copies1383 a h).2.2.1
  have c44259 := (publicBatchWrapper89_copies1383 a h).2.2.2.1
  have c44260 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.1
  have c44261 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_704_8 := arithEq_of_rows h (row := 704) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_8
  simp only [← c44256, ← c44257, ← c44258] at e_704_8
  have e_704_9 := arithEq_of_rows h (row := 704) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_9
  simp only [← c44259, ← c44260, k2, ← c44261] at e_704_9
  have hr := e_704_9
  simp only [e_704_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 704 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4490)) := by
  have c44262 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.1
  have c44263 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.1
  have c44264 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.1
  have c44265 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.1
  have c44266 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44267 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_704_10 := arithEq_of_rows h (row := 704) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_10
  simp only [← c44262, ← c44263, ← c44264] at e_704_10
  have e_704_11 := arithEq_of_rows h (row := 704) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_11
  simp only [← c44265, ← c44266, k2, ← c44267] at e_704_11
  have hr := e_704_11
  simp only [e_704_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 704 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4491)) := by
  have c44268 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44269 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44270 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44271 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44272 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44273 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_704_12 := arithEq_of_rows h (row := 704) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_12
  simp only [← c44268, ← c44269, ← c44270] at e_704_12
  have e_704_13 := arithEq_of_rows h (row := 704) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_13
  simp only [← c44271, ← c44272, k2, ← c44273] at e_704_13
  have hr := e_704_13
  simp only [e_704_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 704 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4492)) := by
  have c44274 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44275 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44276 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44277 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44278 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44279 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_704_14 := arithEq_of_rows h (row := 704) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_14
  simp only [← c44274, ← c44275, ← c44276] at e_704_14
  have e_704_15 := arithEq_of_rows h (row := 704) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_15
  simp only [← c44277, ← c44278, k2, ← c44279] at e_704_15
  have hr := e_704_15
  simp only [e_704_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 704 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4493)) := by
  have c44280 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44281 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44282 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44283 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44284 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44285 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_704_16 := arithEq_of_rows h (row := 704) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_16
  simp only [← c44280, ← c44281, ← c44282] at e_704_16
  have e_704_17 := arithEq_of_rows h (row := 704) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_17
  simp only [← c44283, ← c44284, k2, ← c44285] at e_704_17
  have hr := e_704_17
  simp only [e_704_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 704 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4494)) := by
  have c44286 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44287 := (publicBatchWrapper89_copies1383 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44288 := (publicBatchWrapper89_copies1384 a h).1
  have c44289 := (publicBatchWrapper89_copies1384 a h).2.1
  have c44290 := (publicBatchWrapper89_copies1384 a h).2.2.1
  have c44291 := (publicBatchWrapper89_copies1384 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_704_18 := arithEq_of_rows h (row := 704) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_18
  simp only [← c44286, ← c44287, ← c44288] at e_704_18
  have e_704_19 := arithEq_of_rows h (row := 704) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_704_19
  simp only [← c44289, ← c44290, k2, ← c44291] at e_704_19
  have hr := e_704_19
  simp only [e_704_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 705 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4495)) := by
  have c44292 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.1
  have c44293 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.1
  have c44294 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.1
  have c44295 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.1
  have c44296 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.1
  have c44297 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_705_0 := arithEq_of_rows h (row := 705) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_0
  simp only [← c44292, ← c44293, ← c44294] at e_705_0
  have e_705_1 := arithEq_of_rows h (row := 705) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_1
  simp only [← c44295, ← c44296, k2, ← c44297] at e_705_1
  have hr := e_705_1
  simp only [e_705_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 705 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4496)) := by
  have c44298 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44299 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44300 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44301 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44302 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44303 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_705_2 := arithEq_of_rows h (row := 705) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_2
  simp only [← c44298, ← c44299, ← c44300] at e_705_2
  have e_705_3 := arithEq_of_rows h (row := 705) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_3
  simp only [← c44301, ← c44302, k2, ← c44303] at e_705_3
  have hr := e_705_3
  simp only [e_705_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 705 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4497)) := by
  have c44304 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44305 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44306 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44307 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44308 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44309 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_705_4 := arithEq_of_rows h (row := 705) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_4
  simp only [← c44304, ← c44305, ← c44306] at e_705_4
  have e_705_5 := arithEq_of_rows h (row := 705) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_5
  simp only [← c44307, ← c44308, k2, ← c44309] at e_705_5
  have hr := e_705_5
  simp only [e_705_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 705 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4498)) := by
  have c44310 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44311 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44312 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44313 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44314 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44315 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_705_6 := arithEq_of_rows h (row := 705) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_6
  simp only [← c44310, ← c44311, ← c44312] at e_705_6
  have e_705_7 := arithEq_of_rows h (row := 705) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_7
  simp only [← c44313, ← c44314, k2, ← c44315] at e_705_7
  have hr := e_705_7
  simp only [e_705_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 705 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4499)) := by
  have c44316 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44317 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44318 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44319 := (publicBatchWrapper89_copies1384 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44320 := (publicBatchWrapper89_copies1385 a h).1
  have c44321 := (publicBatchWrapper89_copies1385 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_705_8 := arithEq_of_rows h (row := 705) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_8
  simp only [← c44316, ← c44317, ← c44318] at e_705_8
  have e_705_9 := arithEq_of_rows h (row := 705) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_9
  simp only [← c44319, ← c44320, k2, ← c44321] at e_705_9
  have hr := e_705_9
  simp only [e_705_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 705 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4500)) := by
  have c44322 := (publicBatchWrapper89_copies1385 a h).2.2.1
  have c44323 := (publicBatchWrapper89_copies1385 a h).2.2.2.1
  have c44324 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.1
  have c44325 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.1
  have c44326 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.1
  have c44327 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_705_10 := arithEq_of_rows h (row := 705) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_10
  simp only [← c44322, ← c44323, ← c44324] at e_705_10
  have e_705_11 := arithEq_of_rows h (row := 705) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_11
  simp only [← c44325, ← c44326, k2, ← c44327] at e_705_11
  have hr := e_705_11
  simp only [e_705_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 705 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4501)) := by
  have c44328 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.1
  have c44329 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.1
  have c44330 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44331 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44332 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44333 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_705_12 := arithEq_of_rows h (row := 705) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_12
  simp only [← c44328, ← c44329, ← c44330] at e_705_12
  have e_705_13 := arithEq_of_rows h (row := 705) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_13
  simp only [← c44331, ← c44332, k2, ← c44333] at e_705_13
  have hr := e_705_13
  simp only [e_705_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 705 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4502)) := by
  have c44334 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44335 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44336 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44337 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44338 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44339 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_705_14 := arithEq_of_rows h (row := 705) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_14
  simp only [← c44334, ← c44335, ← c44336] at e_705_14
  have e_705_15 := arithEq_of_rows h (row := 705) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_15
  simp only [← c44337, ← c44338, k2, ← c44339] at e_705_15
  have hr := e_705_15
  simp only [e_705_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 705 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4503)) := by
  have c44340 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44341 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44342 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44343 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44344 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44345 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_705_16 := arithEq_of_rows h (row := 705) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_16
  simp only [← c44340, ← c44341, ← c44342] at e_705_16
  have e_705_17 := arithEq_of_rows h (row := 705) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_17
  simp only [← c44343, ← c44344, k2, ← c44345] at e_705_17
  have hr := e_705_17
  simp only [e_705_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 705 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4504)) := by
  have c44346 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44347 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44348 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44349 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44350 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44351 := (publicBatchWrapper89_copies1385 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_705_18 := arithEq_of_rows h (row := 705) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_18
  simp only [← c44346, ← c44347, ← c44348] at e_705_18
  have e_705_19 := arithEq_of_rows h (row := 705) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_705_19
  simp only [← c44349, ← c44350, k2, ← c44351] at e_705_19
  have hr := e_705_19
  simp only [e_705_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 706 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4505)) := by
  have c44352 := (publicBatchWrapper89_copies1386 a h).1
  have c44353 := (publicBatchWrapper89_copies1386 a h).2.1
  have c44354 := (publicBatchWrapper89_copies1386 a h).2.2.1
  have c44355 := (publicBatchWrapper89_copies1386 a h).2.2.2.1
  have c44356 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.1
  have c44357 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_706_0 := arithEq_of_rows h (row := 706) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_0
  simp only [← c44352, ← c44353, ← c44354] at e_706_0
  have e_706_1 := arithEq_of_rows h (row := 706) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_1
  simp only [← c44355, ← c44356, k2, ← c44357] at e_706_1
  have hr := e_706_1
  simp only [e_706_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 706 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4506)) := by
  have c44358 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.1
  have c44359 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.1
  have c44360 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.1
  have c44361 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.1
  have c44362 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44363 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_706_2 := arithEq_of_rows h (row := 706) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_2
  simp only [← c44358, ← c44359, ← c44360] at e_706_2
  have e_706_3 := arithEq_of_rows h (row := 706) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_3
  simp only [← c44361, ← c44362, k2, ← c44363] at e_706_3
  have hr := e_706_3
  simp only [e_706_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 706 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4507)) := by
  have c44364 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44365 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44366 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44367 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44368 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44369 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_706_4 := arithEq_of_rows h (row := 706) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_4
  simp only [← c44364, ← c44365, ← c44366] at e_706_4
  have e_706_5 := arithEq_of_rows h (row := 706) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_5
  simp only [← c44367, ← c44368, k2, ← c44369] at e_706_5
  have hr := e_706_5
  simp only [e_706_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 706 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4508)) := by
  have c44370 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44371 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44372 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44373 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44374 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44375 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_706_6 := arithEq_of_rows h (row := 706) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_6
  simp only [← c44370, ← c44371, ← c44372] at e_706_6
  have e_706_7 := arithEq_of_rows h (row := 706) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_7
  simp only [← c44373, ← c44374, k2, ← c44375] at e_706_7
  have hr := e_706_7
  simp only [e_706_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 706 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4509)) := by
  have c44376 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44377 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44378 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44379 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44380 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44381 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_706_8 := arithEq_of_rows h (row := 706) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_8
  simp only [← c44376, ← c44377, ← c44378] at e_706_8
  have e_706_9 := arithEq_of_rows h (row := 706) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_9
  simp only [← c44379, ← c44380, k2, ← c44381] at e_706_9
  have hr := e_706_9
  simp only [e_706_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 706 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4510)) := by
  have c44382 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44383 := (publicBatchWrapper89_copies1386 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44384 := (publicBatchWrapper89_copies1387 a h).1
  have c44385 := (publicBatchWrapper89_copies1387 a h).2.1
  have c44386 := (publicBatchWrapper89_copies1387 a h).2.2.1
  have c44387 := (publicBatchWrapper89_copies1387 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_706_10 := arithEq_of_rows h (row := 706) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_10
  simp only [← c44382, ← c44383, ← c44384] at e_706_10
  have e_706_11 := arithEq_of_rows h (row := 706) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_11
  simp only [← c44385, ← c44386, k2, ← c44387] at e_706_11
  have hr := e_706_11
  simp only [e_706_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 706 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4511)) := by
  have c44388 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.1
  have c44389 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.1
  have c44390 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.1
  have c44391 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.1
  have c44392 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.1
  have c44393 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_706_12 := arithEq_of_rows h (row := 706) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_12
  simp only [← c44388, ← c44389, ← c44390] at e_706_12
  have e_706_13 := arithEq_of_rows h (row := 706) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_13
  simp only [← c44391, ← c44392, k2, ← c44393] at e_706_13
  have hr := e_706_13
  simp only [e_706_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 706 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4512)) := by
  have c44394 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44395 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44396 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44397 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44398 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44399 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_706_14 := arithEq_of_rows h (row := 706) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_14
  simp only [← c44394, ← c44395, ← c44396] at e_706_14
  have e_706_15 := arithEq_of_rows h (row := 706) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_15
  simp only [← c44397, ← c44398, k2, ← c44399] at e_706_15
  have hr := e_706_15
  simp only [e_706_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 706 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4513)) := by
  have c44400 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44401 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44402 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44403 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44404 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44405 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_706_16 := arithEq_of_rows h (row := 706) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_16
  simp only [← c44400, ← c44401, ← c44402] at e_706_16
  have e_706_17 := arithEq_of_rows h (row := 706) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_17
  simp only [← c44403, ← c44404, k2, ← c44405] at e_706_17
  have hr := e_706_17
  simp only [e_706_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 706 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4514)) := by
  have c44406 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44407 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44408 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44409 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44410 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44411 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_706_18 := arithEq_of_rows h (row := 706) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_18
  simp only [← c44406, ← c44407, ← c44408] at e_706_18
  have e_706_19 := arithEq_of_rows h (row := 706) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_706_19
  simp only [← c44409, ← c44410, k2, ← c44411] at e_706_19
  have hr := e_706_19
  simp only [e_706_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 707 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4515)) := by
  have c44412 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44413 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44414 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44415 := (publicBatchWrapper89_copies1387 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44416 := (publicBatchWrapper89_copies1388 a h).1
  have c44417 := (publicBatchWrapper89_copies1388 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_707_0 := arithEq_of_rows h (row := 707) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_0
  simp only [← c44412, ← c44413, ← c44414] at e_707_0
  have e_707_1 := arithEq_of_rows h (row := 707) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_1
  simp only [← c44415, ← c44416, k2, ← c44417] at e_707_1
  have hr := e_707_1
  simp only [e_707_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 707 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4516)) := by
  have c44418 := (publicBatchWrapper89_copies1388 a h).2.2.1
  have c44419 := (publicBatchWrapper89_copies1388 a h).2.2.2.1
  have c44420 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.1
  have c44421 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.1
  have c44422 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.1
  have c44423 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_707_2 := arithEq_of_rows h (row := 707) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_2
  simp only [← c44418, ← c44419, ← c44420] at e_707_2
  have e_707_3 := arithEq_of_rows h (row := 707) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_3
  simp only [← c44421, ← c44422, k2, ← c44423] at e_707_3
  have hr := e_707_3
  simp only [e_707_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 707 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4517)) := by
  have c44424 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.1
  have c44425 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.1
  have c44426 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44427 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44428 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44429 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_707_4 := arithEq_of_rows h (row := 707) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_4
  simp only [← c44424, ← c44425, ← c44426] at e_707_4
  have e_707_5 := arithEq_of_rows h (row := 707) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_5
  simp only [← c44427, ← c44428, k2, ← c44429] at e_707_5
  have hr := e_707_5
  simp only [e_707_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 707 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4518)) := by
  have c44430 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44431 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44432 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44433 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44434 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44435 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_707_6 := arithEq_of_rows h (row := 707) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_6
  simp only [← c44430, ← c44431, ← c44432] at e_707_6
  have e_707_7 := arithEq_of_rows h (row := 707) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_7
  simp only [← c44433, ← c44434, k2, ← c44435] at e_707_7
  have hr := e_707_7
  simp only [e_707_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 707 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4519)) := by
  have c44436 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44437 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44438 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44439 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44440 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44441 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_707_8 := arithEq_of_rows h (row := 707) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_8
  simp only [← c44436, ← c44437, ← c44438] at e_707_8
  have e_707_9 := arithEq_of_rows h (row := 707) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_9
  simp only [← c44439, ← c44440, k2, ← c44441] at e_707_9
  have hr := e_707_9
  simp only [e_707_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 707 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4520)) := by
  have c44442 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44443 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44444 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44445 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44446 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44447 := (publicBatchWrapper89_copies1388 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_707_10 := arithEq_of_rows h (row := 707) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_10
  simp only [← c44442, ← c44443, ← c44444] at e_707_10
  have e_707_11 := arithEq_of_rows h (row := 707) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_11
  simp only [← c44445, ← c44446, k2, ← c44447] at e_707_11
  have hr := e_707_11
  simp only [e_707_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 707 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4521)) := by
  have c44448 := (publicBatchWrapper89_copies1389 a h).1
  have c44449 := (publicBatchWrapper89_copies1389 a h).2.1
  have c44450 := (publicBatchWrapper89_copies1389 a h).2.2.1
  have c44451 := (publicBatchWrapper89_copies1389 a h).2.2.2.1
  have c44452 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.1
  have c44453 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_707_12 := arithEq_of_rows h (row := 707) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_12
  simp only [← c44448, ← c44449, ← c44450] at e_707_12
  have e_707_13 := arithEq_of_rows h (row := 707) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_13
  simp only [← c44451, ← c44452, k2, ← c44453] at e_707_13
  have hr := e_707_13
  simp only [e_707_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 707 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4522)) := by
  have c44454 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.1
  have c44455 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.1
  have c44456 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.1
  have c44457 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.1
  have c44458 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44459 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_707_14 := arithEq_of_rows h (row := 707) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_14
  simp only [← c44454, ← c44455, ← c44456] at e_707_14
  have e_707_15 := arithEq_of_rows h (row := 707) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_15
  simp only [← c44457, ← c44458, k2, ← c44459] at e_707_15
  have hr := e_707_15
  simp only [e_707_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 707 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4523)) := by
  have c44460 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44461 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44462 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44463 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44464 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44465 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_707_16 := arithEq_of_rows h (row := 707) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_16
  simp only [← c44460, ← c44461, ← c44462] at e_707_16
  have e_707_17 := arithEq_of_rows h (row := 707) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_17
  simp only [← c44463, ← c44464, k2, ← c44465] at e_707_17
  have hr := e_707_17
  simp only [e_707_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 707 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4524)) := by
  have c44466 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44467 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44468 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44469 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44470 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44471 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_707_18 := arithEq_of_rows h (row := 707) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_18
  simp only [← c44466, ← c44467, ← c44468] at e_707_18
  have e_707_19 := arithEq_of_rows h (row := 707) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_707_19
  simp only [← c44469, ← c44470, k2, ← c44471] at e_707_19
  have hr := e_707_19
  simp only [e_707_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 708 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4525)) := by
  have c44472 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44473 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44474 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44475 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44476 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44477 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_708_0 := arithEq_of_rows h (row := 708) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_0
  simp only [← c44472, ← c44473, ← c44474] at e_708_0
  have e_708_1 := arithEq_of_rows h (row := 708) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_1
  simp only [← c44475, ← c44476, k2, ← c44477] at e_708_1
  have hr := e_708_1
  simp only [e_708_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 708 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4526)) := by
  have c44478 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44479 := (publicBatchWrapper89_copies1389 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44480 := (publicBatchWrapper89_copies1390 a h).1
  have c44481 := (publicBatchWrapper89_copies1390 a h).2.1
  have c44482 := (publicBatchWrapper89_copies1390 a h).2.2.1
  have c44483 := (publicBatchWrapper89_copies1390 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_708_2 := arithEq_of_rows h (row := 708) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_2
  simp only [← c44478, ← c44479, ← c44480] at e_708_2
  have e_708_3 := arithEq_of_rows h (row := 708) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_3
  simp only [← c44481, ← c44482, k2, ← c44483] at e_708_3
  have hr := e_708_3
  simp only [e_708_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 708 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4527)) := by
  have c44484 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.1
  have c44485 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.1
  have c44486 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.1
  have c44487 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.1
  have c44488 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.1
  have c44489 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_708_4 := arithEq_of_rows h (row := 708) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_4
  simp only [← c44484, ← c44485, ← c44486] at e_708_4
  have e_708_5 := arithEq_of_rows h (row := 708) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_5
  simp only [← c44487, ← c44488, k2, ← c44489] at e_708_5
  have hr := e_708_5
  simp only [e_708_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 708 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4528)) := by
  have c44490 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44491 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44492 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44493 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44494 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44495 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_708_6 := arithEq_of_rows h (row := 708) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_6
  simp only [← c44490, ← c44491, ← c44492] at e_708_6
  have e_708_7 := arithEq_of_rows h (row := 708) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_7
  simp only [← c44493, ← c44494, k2, ← c44495] at e_708_7
  have hr := e_708_7
  simp only [e_708_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 708 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4529)) := by
  have c44496 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44497 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44498 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44499 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44500 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44501 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_708_8 := arithEq_of_rows h (row := 708) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_8
  simp only [← c44496, ← c44497, ← c44498] at e_708_8
  have e_708_9 := arithEq_of_rows h (row := 708) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_9
  simp only [← c44499, ← c44500, k2, ← c44501] at e_708_9
  have hr := e_708_9
  simp only [e_708_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 708 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4530)) := by
  have c44502 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44503 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44504 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44505 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44506 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44507 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_708_10 := arithEq_of_rows h (row := 708) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_10
  simp only [← c44502, ← c44503, ← c44504] at e_708_10
  have e_708_11 := arithEq_of_rows h (row := 708) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_11
  simp only [← c44505, ← c44506, k2, ← c44507] at e_708_11
  have hr := e_708_11
  simp only [e_708_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 708 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4531)) := by
  have c44508 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44509 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44510 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44511 := (publicBatchWrapper89_copies1390 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44512 := (publicBatchWrapper89_copies1391 a h).1
  have c44513 := (publicBatchWrapper89_copies1391 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_708_12 := arithEq_of_rows h (row := 708) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_12
  simp only [← c44508, ← c44509, ← c44510] at e_708_12
  have e_708_13 := arithEq_of_rows h (row := 708) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_13
  simp only [← c44511, ← c44512, k2, ← c44513] at e_708_13
  have hr := e_708_13
  simp only [e_708_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 708 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4532)) := by
  have c44514 := (publicBatchWrapper89_copies1391 a h).2.2.1
  have c44515 := (publicBatchWrapper89_copies1391 a h).2.2.2.1
  have c44516 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.1
  have c44517 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.1
  have c44518 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.1
  have c44519 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_708_14 := arithEq_of_rows h (row := 708) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_14
  simp only [← c44514, ← c44515, ← c44516] at e_708_14
  have e_708_15 := arithEq_of_rows h (row := 708) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_15
  simp only [← c44517, ← c44518, k2, ← c44519] at e_708_15
  have hr := e_708_15
  simp only [e_708_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 708 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4533)) := by
  have c44520 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.1
  have c44521 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.1
  have c44522 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44523 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44524 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44525 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_708_16 := arithEq_of_rows h (row := 708) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_16
  simp only [← c44520, ← c44521, ← c44522] at e_708_16
  have e_708_17 := arithEq_of_rows h (row := 708) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_17
  simp only [← c44523, ← c44524, k2, ← c44525] at e_708_17
  have hr := e_708_17
  simp only [e_708_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 708 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4534)) := by
  have c44526 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44527 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44528 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44529 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44530 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44531 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_708_18 := arithEq_of_rows h (row := 708) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_18
  simp only [← c44526, ← c44527, ← c44528] at e_708_18
  have e_708_19 := arithEq_of_rows h (row := 708) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_708_19
  simp only [← c44529, ← c44530, k2, ← c44531] at e_708_19
  have hr := e_708_19
  simp only [e_708_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 709 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4535)) := by
  have c44532 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44533 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44534 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44535 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44536 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44537 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_709_0 := arithEq_of_rows h (row := 709) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_0
  simp only [← c44532, ← c44533, ← c44534] at e_709_0
  have e_709_1 := arithEq_of_rows h (row := 709) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_1
  simp only [← c44535, ← c44536, k2, ← c44537] at e_709_1
  have hr := e_709_1
  simp only [e_709_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 709 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4536)) := by
  have c44538 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44539 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44540 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44541 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44542 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44543 := (publicBatchWrapper89_copies1391 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_709_2 := arithEq_of_rows h (row := 709) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_2
  simp only [← c44538, ← c44539, ← c44540] at e_709_2
  have e_709_3 := arithEq_of_rows h (row := 709) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_3
  simp only [← c44541, ← c44542, k2, ← c44543] at e_709_3
  have hr := e_709_3
  simp only [e_709_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 709 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4537)) := by
  have c44544 := (publicBatchWrapper89_copies1392 a h).1
  have c44545 := (publicBatchWrapper89_copies1392 a h).2.1
  have c44546 := (publicBatchWrapper89_copies1392 a h).2.2.1
  have c44547 := (publicBatchWrapper89_copies1392 a h).2.2.2.1
  have c44548 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.1
  have c44549 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_709_4 := arithEq_of_rows h (row := 709) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_4
  simp only [← c44544, ← c44545, ← c44546] at e_709_4
  have e_709_5 := arithEq_of_rows h (row := 709) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_5
  simp only [← c44547, ← c44548, k2, ← c44549] at e_709_5
  have hr := e_709_5
  simp only [e_709_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 709 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4538)) := by
  have c44550 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.1
  have c44551 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.1
  have c44552 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.1
  have c44553 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.1
  have c44554 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44555 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_709_6 := arithEq_of_rows h (row := 709) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_6
  simp only [← c44550, ← c44551, ← c44552] at e_709_6
  have e_709_7 := arithEq_of_rows h (row := 709) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_7
  simp only [← c44553, ← c44554, k2, ← c44555] at e_709_7
  have hr := e_709_7
  simp only [e_709_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 709 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4539)) := by
  have c44556 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44557 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44558 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44559 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44560 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44561 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_709_8 := arithEq_of_rows h (row := 709) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_8
  simp only [← c44556, ← c44557, ← c44558] at e_709_8
  have e_709_9 := arithEq_of_rows h (row := 709) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_9
  simp only [← c44559, ← c44560, k2, ← c44561] at e_709_9
  have hr := e_709_9
  simp only [e_709_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 709 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4540)) := by
  have c44562 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44563 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44564 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44565 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44566 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44567 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_709_10 := arithEq_of_rows h (row := 709) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_10
  simp only [← c44562, ← c44563, ← c44564] at e_709_10
  have e_709_11 := arithEq_of_rows h (row := 709) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_11
  simp only [← c44565, ← c44566, k2, ← c44567] at e_709_11
  have hr := e_709_11
  simp only [e_709_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 709 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4541)) := by
  have c44568 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44569 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44570 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44571 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44572 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44573 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_709_12 := arithEq_of_rows h (row := 709) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_12
  simp only [← c44568, ← c44569, ← c44570] at e_709_12
  have e_709_13 := arithEq_of_rows h (row := 709) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_13
  simp only [← c44571, ← c44572, k2, ← c44573] at e_709_13
  have hr := e_709_13
  simp only [e_709_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 709 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4542)) := by
  have c44574 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44575 := (publicBatchWrapper89_copies1392 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44576 := (publicBatchWrapper89_copies1393 a h).1
  have c44577 := (publicBatchWrapper89_copies1393 a h).2.1
  have c44578 := (publicBatchWrapper89_copies1393 a h).2.2.1
  have c44579 := (publicBatchWrapper89_copies1393 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_709_14 := arithEq_of_rows h (row := 709) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_14
  simp only [← c44574, ← c44575, ← c44576] at e_709_14
  have e_709_15 := arithEq_of_rows h (row := 709) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_15
  simp only [← c44577, ← c44578, k2, ← c44579] at e_709_15
  have hr := e_709_15
  simp only [e_709_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 709 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4543)) := by
  have c44580 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.1
  have c44581 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.1
  have c44582 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.1
  have c44583 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.1
  have c44584 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.1
  have c44585 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_709_16 := arithEq_of_rows h (row := 709) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_16
  simp only [← c44580, ← c44581, ← c44582] at e_709_16
  have e_709_17 := arithEq_of_rows h (row := 709) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_17
  simp only [← c44583, ← c44584, k2, ← c44585] at e_709_17
  have hr := e_709_17
  simp only [e_709_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 709 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4544)) := by
  have c44586 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44587 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44588 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44589 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44590 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44591 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_709_18 := arithEq_of_rows h (row := 709) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_18
  simp only [← c44586, ← c44587, ← c44588] at e_709_18
  have e_709_19 := arithEq_of_rows h (row := 709) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_709_19
  simp only [← c44589, ← c44590, k2, ← c44591] at e_709_19
  have hr := e_709_19
  simp only [e_709_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 710 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4545)) := by
  have c44592 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44593 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44594 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44595 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44596 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44597 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_710_0 := arithEq_of_rows h (row := 710) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_0
  simp only [← c44592, ← c44593, ← c44594] at e_710_0
  have e_710_1 := arithEq_of_rows h (row := 710) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_1
  simp only [← c44595, ← c44596, k2, ← c44597] at e_710_1
  have hr := e_710_1
  simp only [e_710_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 710 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4546)) := by
  have c44598 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44599 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44600 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44601 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44602 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44603 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_710_2 := arithEq_of_rows h (row := 710) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_2
  simp only [← c44598, ← c44599, ← c44600] at e_710_2
  have e_710_3 := arithEq_of_rows h (row := 710) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_3
  simp only [← c44601, ← c44602, k2, ← c44603] at e_710_3
  have hr := e_710_3
  simp only [e_710_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 710 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4547)) := by
  have c44604 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44605 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44606 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44607 := (publicBatchWrapper89_copies1393 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44608 := (publicBatchWrapper89_copies1394 a h).1
  have c44609 := (publicBatchWrapper89_copies1394 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_710_4 := arithEq_of_rows h (row := 710) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_4
  simp only [← c44604, ← c44605, ← c44606] at e_710_4
  have e_710_5 := arithEq_of_rows h (row := 710) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_5
  simp only [← c44607, ← c44608, k2, ← c44609] at e_710_5
  have hr := e_710_5
  simp only [e_710_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 710 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4548)) := by
  have c44610 := (publicBatchWrapper89_copies1394 a h).2.2.1
  have c44611 := (publicBatchWrapper89_copies1394 a h).2.2.2.1
  have c44612 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.1
  have c44613 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.1
  have c44614 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.1
  have c44615 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_710_6 := arithEq_of_rows h (row := 710) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_6
  simp only [← c44610, ← c44611, ← c44612] at e_710_6
  have e_710_7 := arithEq_of_rows h (row := 710) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_7
  simp only [← c44613, ← c44614, k2, ← c44615] at e_710_7
  have hr := e_710_7
  simp only [e_710_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 710 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4549)) := by
  have c44616 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.1
  have c44617 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.1
  have c44618 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44619 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44620 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44621 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_710_8 := arithEq_of_rows h (row := 710) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_8
  simp only [← c44616, ← c44617, ← c44618] at e_710_8
  have e_710_9 := arithEq_of_rows h (row := 710) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_9
  simp only [← c44619, ← c44620, k2, ← c44621] at e_710_9
  have hr := e_710_9
  simp only [e_710_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 710 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4550)) := by
  have c44622 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44623 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44624 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44625 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44626 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44627 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_710_10 := arithEq_of_rows h (row := 710) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_10
  simp only [← c44622, ← c44623, ← c44624] at e_710_10
  have e_710_11 := arithEq_of_rows h (row := 710) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_11
  simp only [← c44625, ← c44626, k2, ← c44627] at e_710_11
  have hr := e_710_11
  simp only [e_710_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 710 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4551)) := by
  have c44628 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44629 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44630 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44631 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44632 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44633 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_710_12 := arithEq_of_rows h (row := 710) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_12
  simp only [← c44628, ← c44629, ← c44630] at e_710_12
  have e_710_13 := arithEq_of_rows h (row := 710) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_13
  simp only [← c44631, ← c44632, k2, ← c44633] at e_710_13
  have hr := e_710_13
  simp only [e_710_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 710 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4552)) := by
  have c44634 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44635 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44636 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44637 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44638 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44639 := (publicBatchWrapper89_copies1394 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_710_14 := arithEq_of_rows h (row := 710) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_14
  simp only [← c44634, ← c44635, ← c44636] at e_710_14
  have e_710_15 := arithEq_of_rows h (row := 710) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_15
  simp only [← c44637, ← c44638, k2, ← c44639] at e_710_15
  have hr := e_710_15
  simp only [e_710_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 710 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4553)) := by
  have c44640 := (publicBatchWrapper89_copies1395 a h).1
  have c44641 := (publicBatchWrapper89_copies1395 a h).2.1
  have c44642 := (publicBatchWrapper89_copies1395 a h).2.2.1
  have c44643 := (publicBatchWrapper89_copies1395 a h).2.2.2.1
  have c44644 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.1
  have c44645 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_710_16 := arithEq_of_rows h (row := 710) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_16
  simp only [← c44640, ← c44641, ← c44642] at e_710_16
  have e_710_17 := arithEq_of_rows h (row := 710) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_17
  simp only [← c44643, ← c44644, k2, ← c44645] at e_710_17
  have hr := e_710_17
  simp only [e_710_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 710 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4554)) := by
  have c44646 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.1
  have c44647 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.1
  have c44648 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.1
  have c44649 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.1
  have c44650 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44651 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_710_18 := arithEq_of_rows h (row := 710) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_18
  simp only [← c44646, ← c44647, ← c44648] at e_710_18
  have e_710_19 := arithEq_of_rows h (row := 710) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_710_19
  simp only [← c44649, ← c44650, k2, ← c44651] at e_710_19
  have hr := e_710_19
  simp only [e_710_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 711 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4555)) := by
  have c44652 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44653 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44654 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44655 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44656 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44657 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_711_0 := arithEq_of_rows h (row := 711) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_0
  simp only [← c44652, ← c44653, ← c44654] at e_711_0
  have e_711_1 := arithEq_of_rows h (row := 711) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_1
  simp only [← c44655, ← c44656, k2, ← c44657] at e_711_1
  have hr := e_711_1
  simp only [e_711_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 711 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4556)) := by
  have c44658 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44659 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44660 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44661 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44662 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44663 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_711_2 := arithEq_of_rows h (row := 711) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_2
  simp only [← c44658, ← c44659, ← c44660] at e_711_2
  have e_711_3 := arithEq_of_rows h (row := 711) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_3
  simp only [← c44661, ← c44662, k2, ← c44663] at e_711_3
  have hr := e_711_3
  simp only [e_711_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 711 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4557)) := by
  have c44664 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44665 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44666 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44667 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44668 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44669 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_711_4 := arithEq_of_rows h (row := 711) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_4
  simp only [← c44664, ← c44665, ← c44666] at e_711_4
  have e_711_5 := arithEq_of_rows h (row := 711) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_5
  simp only [← c44667, ← c44668, k2, ← c44669] at e_711_5
  have hr := e_711_5
  simp only [e_711_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 711 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4558)) := by
  have c44670 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44671 := (publicBatchWrapper89_copies1395 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44672 := (publicBatchWrapper89_copies1396 a h).1
  have c44673 := (publicBatchWrapper89_copies1396 a h).2.1
  have c44674 := (publicBatchWrapper89_copies1396 a h).2.2.1
  have c44675 := (publicBatchWrapper89_copies1396 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_711_6 := arithEq_of_rows h (row := 711) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_6
  simp only [← c44670, ← c44671, ← c44672] at e_711_6
  have e_711_7 := arithEq_of_rows h (row := 711) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_7
  simp only [← c44673, ← c44674, k2, ← c44675] at e_711_7
  have hr := e_711_7
  simp only [e_711_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 711 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4559)) := by
  have c44676 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.1
  have c44677 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.1
  have c44678 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.1
  have c44679 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.1
  have c44680 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.1
  have c44681 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_711_8 := arithEq_of_rows h (row := 711) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_8
  simp only [← c44676, ← c44677, ← c44678] at e_711_8
  have e_711_9 := arithEq_of_rows h (row := 711) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_9
  simp only [← c44679, ← c44680, k2, ← c44681] at e_711_9
  have hr := e_711_9
  simp only [e_711_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 711 47) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4560)) := by
  have c44682 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44683 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44684 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44685 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44686 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44687 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_711_10 := arithEq_of_rows h (row := 711) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_10
  simp only [← c44682, ← c44683, ← c44684] at e_711_10
  have e_711_11 := arithEq_of_rows h (row := 711) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_11
  simp only [← c44685, ← c44686, k2, ← c44687] at e_711_11
  have hr := e_711_11
  simp only [e_711_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 711 55) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4561)) := by
  have c44688 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44689 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44690 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44691 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44692 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44693 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_711_12 := arithEq_of_rows h (row := 711) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_12
  simp only [← c44688, ← c44689, ← c44690] at e_711_12
  have e_711_13 := arithEq_of_rows h (row := 711) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_13
  simp only [← c44691, ← c44692, k2, ← c44693] at e_711_13
  have hr := e_711_13
  simp only [e_711_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 711 63) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4562)) := by
  have c44694 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44695 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44696 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44697 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44698 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44699 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_711_14 := arithEq_of_rows h (row := 711) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_14
  simp only [← c44694, ← c44695, ← c44696] at e_711_14
  have e_711_15 := arithEq_of_rows h (row := 711) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_15
  simp only [← c44697, ← c44698, k2, ← c44699] at e_711_15
  have hr := e_711_15
  simp only [e_711_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 711 71) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4563)) := by
  have c44700 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44701 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44702 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44703 := (publicBatchWrapper89_copies1396 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44704 := (publicBatchWrapper89_copies1397 a h).1
  have c44705 := (publicBatchWrapper89_copies1397 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_711_16 := arithEq_of_rows h (row := 711) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_16
  simp only [← c44700, ← c44701, ← c44702] at e_711_16
  have e_711_17 := arithEq_of_rows h (row := 711) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_17
  simp only [← c44703, ← c44704, k2, ← c44705] at e_711_17
  have hr := e_711_17
  simp only [e_711_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 711 79) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4564)) := by
  have c44706 := (publicBatchWrapper89_copies1397 a h).2.2.1
  have c44707 := (publicBatchWrapper89_copies1397 a h).2.2.2.1
  have c44708 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.1
  have c44709 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.1
  have c44710 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.1
  have c44711 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_711_18 := arithEq_of_rows h (row := 711) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_18
  simp only [← c44706, ← c44707, ← c44708] at e_711_18
  have e_711_19 := arithEq_of_rows h (row := 711) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_711_19
  simp only [← c44709, ← c44710, k2, ← c44711] at e_711_19
  have hr := e_711_19
  simp only [e_711_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 712 7) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4565)) := by
  have c44712 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.1
  have c44713 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.1
  have c44714 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44715 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44716 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44717 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_712_0 := arithEq_of_rows h (row := 712) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_0
  simp only [← c44712, ← c44713, ← c44714] at e_712_0
  have e_712_1 := arithEq_of_rows h (row := 712) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_1
  simp only [← c44715, ← c44716, k2, ← c44717] at e_712_1
  have hr := e_712_1
  simp only [e_712_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 712 15) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4566)) := by
  have c44718 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44719 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44720 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44721 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44722 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44723 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_712_2 := arithEq_of_rows h (row := 712) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_2
  simp only [← c44718, ← c44719, ← c44720] at e_712_2
  have e_712_3 := arithEq_of_rows h (row := 712) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_3
  simp only [← c44721, ← c44722, k2, ← c44723] at e_712_3
  have hr := e_712_3
  simp only [e_712_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 712 23) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4567)) := by
  have c44724 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44725 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44726 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44727 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44728 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44729 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_712_4 := arithEq_of_rows h (row := 712) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_4
  simp only [← c44724, ← c44725, ← c44726] at e_712_4
  have e_712_5 := arithEq_of_rows h (row := 712) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_5
  simp only [← c44727, ← c44728, k2, ← c44729] at e_712_5
  have hr := e_712_5
  simp only [e_712_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 712 31) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4568)) := by
  have c44730 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44731 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44732 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44733 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44734 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44735 := (publicBatchWrapper89_copies1397 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_712_6 := arithEq_of_rows h (row := 712) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_6
  simp only [← c44730, ← c44731, ← c44732] at e_712_6
  have e_712_7 := arithEq_of_rows h (row := 712) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_7
  simp only [← c44733, ← c44734, k2, ← c44735] at e_712_7
  have hr := e_712_7
  simp only [e_712_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 712 39) = bselect (a (.wire 22 67)) (a (.virt 200)) (a (.virt 4569)) := by
  have c44736 := (publicBatchWrapper89_copies1398 a h).1
  have c44737 := (publicBatchWrapper89_copies1398 a h).2.1
  have c44738 := (publicBatchWrapper89_copies1398 a h).2.2.1
  have c44739 := (publicBatchWrapper89_copies1398 a h).2.2.2.1
  have c44740 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.1
  have c44741 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_712_8 := arithEq_of_rows h (row := 712) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_8
  simp only [← c44736, ← c44737, ← c44738] at e_712_8
  have e_712_9 := arithEq_of_rows h (row := 712) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_9
  simp only [← c44739, ← c44740, k2, ← c44741] at e_712_9
  have hr := e_712_9
  simp only [e_712_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 712 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4610)) := by
  have c44742 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.1
  have c44743 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.1
  have c44744 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.1
  have c44745 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.1
  have c44746 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44747 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_712_10 := arithEq_of_rows h (row := 712) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_10
  simp only [← c44742, ← c44743, ← c44744] at e_712_10
  have e_712_11 := arithEq_of_rows h (row := 712) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_11
  simp only [← c44745, ← c44746, k2, ← c44747] at e_712_11
  have hr := e_712_11
  simp only [e_712_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 712 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4611)) := by
  have c44748 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44749 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44750 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44751 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44752 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44753 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_712_12 := arithEq_of_rows h (row := 712) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_12
  simp only [← c44748, ← c44749, ← c44750] at e_712_12
  have e_712_13 := arithEq_of_rows h (row := 712) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_13
  simp only [← c44751, ← c44752, k2, ← c44753] at e_712_13
  have hr := e_712_13
  simp only [e_712_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 712 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4612)) := by
  have c44754 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44755 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44756 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44757 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44758 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44759 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_712_14 := arithEq_of_rows h (row := 712) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_14
  simp only [← c44754, ← c44755, ← c44756] at e_712_14
  have e_712_15 := arithEq_of_rows h (row := 712) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_15
  simp only [← c44757, ← c44758, k2, ← c44759] at e_712_15
  have hr := e_712_15
  simp only [e_712_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 712 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4613)) := by
  have c44760 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44761 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44762 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44763 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44764 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44765 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_712_16 := arithEq_of_rows h (row := 712) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_16
  simp only [← c44760, ← c44761, ← c44762] at e_712_16
  have e_712_17 := arithEq_of_rows h (row := 712) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_17
  simp only [← c44763, ← c44764, k2, ← c44765] at e_712_17
  have hr := e_712_17
  simp only [e_712_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 712 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4614)) := by
  have c44766 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44767 := (publicBatchWrapper89_copies1398 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44768 := (publicBatchWrapper89_copies1399 a h).1
  have c44769 := (publicBatchWrapper89_copies1399 a h).2.1
  have c44770 := (publicBatchWrapper89_copies1399 a h).2.2.1
  have c44771 := (publicBatchWrapper89_copies1399 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_712_18 := arithEq_of_rows h (row := 712) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_18
  simp only [← c44766, ← c44767, ← c44768] at e_712_18
  have e_712_19 := arithEq_of_rows h (row := 712) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_712_19
  simp only [← c44769, ← c44770, k2, ← c44771] at e_712_19
  have hr := e_712_19
  simp only [e_712_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 713 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4615)) := by
  have c44772 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.1
  have c44773 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.1
  have c44774 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.1
  have c44775 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.1
  have c44776 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.1
  have c44777 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_713_0 := arithEq_of_rows h (row := 713) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_0
  simp only [← c44772, ← c44773, ← c44774] at e_713_0
  have e_713_1 := arithEq_of_rows h (row := 713) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_1
  simp only [← c44775, ← c44776, k2, ← c44777] at e_713_1
  have hr := e_713_1
  simp only [e_713_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 713 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4616)) := by
  have c44778 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44779 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44780 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44781 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44782 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44783 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_713_2 := arithEq_of_rows h (row := 713) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_2
  simp only [← c44778, ← c44779, ← c44780] at e_713_2
  have e_713_3 := arithEq_of_rows h (row := 713) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_3
  simp only [← c44781, ← c44782, k2, ← c44783] at e_713_3
  have hr := e_713_3
  simp only [e_713_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 713 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4617)) := by
  have c44784 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44785 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44786 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44787 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44788 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44789 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_713_4 := arithEq_of_rows h (row := 713) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_4
  simp only [← c44784, ← c44785, ← c44786] at e_713_4
  have e_713_5 := arithEq_of_rows h (row := 713) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_5
  simp only [← c44787, ← c44788, k2, ← c44789] at e_713_5
  have hr := e_713_5
  simp only [e_713_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 713 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4618)) := by
  have c44790 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44791 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44792 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44793 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44794 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44795 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_713_6 := arithEq_of_rows h (row := 713) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_6
  simp only [← c44790, ← c44791, ← c44792] at e_713_6
  have e_713_7 := arithEq_of_rows h (row := 713) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_7
  simp only [← c44793, ← c44794, k2, ← c44795] at e_713_7
  have hr := e_713_7
  simp only [e_713_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 713 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4619)) := by
  have c44796 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44797 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44798 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44799 := (publicBatchWrapper89_copies1399 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44800 := (publicBatchWrapper89_copies1400 a h).1
  have c44801 := (publicBatchWrapper89_copies1400 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_713_8 := arithEq_of_rows h (row := 713) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_8
  simp only [← c44796, ← c44797, ← c44798] at e_713_8
  have e_713_9 := arithEq_of_rows h (row := 713) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_9
  simp only [← c44799, ← c44800, k2, ← c44801] at e_713_9
  have hr := e_713_9
  simp only [e_713_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 713 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4620)) := by
  have c44802 := (publicBatchWrapper89_copies1400 a h).2.2.1
  have c44803 := (publicBatchWrapper89_copies1400 a h).2.2.2.1
  have c44804 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.1
  have c44805 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.1
  have c44806 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.1
  have c44807 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_713_10 := arithEq_of_rows h (row := 713) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_10
  simp only [← c44802, ← c44803, ← c44804] at e_713_10
  have e_713_11 := arithEq_of_rows h (row := 713) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_11
  simp only [← c44805, ← c44806, k2, ← c44807] at e_713_11
  have hr := e_713_11
  simp only [e_713_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 713 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4621)) := by
  have c44808 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.1
  have c44809 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.1
  have c44810 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44811 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44812 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44813 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_713_12 := arithEq_of_rows h (row := 713) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_12
  simp only [← c44808, ← c44809, ← c44810] at e_713_12
  have e_713_13 := arithEq_of_rows h (row := 713) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_13
  simp only [← c44811, ← c44812, k2, ← c44813] at e_713_13
  have hr := e_713_13
  simp only [e_713_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 713 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4622)) := by
  have c44814 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44815 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44816 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44817 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44818 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44819 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_713_14 := arithEq_of_rows h (row := 713) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_14
  simp only [← c44814, ← c44815, ← c44816] at e_713_14
  have e_713_15 := arithEq_of_rows h (row := 713) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_15
  simp only [← c44817, ← c44818, k2, ← c44819] at e_713_15
  have hr := e_713_15
  simp only [e_713_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 713 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4623)) := by
  have c44820 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44821 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44822 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44823 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44824 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44825 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_713_16 := arithEq_of_rows h (row := 713) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_16
  simp only [← c44820, ← c44821, ← c44822] at e_713_16
  have e_713_17 := arithEq_of_rows h (row := 713) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_17
  simp only [← c44823, ← c44824, k2, ← c44825] at e_713_17
  have hr := e_713_17
  simp only [e_713_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 713 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4624)) := by
  have c44826 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44827 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44828 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44829 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44830 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44831 := (publicBatchWrapper89_copies1400 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_713_18 := arithEq_of_rows h (row := 713) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_18
  simp only [← c44826, ← c44827, ← c44828] at e_713_18
  have e_713_19 := arithEq_of_rows h (row := 713) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_713_19
  simp only [← c44829, ← c44830, k2, ← c44831] at e_713_19
  have hr := e_713_19
  simp only [e_713_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 714 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4625)) := by
  have c44832 := (publicBatchWrapper89_copies1401 a h).1
  have c44833 := (publicBatchWrapper89_copies1401 a h).2.1
  have c44834 := (publicBatchWrapper89_copies1401 a h).2.2.1
  have c44835 := (publicBatchWrapper89_copies1401 a h).2.2.2.1
  have c44836 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.1
  have c44837 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_714_0 := arithEq_of_rows h (row := 714) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_0
  simp only [← c44832, ← c44833, ← c44834] at e_714_0
  have e_714_1 := arithEq_of_rows h (row := 714) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_1
  simp only [← c44835, ← c44836, k2, ← c44837] at e_714_1
  have hr := e_714_1
  simp only [e_714_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 714 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4626)) := by
  have c44838 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.1
  have c44839 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.1
  have c44840 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.1
  have c44841 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.1
  have c44842 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44843 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_714_2 := arithEq_of_rows h (row := 714) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_2
  simp only [← c44838, ← c44839, ← c44840] at e_714_2
  have e_714_3 := arithEq_of_rows h (row := 714) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_3
  simp only [← c44841, ← c44842, k2, ← c44843] at e_714_3
  have hr := e_714_3
  simp only [e_714_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 714 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4627)) := by
  have c44844 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44845 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44846 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44847 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44848 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44849 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_714_4 := arithEq_of_rows h (row := 714) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_4
  simp only [← c44844, ← c44845, ← c44846] at e_714_4
  have e_714_5 := arithEq_of_rows h (row := 714) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_5
  simp only [← c44847, ← c44848, k2, ← c44849] at e_714_5
  have hr := e_714_5
  simp only [e_714_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 714 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4628)) := by
  have c44850 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44851 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44852 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44853 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44854 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44855 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_714_6 := arithEq_of_rows h (row := 714) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_6
  simp only [← c44850, ← c44851, ← c44852] at e_714_6
  have e_714_7 := arithEq_of_rows h (row := 714) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_7
  simp only [← c44853, ← c44854, k2, ← c44855] at e_714_7
  have hr := e_714_7
  simp only [e_714_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 714 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4629)) := by
  have c44856 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44857 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44858 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44859 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44860 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44861 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_714_8 := arithEq_of_rows h (row := 714) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_8
  simp only [← c44856, ← c44857, ← c44858] at e_714_8
  have e_714_9 := arithEq_of_rows h (row := 714) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_9
  simp only [← c44859, ← c44860, k2, ← c44861] at e_714_9
  have hr := e_714_9
  simp only [e_714_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 714 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4630)) := by
  have c44862 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44863 := (publicBatchWrapper89_copies1401 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44864 := (publicBatchWrapper89_copies1402 a h).1
  have c44865 := (publicBatchWrapper89_copies1402 a h).2.1
  have c44866 := (publicBatchWrapper89_copies1402 a h).2.2.1
  have c44867 := (publicBatchWrapper89_copies1402 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_714_10 := arithEq_of_rows h (row := 714) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_10
  simp only [← c44862, ← c44863, ← c44864] at e_714_10
  have e_714_11 := arithEq_of_rows h (row := 714) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_11
  simp only [← c44865, ← c44866, k2, ← c44867] at e_714_11
  have hr := e_714_11
  simp only [e_714_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 714 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4631)) := by
  have c44868 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.1
  have c44869 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.1
  have c44870 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.1
  have c44871 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.1
  have c44872 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.1
  have c44873 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_714_12 := arithEq_of_rows h (row := 714) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_12
  simp only [← c44868, ← c44869, ← c44870] at e_714_12
  have e_714_13 := arithEq_of_rows h (row := 714) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_13
  simp only [← c44871, ← c44872, k2, ← c44873] at e_714_13
  have hr := e_714_13
  simp only [e_714_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 714 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4632)) := by
  have c44874 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44875 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44876 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44877 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44878 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44879 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_714_14 := arithEq_of_rows h (row := 714) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_14
  simp only [← c44874, ← c44875, ← c44876] at e_714_14
  have e_714_15 := arithEq_of_rows h (row := 714) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_15
  simp only [← c44877, ← c44878, k2, ← c44879] at e_714_15
  have hr := e_714_15
  simp only [e_714_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 714 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4633)) := by
  have c44880 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44881 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44882 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44883 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44884 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44885 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_714_16 := arithEq_of_rows h (row := 714) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_16
  simp only [← c44880, ← c44881, ← c44882] at e_714_16
  have e_714_17 := arithEq_of_rows h (row := 714) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_17
  simp only [← c44883, ← c44884, k2, ← c44885] at e_714_17
  have hr := e_714_17
  simp only [e_714_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 714 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4634)) := by
  have c44886 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44887 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44888 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44889 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44890 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44891 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_714_18 := arithEq_of_rows h (row := 714) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_18
  simp only [← c44886, ← c44887, ← c44888] at e_714_18
  have e_714_19 := arithEq_of_rows h (row := 714) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_714_19
  simp only [← c44889, ← c44890, k2, ← c44891] at e_714_19
  have hr := e_714_19
  simp only [e_714_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 715 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4635)) := by
  have c44892 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44893 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44894 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44895 := (publicBatchWrapper89_copies1402 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44896 := (publicBatchWrapper89_copies1403 a h).1
  have c44897 := (publicBatchWrapper89_copies1403 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_715_0 := arithEq_of_rows h (row := 715) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_0
  simp only [← c44892, ← c44893, ← c44894] at e_715_0
  have e_715_1 := arithEq_of_rows h (row := 715) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_1
  simp only [← c44895, ← c44896, k2, ← c44897] at e_715_1
  have hr := e_715_1
  simp only [e_715_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 715 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4636)) := by
  have c44898 := (publicBatchWrapper89_copies1403 a h).2.2.1
  have c44899 := (publicBatchWrapper89_copies1403 a h).2.2.2.1
  have c44900 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.1
  have c44901 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.1
  have c44902 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.1
  have c44903 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_715_2 := arithEq_of_rows h (row := 715) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_2
  simp only [← c44898, ← c44899, ← c44900] at e_715_2
  have e_715_3 := arithEq_of_rows h (row := 715) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_3
  simp only [← c44901, ← c44902, k2, ← c44903] at e_715_3
  have hr := e_715_3
  simp only [e_715_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 715 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4637)) := by
  have c44904 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.1
  have c44905 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.1
  have c44906 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44907 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44908 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44909 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_715_4 := arithEq_of_rows h (row := 715) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_4
  simp only [← c44904, ← c44905, ← c44906] at e_715_4
  have e_715_5 := arithEq_of_rows h (row := 715) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_5
  simp only [← c44907, ← c44908, k2, ← c44909] at e_715_5
  have hr := e_715_5
  simp only [e_715_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 715 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4638)) := by
  have c44910 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44911 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44912 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44913 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44914 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44915 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_715_6 := arithEq_of_rows h (row := 715) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_6
  simp only [← c44910, ← c44911, ← c44912] at e_715_6
  have e_715_7 := arithEq_of_rows h (row := 715) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_7
  simp only [← c44913, ← c44914, k2, ← c44915] at e_715_7
  have hr := e_715_7
  simp only [e_715_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 715 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4639)) := by
  have c44916 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44917 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44918 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44919 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44920 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44921 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_715_8 := arithEq_of_rows h (row := 715) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_8
  simp only [← c44916, ← c44917, ← c44918] at e_715_8
  have e_715_9 := arithEq_of_rows h (row := 715) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_9
  simp only [← c44919, ← c44920, k2, ← c44921] at e_715_9
  have hr := e_715_9
  simp only [e_715_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 715 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4640)) := by
  have c44922 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44923 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44924 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44925 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44926 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44927 := (publicBatchWrapper89_copies1403 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_715_10 := arithEq_of_rows h (row := 715) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_10
  simp only [← c44922, ← c44923, ← c44924] at e_715_10
  have e_715_11 := arithEq_of_rows h (row := 715) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_11
  simp only [← c44925, ← c44926, k2, ← c44927] at e_715_11
  have hr := e_715_11
  simp only [e_715_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 715 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4641)) := by
  have c44928 := (publicBatchWrapper89_copies1404 a h).1
  have c44929 := (publicBatchWrapper89_copies1404 a h).2.1
  have c44930 := (publicBatchWrapper89_copies1404 a h).2.2.1
  have c44931 := (publicBatchWrapper89_copies1404 a h).2.2.2.1
  have c44932 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.1
  have c44933 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_715_12 := arithEq_of_rows h (row := 715) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_12
  simp only [← c44928, ← c44929, ← c44930] at e_715_12
  have e_715_13 := arithEq_of_rows h (row := 715) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_13
  simp only [← c44931, ← c44932, k2, ← c44933] at e_715_13
  have hr := e_715_13
  simp only [e_715_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 715 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4642)) := by
  have c44934 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.1
  have c44935 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.1
  have c44936 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.1
  have c44937 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.1
  have c44938 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44939 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_715_14 := arithEq_of_rows h (row := 715) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_14
  simp only [← c44934, ← c44935, ← c44936] at e_715_14
  have e_715_15 := arithEq_of_rows h (row := 715) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_15
  simp only [← c44937, ← c44938, k2, ← c44939] at e_715_15
  have hr := e_715_15
  simp only [e_715_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 715 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4643)) := by
  have c44940 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44941 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44942 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44943 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44944 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44945 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_715_16 := arithEq_of_rows h (row := 715) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_16
  simp only [← c44940, ← c44941, ← c44942] at e_715_16
  have e_715_17 := arithEq_of_rows h (row := 715) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_17
  simp only [← c44943, ← c44944, k2, ← c44945] at e_715_17
  have hr := e_715_17
  simp only [e_715_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 715 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4644)) := by
  have c44946 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44947 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44948 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44949 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44950 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44951 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_715_18 := arithEq_of_rows h (row := 715) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_18
  simp only [← c44946, ← c44947, ← c44948] at e_715_18
  have e_715_19 := arithEq_of_rows h (row := 715) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_715_19
  simp only [← c44949, ← c44950, k2, ← c44951] at e_715_19
  have hr := e_715_19
  simp only [e_715_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 716 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4645)) := by
  have c44952 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44953 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44954 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44955 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44956 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44957 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_716_0 := arithEq_of_rows h (row := 716) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_0
  simp only [← c44952, ← c44953, ← c44954] at e_716_0
  have e_716_1 := arithEq_of_rows h (row := 716) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_1
  simp only [← c44955, ← c44956, k2, ← c44957] at e_716_1
  have hr := e_716_1
  simp only [e_716_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 716 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4646)) := by
  have c44958 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44959 := (publicBatchWrapper89_copies1404 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44960 := (publicBatchWrapper89_copies1405 a h).1
  have c44961 := (publicBatchWrapper89_copies1405 a h).2.1
  have c44962 := (publicBatchWrapper89_copies1405 a h).2.2.1
  have c44963 := (publicBatchWrapper89_copies1405 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_716_2 := arithEq_of_rows h (row := 716) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_2
  simp only [← c44958, ← c44959, ← c44960] at e_716_2
  have e_716_3 := arithEq_of_rows h (row := 716) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_3
  simp only [← c44961, ← c44962, k2, ← c44963] at e_716_3
  have hr := e_716_3
  simp only [e_716_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 716 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4647)) := by
  have c44964 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.1
  have c44965 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.1
  have c44966 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.1
  have c44967 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.1
  have c44968 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.1
  have c44969 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_716_4 := arithEq_of_rows h (row := 716) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_4
  simp only [← c44964, ← c44965, ← c44966] at e_716_4
  have e_716_5 := arithEq_of_rows h (row := 716) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_5
  simp only [← c44967, ← c44968, k2, ← c44969] at e_716_5
  have hr := e_716_5
  simp only [e_716_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 716 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4648)) := by
  have c44970 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.1
  have c44971 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c44972 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44973 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44974 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44975 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_716_6 := arithEq_of_rows h (row := 716) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_6
  simp only [← c44970, ← c44971, ← c44972] at e_716_6
  have e_716_7 := arithEq_of_rows h (row := 716) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_7
  simp only [← c44973, ← c44974, k2, ← c44975] at e_716_7
  have hr := e_716_7
  simp only [e_716_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 716 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4649)) := by
  have c44976 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44977 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44978 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44979 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44980 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44981 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_716_8 := arithEq_of_rows h (row := 716) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_8
  simp only [← c44976, ← c44977, ← c44978] at e_716_8
  have e_716_9 := arithEq_of_rows h (row := 716) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_9
  simp only [← c44979, ← c44980, k2, ← c44981] at e_716_9
  have hr := e_716_9
  simp only [e_716_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 716 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4650)) := by
  have c44982 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44983 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44984 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44985 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44986 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44987 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_716_10 := arithEq_of_rows h (row := 716) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_10
  simp only [← c44982, ← c44983, ← c44984] at e_716_10
  have e_716_11 := arithEq_of_rows h (row := 716) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_11
  simp only [← c44985, ← c44986, k2, ← c44987] at e_716_11
  have hr := e_716_11
  simp only [e_716_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 716 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4651)) := by
  have c44988 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44989 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44990 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c44991 := (publicBatchWrapper89_copies1405 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c44992 := (publicBatchWrapper89_copies1406 a h).1
  have c44993 := (publicBatchWrapper89_copies1406 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_716_12 := arithEq_of_rows h (row := 716) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_12
  simp only [← c44988, ← c44989, ← c44990] at e_716_12
  have e_716_13 := arithEq_of_rows h (row := 716) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_13
  simp only [← c44991, ← c44992, k2, ← c44993] at e_716_13
  have hr := e_716_13
  simp only [e_716_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 716 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4652)) := by
  have c44994 := (publicBatchWrapper89_copies1406 a h).2.2.1
  have c44995 := (publicBatchWrapper89_copies1406 a h).2.2.2.1
  have c44996 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.1
  have c44997 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.1
  have c44998 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.1
  have c44999 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_716_14 := arithEq_of_rows h (row := 716) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_14
  simp only [← c44994, ← c44995, ← c44996] at e_716_14
  have e_716_15 := arithEq_of_rows h (row := 716) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_15
  simp only [← c44997, ← c44998, k2, ← c44999] at e_716_15
  have hr := e_716_15
  simp only [e_716_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 716 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4653)) := by
  have c45000 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.1
  have c45001 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.1
  have c45002 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45003 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45004 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45005 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_716_16 := arithEq_of_rows h (row := 716) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_16
  simp only [← c45000, ← c45001, ← c45002] at e_716_16
  have e_716_17 := arithEq_of_rows h (row := 716) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_17
  simp only [← c45003, ← c45004, k2, ← c45005] at e_716_17
  have hr := e_716_17
  simp only [e_716_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 716 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4654)) := by
  have c45006 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45007 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45008 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45009 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45010 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45011 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_716_18 := arithEq_of_rows h (row := 716) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_18
  simp only [← c45006, ← c45007, ← c45008] at e_716_18
  have e_716_19 := arithEq_of_rows h (row := 716) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_716_19
  simp only [← c45009, ← c45010, k2, ← c45011] at e_716_19
  have hr := e_716_19
  simp only [e_716_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 717 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4655)) := by
  have c45012 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45013 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45014 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45015 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45016 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45017 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_717_0 := arithEq_of_rows h (row := 717) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_0
  simp only [← c45012, ← c45013, ← c45014] at e_717_0
  have e_717_1 := arithEq_of_rows h (row := 717) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_1
  simp only [← c45015, ← c45016, k2, ← c45017] at e_717_1
  have hr := e_717_1
  simp only [e_717_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 717 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4656)) := by
  have c45018 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45019 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45020 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45021 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45022 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45023 := (publicBatchWrapper89_copies1406 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_717_2 := arithEq_of_rows h (row := 717) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_2
  simp only [← c45018, ← c45019, ← c45020] at e_717_2
  have e_717_3 := arithEq_of_rows h (row := 717) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_3
  simp only [← c45021, ← c45022, k2, ← c45023] at e_717_3
  have hr := e_717_3
  simp only [e_717_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 717 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4657)) := by
  have c45024 := (publicBatchWrapper89_copies1407 a h).1
  have c45025 := (publicBatchWrapper89_copies1407 a h).2.1
  have c45026 := (publicBatchWrapper89_copies1407 a h).2.2.1
  have c45027 := (publicBatchWrapper89_copies1407 a h).2.2.2.1
  have c45028 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.1
  have c45029 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_717_4 := arithEq_of_rows h (row := 717) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_4
  simp only [← c45024, ← c45025, ← c45026] at e_717_4
  have e_717_5 := arithEq_of_rows h (row := 717) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_5
  simp only [← c45027, ← c45028, k2, ← c45029] at e_717_5
  have hr := e_717_5
  simp only [e_717_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 717 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4658)) := by
  have c45030 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.1
  have c45031 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.1
  have c45032 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.1
  have c45033 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.1
  have c45034 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45035 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_717_6 := arithEq_of_rows h (row := 717) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_6
  simp only [← c45030, ← c45031, ← c45032] at e_717_6
  have e_717_7 := arithEq_of_rows h (row := 717) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_7
  simp only [← c45033, ← c45034, k2, ← c45035] at e_717_7
  have hr := e_717_7
  simp only [e_717_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 717 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4659)) := by
  have c45036 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45037 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45038 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45039 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45040 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45041 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_717_8 := arithEq_of_rows h (row := 717) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_8
  simp only [← c45036, ← c45037, ← c45038] at e_717_8
  have e_717_9 := arithEq_of_rows h (row := 717) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_9
  simp only [← c45039, ← c45040, k2, ← c45041] at e_717_9
  have hr := e_717_9
  simp only [e_717_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 717 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4660)) := by
  have c45042 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45043 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45044 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45045 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45046 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45047 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_717_10 := arithEq_of_rows h (row := 717) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_10
  simp only [← c45042, ← c45043, ← c45044] at e_717_10
  have e_717_11 := arithEq_of_rows h (row := 717) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_11
  simp only [← c45045, ← c45046, k2, ← c45047] at e_717_11
  have hr := e_717_11
  simp only [e_717_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 717 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4661)) := by
  have c45048 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45049 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45050 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45051 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45052 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45053 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_717_12 := arithEq_of_rows h (row := 717) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_12
  simp only [← c45048, ← c45049, ← c45050] at e_717_12
  have e_717_13 := arithEq_of_rows h (row := 717) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_13
  simp only [← c45051, ← c45052, k2, ← c45053] at e_717_13
  have hr := e_717_13
  simp only [e_717_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 717 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4662)) := by
  have c45054 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45055 := (publicBatchWrapper89_copies1407 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45056 := (publicBatchWrapper89_copies1408 a h).1
  have c45057 := (publicBatchWrapper89_copies1408 a h).2.1
  have c45058 := (publicBatchWrapper89_copies1408 a h).2.2.1
  have c45059 := (publicBatchWrapper89_copies1408 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_717_14 := arithEq_of_rows h (row := 717) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_14
  simp only [← c45054, ← c45055, ← c45056] at e_717_14
  have e_717_15 := arithEq_of_rows h (row := 717) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_15
  simp only [← c45057, ← c45058, k2, ← c45059] at e_717_15
  have hr := e_717_15
  simp only [e_717_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 717 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4663)) := by
  have c45060 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.1
  have c45061 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.1
  have c45062 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.1
  have c45063 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.1
  have c45064 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.1
  have c45065 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_717_16 := arithEq_of_rows h (row := 717) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_16
  simp only [← c45060, ← c45061, ← c45062] at e_717_16
  have e_717_17 := arithEq_of_rows h (row := 717) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_17
  simp only [← c45063, ← c45064, k2, ← c45065] at e_717_17
  have hr := e_717_17
  simp only [e_717_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 717 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4664)) := by
  have c45066 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45067 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45068 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45069 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45070 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45071 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_717_18 := arithEq_of_rows h (row := 717) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_18
  simp only [← c45066, ← c45067, ← c45068] at e_717_18
  have e_717_19 := arithEq_of_rows h (row := 717) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_717_19
  simp only [← c45069, ← c45070, k2, ← c45071] at e_717_19
  have hr := e_717_19
  simp only [e_717_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 718 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4665)) := by
  have c45072 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45073 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45074 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45075 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45076 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45077 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_718_0 := arithEq_of_rows h (row := 718) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_0
  simp only [← c45072, ← c45073, ← c45074] at e_718_0
  have e_718_1 := arithEq_of_rows h (row := 718) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_1
  simp only [← c45075, ← c45076, k2, ← c45077] at e_718_1
  have hr := e_718_1
  simp only [e_718_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 718 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4666)) := by
  have c45078 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45079 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45080 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45081 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45082 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45083 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_718_2 := arithEq_of_rows h (row := 718) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_2
  simp only [← c45078, ← c45079, ← c45080] at e_718_2
  have e_718_3 := arithEq_of_rows h (row := 718) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_3
  simp only [← c45081, ← c45082, k2, ← c45083] at e_718_3
  have hr := e_718_3
  simp only [e_718_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 718 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4667)) := by
  have c45084 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45085 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45086 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45087 := (publicBatchWrapper89_copies1408 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45088 := (publicBatchWrapper89_copies1409 a h).1
  have c45089 := (publicBatchWrapper89_copies1409 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_718_4 := arithEq_of_rows h (row := 718) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_4
  simp only [← c45084, ← c45085, ← c45086] at e_718_4
  have e_718_5 := arithEq_of_rows h (row := 718) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_5
  simp only [← c45087, ← c45088, k2, ← c45089] at e_718_5
  have hr := e_718_5
  simp only [e_718_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 718 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4668)) := by
  have c45090 := (publicBatchWrapper89_copies1409 a h).2.2.1
  have c45091 := (publicBatchWrapper89_copies1409 a h).2.2.2.1
  have c45092 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.1
  have c45093 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.1
  have c45094 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.1
  have c45095 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_718_6 := arithEq_of_rows h (row := 718) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_6
  simp only [← c45090, ← c45091, ← c45092] at e_718_6
  have e_718_7 := arithEq_of_rows h (row := 718) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_7
  simp only [← c45093, ← c45094, k2, ← c45095] at e_718_7
  have hr := e_718_7
  simp only [e_718_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 718 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4669)) := by
  have c45096 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.1
  have c45097 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.1
  have c45098 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45099 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45100 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45101 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_718_8 := arithEq_of_rows h (row := 718) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_8
  simp only [← c45096, ← c45097, ← c45098] at e_718_8
  have e_718_9 := arithEq_of_rows h (row := 718) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_9
  simp only [← c45099, ← c45100, k2, ← c45101] at e_718_9
  have hr := e_718_9
  simp only [e_718_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 718 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4670)) := by
  have c45102 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45103 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45104 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45105 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45106 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45107 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_718_10 := arithEq_of_rows h (row := 718) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_10
  simp only [← c45102, ← c45103, ← c45104] at e_718_10
  have e_718_11 := arithEq_of_rows h (row := 718) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_11
  simp only [← c45105, ← c45106, k2, ← c45107] at e_718_11
  have hr := e_718_11
  simp only [e_718_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 718 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4671)) := by
  have c45108 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45109 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45110 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45111 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45112 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45113 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_718_12 := arithEq_of_rows h (row := 718) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_12
  simp only [← c45108, ← c45109, ← c45110] at e_718_12
  have e_718_13 := arithEq_of_rows h (row := 718) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_13
  simp only [← c45111, ← c45112, k2, ← c45113] at e_718_13
  have hr := e_718_13
  simp only [e_718_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 718 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4672)) := by
  have c45114 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45115 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45116 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45117 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45118 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45119 := (publicBatchWrapper89_copies1409 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_718_14 := arithEq_of_rows h (row := 718) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_14
  simp only [← c45114, ← c45115, ← c45116] at e_718_14
  have e_718_15 := arithEq_of_rows h (row := 718) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_15
  simp only [← c45117, ← c45118, k2, ← c45119] at e_718_15
  have hr := e_718_15
  simp only [e_718_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 718 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4673)) := by
  have c45120 := (publicBatchWrapper89_copies1410 a h).1
  have c45121 := (publicBatchWrapper89_copies1410 a h).2.1
  have c45122 := (publicBatchWrapper89_copies1410 a h).2.2.1
  have c45123 := (publicBatchWrapper89_copies1410 a h).2.2.2.1
  have c45124 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.1
  have c45125 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_718_16 := arithEq_of_rows h (row := 718) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_16
  simp only [← c45120, ← c45121, ← c45122] at e_718_16
  have e_718_17 := arithEq_of_rows h (row := 718) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_17
  simp only [← c45123, ← c45124, k2, ← c45125] at e_718_17
  have hr := e_718_17
  simp only [e_718_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 718 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4674)) := by
  have c45126 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.1
  have c45127 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.1
  have c45128 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.1
  have c45129 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.1
  have c45130 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45131 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_718_18 := arithEq_of_rows h (row := 718) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_18
  simp only [← c45126, ← c45127, ← c45128] at e_718_18
  have e_718_19 := arithEq_of_rows h (row := 718) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_718_19
  simp only [← c45129, ← c45130, k2, ← c45131] at e_718_19
  have hr := e_718_19
  simp only [e_718_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 719 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4675)) := by
  have c45132 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45133 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45134 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45135 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45136 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45137 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_719_0 := arithEq_of_rows h (row := 719) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_0
  simp only [← c45132, ← c45133, ← c45134] at e_719_0
  have e_719_1 := arithEq_of_rows h (row := 719) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_1
  simp only [← c45135, ← c45136, k2, ← c45137] at e_719_1
  have hr := e_719_1
  simp only [e_719_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 719 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4676)) := by
  have c45138 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45139 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45140 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45141 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45142 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45143 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_719_2 := arithEq_of_rows h (row := 719) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_2
  simp only [← c45138, ← c45139, ← c45140] at e_719_2
  have e_719_3 := arithEq_of_rows h (row := 719) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_3
  simp only [← c45141, ← c45142, k2, ← c45143] at e_719_3
  have hr := e_719_3
  simp only [e_719_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 719 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4677)) := by
  have c45144 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45145 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45146 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45147 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45148 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45149 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_719_4 := arithEq_of_rows h (row := 719) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_4
  simp only [← c45144, ← c45145, ← c45146] at e_719_4
  have e_719_5 := arithEq_of_rows h (row := 719) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_5
  simp only [← c45147, ← c45148, k2, ← c45149] at e_719_5
  have hr := e_719_5
  simp only [e_719_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 719 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4678)) := by
  have c45150 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45151 := (publicBatchWrapper89_copies1410 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45152 := (publicBatchWrapper89_copies1411 a h).1
  have c45153 := (publicBatchWrapper89_copies1411 a h).2.1
  have c45154 := (publicBatchWrapper89_copies1411 a h).2.2.1
  have c45155 := (publicBatchWrapper89_copies1411 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_719_6 := arithEq_of_rows h (row := 719) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_6
  simp only [← c45150, ← c45151, ← c45152] at e_719_6
  have e_719_7 := arithEq_of_rows h (row := 719) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_7
  simp only [← c45153, ← c45154, k2, ← c45155] at e_719_7
  have hr := e_719_7
  simp only [e_719_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 719 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4679)) := by
  have c45156 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.1
  have c45157 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.1
  have c45158 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.1
  have c45159 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.1
  have c45160 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.1
  have c45161 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_719_8 := arithEq_of_rows h (row := 719) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_8
  simp only [← c45156, ← c45157, ← c45158] at e_719_8
  have e_719_9 := arithEq_of_rows h (row := 719) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_9
  simp only [← c45159, ← c45160, k2, ← c45161] at e_719_9
  have hr := e_719_9
  simp only [e_719_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 719 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4680)) := by
  have c45162 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45163 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45164 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45165 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45166 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45167 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_719_10 := arithEq_of_rows h (row := 719) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_10
  simp only [← c45162, ← c45163, ← c45164] at e_719_10
  have e_719_11 := arithEq_of_rows h (row := 719) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_11
  simp only [← c45165, ← c45166, k2, ← c45167] at e_719_11
  have hr := e_719_11
  simp only [e_719_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 719 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4681)) := by
  have c45168 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45169 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45170 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45171 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45172 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45173 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_719_12 := arithEq_of_rows h (row := 719) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_12
  simp only [← c45168, ← c45169, ← c45170] at e_719_12
  have e_719_13 := arithEq_of_rows h (row := 719) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_13
  simp only [← c45171, ← c45172, k2, ← c45173] at e_719_13
  have hr := e_719_13
  simp only [e_719_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 719 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4682)) := by
  have c45174 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45175 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45176 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45177 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45178 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45179 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_719_14 := arithEq_of_rows h (row := 719) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_14
  simp only [← c45174, ← c45175, ← c45176] at e_719_14
  have e_719_15 := arithEq_of_rows h (row := 719) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_15
  simp only [← c45177, ← c45178, k2, ← c45179] at e_719_15
  have hr := e_719_15
  simp only [e_719_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 719 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4683)) := by
  have c45180 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45181 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45182 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45183 := (publicBatchWrapper89_copies1411 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45184 := (publicBatchWrapper89_copies1412 a h).1
  have c45185 := (publicBatchWrapper89_copies1412 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_719_16 := arithEq_of_rows h (row := 719) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_16
  simp only [← c45180, ← c45181, ← c45182] at e_719_16
  have e_719_17 := arithEq_of_rows h (row := 719) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_17
  simp only [← c45183, ← c45184, k2, ← c45185] at e_719_17
  have hr := e_719_17
  simp only [e_719_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 719 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4684)) := by
  have c45186 := (publicBatchWrapper89_copies1412 a h).2.2.1
  have c45187 := (publicBatchWrapper89_copies1412 a h).2.2.2.1
  have c45188 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.1
  have c45189 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.1
  have c45190 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.1
  have c45191 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_719_18 := arithEq_of_rows h (row := 719) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_18
  simp only [← c45186, ← c45187, ← c45188] at e_719_18
  have e_719_19 := arithEq_of_rows h (row := 719) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_719_19
  simp only [← c45189, ← c45190, k2, ← c45191] at e_719_19
  have hr := e_719_19
  simp only [e_719_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 720 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4685)) := by
  have c45192 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.1
  have c45193 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.1
  have c45194 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45195 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45196 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45197 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_720_0 := arithEq_of_rows h (row := 720) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_0
  simp only [← c45192, ← c45193, ← c45194] at e_720_0
  have e_720_1 := arithEq_of_rows h (row := 720) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_1
  simp only [← c45195, ← c45196, k2, ← c45197] at e_720_1
  have hr := e_720_1
  simp only [e_720_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 720 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4686)) := by
  have c45198 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45199 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45200 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45201 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45202 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45203 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_720_2 := arithEq_of_rows h (row := 720) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_2
  simp only [← c45198, ← c45199, ← c45200] at e_720_2
  have e_720_3 := arithEq_of_rows h (row := 720) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_3
  simp only [← c45201, ← c45202, k2, ← c45203] at e_720_3
  have hr := e_720_3
  simp only [e_720_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 720 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4687)) := by
  have c45204 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45205 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45206 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45207 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45208 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45209 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_720_4 := arithEq_of_rows h (row := 720) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_4
  simp only [← c45204, ← c45205, ← c45206] at e_720_4
  have e_720_5 := arithEq_of_rows h (row := 720) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_5
  simp only [← c45207, ← c45208, k2, ← c45209] at e_720_5
  have hr := e_720_5
  simp only [e_720_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 720 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4688)) := by
  have c45210 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45211 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45212 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45213 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45214 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45215 := (publicBatchWrapper89_copies1412 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_720_6 := arithEq_of_rows h (row := 720) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_6
  simp only [← c45210, ← c45211, ← c45212] at e_720_6
  have e_720_7 := arithEq_of_rows h (row := 720) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_7
  simp only [← c45213, ← c45214, k2, ← c45215] at e_720_7
  have hr := e_720_7
  simp only [e_720_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 720 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4689)) := by
  have c45216 := (publicBatchWrapper89_copies1413 a h).1
  have c45217 := (publicBatchWrapper89_copies1413 a h).2.1
  have c45218 := (publicBatchWrapper89_copies1413 a h).2.2.1
  have c45219 := (publicBatchWrapper89_copies1413 a h).2.2.2.1
  have c45220 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.1
  have c45221 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_720_8 := arithEq_of_rows h (row := 720) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_8
  simp only [← c45216, ← c45217, ← c45218] at e_720_8
  have e_720_9 := arithEq_of_rows h (row := 720) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_9
  simp only [← c45219, ← c45220, k2, ← c45221] at e_720_9
  have hr := e_720_9
  simp only [e_720_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 720 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4690)) := by
  have c45222 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.1
  have c45223 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.1
  have c45224 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.1
  have c45225 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.1
  have c45226 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45227 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_720_10 := arithEq_of_rows h (row := 720) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_10
  simp only [← c45222, ← c45223, ← c45224] at e_720_10
  have e_720_11 := arithEq_of_rows h (row := 720) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_11
  simp only [← c45225, ← c45226, k2, ← c45227] at e_720_11
  have hr := e_720_11
  simp only [e_720_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 720 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4691)) := by
  have c45228 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45229 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45230 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45231 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45232 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45233 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_720_12 := arithEq_of_rows h (row := 720) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_12
  simp only [← c45228, ← c45229, ← c45230] at e_720_12
  have e_720_13 := arithEq_of_rows h (row := 720) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_13
  simp only [← c45231, ← c45232, k2, ← c45233] at e_720_13
  have hr := e_720_13
  simp only [e_720_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 720 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4692)) := by
  have c45234 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45235 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45236 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45237 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45238 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45239 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_720_14 := arithEq_of_rows h (row := 720) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_14
  simp only [← c45234, ← c45235, ← c45236] at e_720_14
  have e_720_15 := arithEq_of_rows h (row := 720) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_15
  simp only [← c45237, ← c45238, k2, ← c45239] at e_720_15
  have hr := e_720_15
  simp only [e_720_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 720 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4693)) := by
  have c45240 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45241 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45242 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45243 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45244 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45245 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_720_16 := arithEq_of_rows h (row := 720) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_16
  simp only [← c45240, ← c45241, ← c45242] at e_720_16
  have e_720_17 := arithEq_of_rows h (row := 720) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_17
  simp only [← c45243, ← c45244, k2, ← c45245] at e_720_17
  have hr := e_720_17
  simp only [e_720_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 720 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4694)) := by
  have c45246 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45247 := (publicBatchWrapper89_copies1413 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45248 := (publicBatchWrapper89_copies1414 a h).1
  have c45249 := (publicBatchWrapper89_copies1414 a h).2.1
  have c45250 := (publicBatchWrapper89_copies1414 a h).2.2.1
  have c45251 := (publicBatchWrapper89_copies1414 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_720_18 := arithEq_of_rows h (row := 720) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_18
  simp only [← c45246, ← c45247, ← c45248] at e_720_18
  have e_720_19 := arithEq_of_rows h (row := 720) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_720_19
  simp only [← c45249, ← c45250, k2, ← c45251] at e_720_19
  have hr := e_720_19
  simp only [e_720_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 721 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4695)) := by
  have c45252 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.1
  have c45253 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.1
  have c45254 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.1
  have c45255 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.1
  have c45256 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.1
  have c45257 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_721_0 := arithEq_of_rows h (row := 721) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_0
  simp only [← c45252, ← c45253, ← c45254] at e_721_0
  have e_721_1 := arithEq_of_rows h (row := 721) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_1
  simp only [← c45255, ← c45256, k2, ← c45257] at e_721_1
  have hr := e_721_1
  simp only [e_721_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 721 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4696)) := by
  have c45258 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45259 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45260 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45261 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45262 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45263 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_721_2 := arithEq_of_rows h (row := 721) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_2
  simp only [← c45258, ← c45259, ← c45260] at e_721_2
  have e_721_3 := arithEq_of_rows h (row := 721) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_3
  simp only [← c45261, ← c45262, k2, ← c45263] at e_721_3
  have hr := e_721_3
  simp only [e_721_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 721 23) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4697)) := by
  have c45264 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45265 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45266 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45267 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45268 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45269 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_721_4 := arithEq_of_rows h (row := 721) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_4
  simp only [← c45264, ← c45265, ← c45266] at e_721_4
  have e_721_5 := arithEq_of_rows h (row := 721) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_5
  simp only [← c45267, ← c45268, k2, ← c45269] at e_721_5
  have hr := e_721_5
  simp only [e_721_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 721 31) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4698)) := by
  have c45270 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45271 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45272 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45273 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45274 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45275 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_721_6 := arithEq_of_rows h (row := 721) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_6
  simp only [← c45270, ← c45271, ← c45272] at e_721_6
  have e_721_7 := arithEq_of_rows h (row := 721) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_7
  simp only [← c45273, ← c45274, k2, ← c45275] at e_721_7
  have hr := e_721_7
  simp only [e_721_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 721 39) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4699)) := by
  have c45276 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45277 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45278 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45279 := (publicBatchWrapper89_copies1414 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c45280 := (publicBatchWrapper89_copies1415 a h).1
  have c45281 := (publicBatchWrapper89_copies1415 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_721_8 := arithEq_of_rows h (row := 721) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_8
  simp only [← c45276, ← c45277, ← c45278] at e_721_8
  have e_721_9 := arithEq_of_rows h (row := 721) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_9
  simp only [← c45279, ← c45280, k2, ← c45281] at e_721_9
  have hr := e_721_9
  simp only [e_721_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 721 47) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4700)) := by
  have c45282 := (publicBatchWrapper89_copies1415 a h).2.2.1
  have c45283 := (publicBatchWrapper89_copies1415 a h).2.2.2.1
  have c45284 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.1
  have c45285 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.1
  have c45286 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.1
  have c45287 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_721_10 := arithEq_of_rows h (row := 721) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_10
  simp only [← c45282, ← c45283, ← c45284] at e_721_10
  have e_721_11 := arithEq_of_rows h (row := 721) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_11
  simp only [← c45285, ← c45286, k2, ← c45287] at e_721_11
  have hr := e_721_11
  simp only [e_721_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 721 55) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4701)) := by
  have c45288 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.1
  have c45289 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.1
  have c45290 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45291 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c45292 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45293 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_721_12 := arithEq_of_rows h (row := 721) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_12
  simp only [← c45288, ← c45289, ← c45290] at e_721_12
  have e_721_13 := arithEq_of_rows h (row := 721) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_13
  simp only [← c45291, ← c45292, k2, ← c45293] at e_721_13
  have hr := e_721_13
  simp only [e_721_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 721 63) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4702)) := by
  have c45294 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45295 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45296 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45297 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45298 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45299 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_721_14 := arithEq_of_rows h (row := 721) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_14
  simp only [← c45294, ← c45295, ← c45296] at e_721_14
  have e_721_15 := arithEq_of_rows h (row := 721) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_15
  simp only [← c45297, ← c45298, k2, ← c45299] at e_721_15
  have hr := e_721_15
  simp only [e_721_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 721 71) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4703)) := by
  have c45300 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45301 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45302 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45303 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45304 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45305 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_721_16 := arithEq_of_rows h (row := 721) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_16
  simp only [← c45300, ← c45301, ← c45302] at e_721_16
  have e_721_17 := arithEq_of_rows h (row := 721) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_17
  simp only [← c45303, ← c45304, k2, ← c45305] at e_721_17
  have hr := e_721_17
  simp only [e_721_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 721 79) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4704)) := by
  have c45306 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45307 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45308 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45309 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45310 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c45311 := (publicBatchWrapper89_copies1415 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_721_18 := arithEq_of_rows h (row := 721) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_18
  simp only [← c45306, ← c45307, ← c45308] at e_721_18
  have e_721_19 := arithEq_of_rows h (row := 721) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_721_19
  simp only [← c45309, ← c45310, k2, ← c45311] at e_721_19
  have hr := e_721_19
  simp only [e_721_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 722 7) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4705)) := by
  have c45312 := (publicBatchWrapper89_copies1416 a h).1
  have c45313 := (publicBatchWrapper89_copies1416 a h).2.1
  have c45314 := (publicBatchWrapper89_copies1416 a h).2.2.1
  have c45315 := (publicBatchWrapper89_copies1416 a h).2.2.2.1
  have c45316 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.1
  have c45317 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_722_0 := arithEq_of_rows h (row := 722) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_0
  simp only [← c45312, ← c45313, ← c45314] at e_722_0
  have e_722_1 := arithEq_of_rows h (row := 722) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_1
  simp only [← c45315, ← c45316, k2, ← c45317] at e_722_1
  have hr := e_722_1
  simp only [e_722_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f6802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 722 15) = bselect (a (.wire 24 31)) (a (.virt 200)) (a (.virt 4706)) := by
  have c45318 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.1
  have c45319 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.1
  have c45320 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.1
  have c45321 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.1
  have c45322 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.1
  have c45323 := (publicBatchWrapper89_copies1416 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_722_2 := arithEq_of_rows h (row := 722) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_2
  simp only [← c45318, ← c45319, ← c45320] at e_722_2
  have e_722_3 := arithEq_of_rows h (row := 722) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_722_3
  simp only [← c45321, ← c45322, k2, ← c45323] at e_722_3
  have hr := e_722_3
  simp only [e_722_2] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
